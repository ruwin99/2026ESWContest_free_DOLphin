from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from common import resolve_path


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CrackDetailHead(nn.Module):
    def __init__(
        self,
        detail_channels: int = 48,
        context_channels: int = 64,
        fusion_channels: int = 96,
    ) -> None:
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(24, detail_channels, 1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(detail_channels, detail_channels),
        )
        self.context = nn.Sequential(
            nn.Conv2d(320, context_channels, 1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(context_channels, context_channels),
        )
        self.fuse = nn.Sequential(
            DepthwiseSeparableBlock(detail_channels + context_channels, fusion_channels),
            DepthwiseSeparableBlock(fusion_channels, fusion_channels),
            nn.Conv2d(fusion_channels, 1, 1),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        detail = self.detail(features["low_level"])
        context = self.context(features["out"])
        context = F.interpolate(
            context, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat((detail, context), dim=1))


class DualHeadMobileNetV2OS8(nn.Module):
    """Exact inherited rust path plus a thin-crack detail head on shared features."""

    def __init__(self, rust_base: nn.Module, student_config: dict[str, Any]) -> None:
        super().__init__()
        # Keep these names identical to the inherited rust checkpoint.
        self.backbone = rust_base.backbone
        self.classifier = rust_base.classifier
        self.crack_head = CrackDetailHead(
            int(student_config["crack_detail_channels"]),
            int(student_config["crack_context_channels"]),
            int(student_config["crack_fusion_channels"]),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output_size = images.shape[-2:]
        features = self.backbone(images)
        rust_logits = self.classifier(features)
        crack_logits = self.crack_head(features)
        rust_logits = F.interpolate(
            rust_logits, size=output_size, mode="bilinear", align_corners=False
        )
        crack_logits = F.interpolate(
            crack_logits, size=output_size, mode="bilinear", align_corners=False
        )
        return torch.cat((rust_logits, crack_logits), dim=1)


def _official_model(config: dict[str, Any]) -> nn.Module:
    official_root = resolve_path(config["paths"]["official_training"])
    if not (official_root / "network" / "modeling.py").is_file():
        raise FileNotFoundError(f"Official MobileNetV2 source missing: {official_root}")
    text = str(official_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    from network.modeling import deeplabv3plus_mobilenet

    return deeplabv3plus_mobilenet(
        num_classes=4, output_stride=8, pretrained_backbone=False
    )


def build_model(config: dict[str, Any]) -> DualHeadMobileNetV2OS8:
    return DualHeadMobileNetV2OS8(_official_model(config), config["student"])


def load_rust_checkpoint_strict(
    model: DualHeadMobileNetV2OS8, checkpoint_path: Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_name = "deeplabv3plus_mobilenet"
    if checkpoint.get("model_name") != expected_name:
        raise ValueError(
            f"Rust checkpoint architecture mismatch: expected={expected_name}, "
            f"actual={checkpoint.get('model_name')}"
        )
    if int(checkpoint.get("num_classes", -1)) != 4 or int(
        checkpoint.get("output_stride", -1)
    ) != 8:
        raise ValueError("Rust checkpoint must have 4 classes and output_stride=8")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Rust checkpoint has no model_state_dict")

    model_keys = set(model.state_dict())
    crack_keys = {key for key in model_keys if key.startswith("crack_head.")}
    rust_keys = model_keys - crack_keys
    state_keys = set(state)
    if state_keys != rust_keys:
        missing = sorted(rust_keys - state_keys)[:20]
        unexpected = sorted(state_keys - rust_keys)[:20]
        raise ValueError(
            f"Rust strict-load key mismatch: missing={missing}, unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if (
        not set(incompatible.missing_keys).issubset(crack_keys)
        or any(not key.startswith("crack_head.") for key in incompatible.missing_keys)
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Rust partial load was not exact: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return checkpoint


def set_training_stage(model: DualHeadMobileNetV2OS8, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "rust_restore":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif stage == "crack_bootstrap":
        for parameter in model.crack_head.parameters():
            parameter.requires_grad = True
    elif stage == "joint":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        for parameter in model.crack_head.parameters():
            parameter.requires_grad = True
        high = model.backbone["high_level_features"]
        for block in list(high.children())[-4:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported training stage: {stage}")


def enforce_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()
