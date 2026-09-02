from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mobilenetv2_os8_encoder import MobileNetV2OS8Encoder
from lraspp_96 import LRASPP96


class LightDualHead96(nn.Module):
    def __init__(self, official_training: Path) -> None:
        super().__init__()
        self.encoder = MobileNetV2OS8Encoder(official_training)
        self.low_project = nn.Sequential(
            nn.Conv2d(24, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.lraspp = LRASPP96(320, 96)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True),
        )
        self.rust_dropout = nn.Dropout2d(0.1)
        self.rust_head = nn.Conv2d(96, 4, 1)
        self.crack_detail = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output_size = images.shape[-2:]
        features = self.encoder(images)
        low = self.low_project(features["low"])
        high = self.lraspp(features["high"])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        shared = self.shared_decoder(torch.cat((high, low), dim=1))
        rust = self.rust_head(self.rust_dropout(shared))
        crack = self.crack_detail(torch.cat((shared, low), dim=1))
        rust = F.interpolate(rust, size=output_size, mode="bilinear", align_corners=False)
        crack = F.interpolate(crack, size=output_size, mode="bilinear", align_corners=False)
        return torch.cat((rust, crack), dim=1)


def load_encoder_from_rust_checkpoint_strict(model: LightDualHead96, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "deeplabv3plus_mobilenet":
        raise ValueError("Rust initialization checkpoint architecture mismatch")
    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError("Rust initialization checkpoint has no model_state_dict")
    source_backbone = {key: value for key, value in source.items() if key.startswith("backbone.")}
    target = model.encoder.state_dict()
    # MobileNetV2OS8Encoder owns the official module as ``encoder.backbone``;
    # consequently its state_dict intentionally retains the ``backbone.`` prefix.
    mapped = source_backbone
    if set(mapped) != set(target):
        raise ValueError(
            f"Encoder strict key mismatch: missing={sorted(set(target)-set(mapped))[:10]}, "
            f"unexpected={sorted(set(mapped)-set(target))[:10]}"
        )
    for key, value in mapped.items():
        if value.shape != target[key].shape:
            raise ValueError(f"Encoder shape mismatch at {key}: {value.shape} != {target[key].shape}")
    model.encoder.load_state_dict(mapped, strict=True)
    return checkpoint


def freeze_batchnorm_running_stats(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def stage4_encoder_modules(model: LightDualHead96) -> tuple[nn.Module, ...]:
    """The c160 (3 blocks) and c320 (1 block) MobileNetV2 stages."""
    high = model.encoder.backbone["high_level_features"]
    blocks = tuple(high.children())
    if len(blocks) < 4:
        raise RuntimeError("Official MobileNetV2 high-level feature layout changed")
    selected = blocks[-4:]
    output_channels = [module.conv[-1].num_features for module in selected]
    if output_channels != [160, 160, 160, 320]:
        raise RuntimeError(f"Unexpected final MobileNetV2 stages: {output_channels}")
    return selected


def set_stage(model: LightDualHead96, stage: int) -> None:
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == 1:
        modules = (model.low_project, model.lraspp, model.shared_decoder, model.rust_head)
    elif stage == 2:
        modules = (model.crack_detail,)
    else:
        modules = (
            model.low_project,
            model.lraspp,
            model.shared_decoder,
            model.rust_head,
            model.crack_detail,
        )
        if stage == 4:
            modules = (*modules, *stage4_encoder_modules(model))
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    freeze_batchnorm_running_stats(model)
