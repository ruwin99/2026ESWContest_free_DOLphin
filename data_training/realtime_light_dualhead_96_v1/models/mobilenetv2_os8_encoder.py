from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class MobileNetV2OS8Encoder(nn.Module):
    """Virginia Tech MobileNetV2 OS8 feature extractor with named /4 and /8 outputs."""

    def __init__(self, official_training: Path) -> None:
        super().__init__()
        source = str(official_training.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from network.modeling import deeplabv3plus_mobilenet

        base = deeplabv3plus_mobilenet(
            num_classes=4, output_stride=8, pretrained_backbone=False
        )
        self.backbone = base.backbone

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        low = features["low_level"]
        high = features["out"]
        if low.shape[1] != 24 or high.shape[1] != 320:
            raise RuntimeError(
                f"MobileNetV2 feature contract mismatch: low={tuple(low.shape)}, high={tuple(high.shape)}"
            )
        return {"low": low, "high": high}
