from __future__ import annotations

import torch
from torch import nn


class LRASPP96(nn.Module):
    def __init__(self, in_channels: int = 320, out_channels: int = 96) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.context_conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.context_gate = nn.Sigmoid()

    def forward(self, high: torch.Tensor) -> torch.Tensor:
        return self.main(high) * self.context_gate(self.context_conv(self.context_pool(high)))
