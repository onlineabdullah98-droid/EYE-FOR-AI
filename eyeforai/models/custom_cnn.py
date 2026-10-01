"""Baseline CNN trained from scratch (FYP-1 baseline to compare against transfer learning)."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """(Conv3x3 -> BN -> ReLU) x 2. Pooling is applied outside so Grad-CAM can hook the block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        chans = (3, *channels)
        self.blocks = nn.ModuleList(ConvBlock(chans[i], chans[i + 1]) for i in range(len(channels)))
        self.pool = nn.MaxPool2d(2)
        self.spatial_dropout = nn.Dropout2d(0.1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:  # keep the last feature map un-pooled for Grad-CAM resolution
                x = self.spatial_dropout(self.pool(x))
        return self.classifier(self.gap(x))
