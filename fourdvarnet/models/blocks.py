"""Neural building blocks for 4DVarNet (paper Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvFeatureNet(nn.Module):
    """Trainable G/H: 3 tanh conv layers + linear conv + batch norm (20-d features)."""

    def __init__(self, in_channels: int, n_features: int = 20, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, n_features, 3, padding=1),
            nn.BatchNorm2d(n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorUNet(nn.Module):
    """Data-driven prior Phi (two-scale U-Net, paper Sec. 4.1)."""

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
