"""Minimal U-Net (Ronneberger et al. 2015) for small images.

This is a compact variant: two down-steps, two up-steps, with skip concat.
All convs use padding=1 so spatial shape is preserved within a block; down/up
are via MaxPool/ConvTranspose2d with stride=2.
"""
from __future__ import annotations

from typing import Optional

from nanograd import cat as _cat
from nanograd.nn import functional as F
from nanograd.nn.conv import Conv2d, ConvTranspose2d, MaxPool2d
from nanograd.nn.module import Module
from nanograd.tensor import Tensor


class DoubleConv(Module):
    def __init__(self, in_c: int, out_c: int, seed: int = 0):
        super().__init__()
        self.c1 = Conv2d(in_c, out_c, kernel_size=3, padding=1, seed=seed)
        self.c2 = Conv2d(out_c, out_c, kernel_size=3, padding=1, seed=seed + 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.c2(F.relu(self.c1(x))))


class UNet(Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 16,
        seed: Optional[int] = None,
    ):
        super().__init__()
        s = seed or 0
        self.d1 = DoubleConv(in_channels, base, seed=s)
        self.p1 = MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2, seed=s + 10)
        self.p2 = MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 2, base * 4, seed=s + 20)

        self.u2 = ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2, seed=s + 30)
        self.dec2 = DoubleConv(base * 4, base * 2, seed=s + 40)
        self.u1 = ConvTranspose2d(base * 2, base, kernel_size=2, stride=2, seed=s + 50)
        self.dec1 = DoubleConv(base * 2, base, seed=s + 60)

        self.out_conv = Conv2d(base, out_channels, kernel_size=1, seed=s + 70)

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.d1(x)            # base   × H   × W
        x = self.p1(s1)            # base   × H/2
        s2 = self.d2(x)            # 2base  × H/2
        x = self.p2(s2)            # 2base  × H/4

        x = self.bottleneck(x)     # 4base  × H/4

        x = self.u2(x)             # 2base  × H/2
        x = _cat([x, s2], axis=1)  # 4base
        x = self.dec2(x)           # 2base

        x = self.u1(x)             # base   × H
        x = _cat([x, s1], axis=1)  # 2base
        x = self.dec1(x)           # base

        return self.out_conv(x)
