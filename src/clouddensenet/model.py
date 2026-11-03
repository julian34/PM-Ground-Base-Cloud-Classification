"""Arsitektur utama CloudDenseNet.

Backbone bergaya DenseNet-121 (stem + 4 dense block + 3 transisi) yang
dimodifikasi dengan SEBlock (atensi channel) dan SpatialAttnBR (atensi spasial
sadar-B/R) di setiap block. Top Block memproyeksikan ke `NUM_CLASSES` logit.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .attention import SEBlock, SpatialAttnBR
from .blocks import (
    DenseBlock1, DenseBlock2, DenseBlock3, DenseBlock4,
    TopBlock, TransitionLayer1, TransitionLayer2, TransitionLayer3,
)
from .config import ND_CH, NUM_CLASSES, USE_SE


class CloudDenseNet(nn.Module):
    def __init__(self, growth_rate=32, n_cls=NUM_CLASSES):
        super().__init__()
        kw = dict(growth_rate=growth_rate, nd=ND_CH, use_se=USE_SE)

        # ===== Backbone =====
        # Stem: 3 -> 64, 224 -> 56 (setelah conv7x7 s2 + maxpool)
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 1: 64 -> 256 | Trans 1: -> 128
        self.block1 = DenseBlock1(num_layers=6, in_channels=64, **kw)
        self.se1, self.sa = SEBlock(256), SpatialAttnBR()
        self.trans1 = TransitionLayer1(256, 128)

        # Block 2: 128 -> 512 | Trans 2: -> 256
        self.block2 = DenseBlock2(num_layers=12, in_channels=128, **kw)
        self.se2, self.sa2 = SEBlock(512), SpatialAttnBR()
        self.trans2 = TransitionLayer2(512, 256)

        # Block 3: 256 -> 1024 | Trans 3: -> 512
        self.block3 = DenseBlock3(num_layers=24, in_channels=256, **kw)
        self.se3, self.sa3 = SEBlock(1024), SpatialAttnBR()
        self.trans3 = TransitionLayer3(1024, 512)

        # Block 4: 512 -> 1024 | BN akhir
        self.block4 = DenseBlock4(num_layers=16, in_channels=512, **kw)
        self.se4, self.sa4 = SEBlock(1024), SpatialAttnBR()
        self.norm5 = nn.BatchNorm2d(1024)

        # ===== Top Block =====
        self.top = TopBlock()

    def forward(self, x, lam=None, return_all=False):
        s = self.maxpool(self.relu(self.bn(self.conv(x))))        # (B, 64, 56, 56)
        b1 = self.sa(self.se1(self.block1(s)), lam)               # (B, 256, 56, 56)
        t1 = self.trans1(b1)
        b2 = self.sa2(self.se2(self.block2(t1)), lam)             # (B, 512, 28, 28)
        t2 = self.trans2(b2)
        b3 = self.sa3(self.se3(self.block3(t2)), lam)             # (B, 1024, 14, 14)
        t3 = self.trans3(b3)
        b4 = self.sa4(self.se4(self.block4(t3)), lam)             # (B, 1024, 7, 7)

        out = F.relu(self.norm5(b4))
        logits = self.top(out)                                    # (B, n_cls)
        if return_all:
            return {"stem": s, "block1": b1, "trans1": t1, "block2": b2, "trans2": t2,
                    "block3": b3, "trans3": t3, "block4": b4, "out": out,
                    "logits": logits}
        return logits
