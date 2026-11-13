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
from .config import ND_CH, NUM_CLASSES, USE_SE, USE_VIT, VIT_DIM
from .losses import init_new_layers
from .vit_branch import ViTBranch


class CloudDenseNet(nn.Module):
    def __init__(self, growth_rate=32, n_cls=NUM_CLASSES, top_hidden=None, use_vit=USE_VIT):
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

        # ===== Cabang ViT (opsional) =====
        # Dibangun hanya bila use_vit=True. Membaca peta stem (64 channel) dan
        # menghasilkan vektor VIT_DIM yang di-concat sebelum Top Block.
        self.use_vit = use_vit
        if use_vit:
            self.vit = ViTBranch(in_channels=64)
            extra_dim = VIT_DIM
        else:
            self.vit = None
            extra_dim = 0

        # ===== Top Block =====
        # top_hidden=None -> config.TOP_HIDDEN; isi dari checkpoint saat inferensi.
        # extra_dim>0 memperlebar Linear pertama agar muat vektor cabang ViT.
        self.top = TopBlock(n_cls=n_cls, hidden=top_hidden, extra_dim=extra_dim)

        # Init LeCun-uniform untuk lapisan BARU (lihat losses.init_new_layers).
        self._init_lapisan_baru()

    def _init_lapisan_baru(self):
        """Terapkan init LeCun-uniform HANYA ke lapisan baru: Top Block + cabang ND (pq).

        Yang sengaja DILEWATI:
          * SEBlock & SpatialAttnBR — sudah di-init mendekati identitas di
            konstruktornya (bobot nol + bias 3.0); menimpanya merusak desain
            'atensi awal ≈ identitas'.
          * Backbone (stem, DenseLayer.h/conv, transisi) — pada jalur training
            (train.py) bobotnya ditimpa bobot ImageNet via `muat_imagenet`; saat
            dibangun tanpa itu (test / latih-dari-nol) tetap pakai init default.
        """
        self.top.apply(init_new_layers)
        for b in range(1, 5):
            for layer in getattr(self, f"block{b}").layers:
                if layer.pq is not None:                 # cabang ND (P/Q) — lapisan baru
                    init_new_layers(layer.pq)

    def forward(self, x, lam=None, return_all=False):
        s = self.maxpool(self.relu(self.bn(self.conv(x))))        # (B, 64, 56, 56)

        # --- Cabang A: backbone DenseNet (seperti semula) ---
        b1 = self.sa(self.se1(self.block1(s)), lam)               # (B, 256, 56, 56)
        t1 = self.trans1(b1)
        b2 = self.sa2(self.se2(self.block2(t1)), lam)             # (B, 512, 28, 28)
        t2 = self.trans2(b2)
        b3 = self.sa3(self.se3(self.block3(t2)), lam)             # (B, 1024, 14, 14)
        t3 = self.trans3(b3)
        b4 = self.sa4(self.se4(self.block4(t3)), lam)             # (B, 1024, 7, 7)
        out = F.relu(self.norm5(b4))

        # --- Cabang B: ViT (opsional) dari peta stem yang sama ---
        # vec_vit = None bila ViT nonaktif -> Top Block berperilaku lama.
        vec_vit = self.vit(s) if self.vit is not None else None   # (B, VIT_DIM) | None

        # --- Fusi: Top Block meng-GAP 'out' lalu concat dgn vec_vit ---
        logits = self.top(out, extra=vec_vit)                     # (B, n_cls)
        if return_all:
            return {"stem": s, "block1": b1, "trans1": t1, "block2": b2, "trans2": t2,
                    "block3": b3, "trans3": t3, "block4": b4, "out": out,
                    "vit": vec_vit, "logits": logits}
        return logits
