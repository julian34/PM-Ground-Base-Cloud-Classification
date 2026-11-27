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
from .config import (
    CLOUDVIT_DROP, CLOUDVIT_GDEPTH, IMG_SIZE, MOBILEVIT_NAME, ND_CH, NUM_CLASSES, USE_SCCONV,
    USE_SE, USE_VIT, VIT_ARCH, VIT_DIM, VIT_INPUT, VIT_PATCH, VIT_PATCH_RAW,
)
from .cloudvit_branch import CloudViTBranch
from .losses import init_new_layers
from .mobilevit_branch import MobileViTv2Branch
from .vit_branch import ViTBranch


class CloudDenseNet(nn.Module):
    def __init__(self, growth_rate=32, n_cls=NUM_CLASSES, top_hidden=None, use_vit=USE_VIT,
                 vit_input=VIT_INPUT, vit_arch=VIT_ARCH, use_se=USE_SE, use_scconv=USE_SCCONV,
                 vit_pretrained_weights=True):
        super().__init__()
        # Simpan flag di instance -> dibaca finetune (bn_beku) & untuk introspeksi/ablasi.
        self.use_se, self.use_scconv = use_se, use_scconv
        kw = dict(growth_rate=growth_rate, nd=ND_CH, use_se=use_se, use_scconv=use_scconv)

        # ===== Backbone =====
        # Stem: 3 -> 64, 224 -> 56 (setelah conv7x7 s2 + maxpool)
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 1: 64 -> 256 | Trans 1: -> 128
        self.block1 = DenseBlock1(num_layers=6, in_channels=64, **kw)
        self.se1 = SEBlock(256) if use_se else nn.Identity()      # gated: baseline tanpa SE
        self.sa = SpatialAttnBR()
        self.trans1 = TransitionLayer1(256, 128)

        # Block 2: 128 -> 512 | Trans 2: -> 256
        self.block2 = DenseBlock2(num_layers=12, in_channels=128, **kw)
        self.se2 = SEBlock(512) if use_se else nn.Identity()
        self.sa2 = SpatialAttnBR()
        self.trans2 = TransitionLayer2(512, 256)

        # Block 3: 256 -> 1024 | Trans 3: -> 512
        self.block3 = DenseBlock3(num_layers=24, in_channels=256, **kw)
        self.se3 = SEBlock(1024) if use_se else nn.Identity()
        self.sa3 = SpatialAttnBR()
        self.trans3 = TransitionLayer3(1024, 512)

        # Block 4: 512 -> 1024 | BN akhir
        self.block4 = DenseBlock4(num_layers=16, in_channels=512, **kw)
        self.se4 = SEBlock(1024) if use_se else nn.Identity()
        self.sa4 = SpatialAttnBR()
        self.norm5 = nn.BatchNorm2d(1024)

        # ===== Cabang B (opsional) =====
        # Dibangun hanya bila use_vit=True. Arsitektur dipilih lewat vit_arch:
        #   "cloudvit" -> backbone CloudViT penuh (I/D/E/G_Module), baca citra mentah,
        #                 jalur INDEPENDEN dari DenseNet, keluaran 256-dim.
        #   "simple"   -> ViT kecil; sumber input lewat vit_input:
        #                   "stem" -> peta stem (64 ch, 56x56), berbagi konvolusi awal.
        #                   "raw"  -> citra mentah (3 ch, IMG_SIZE), jalur independen.
        # Keluaran cabang di-concat (fusi) sebelum Top Block via extra_dim.
        self.use_vit = use_vit
        self.vit_arch = vit_arch
        if use_vit and vit_arch == "cloudvit":
            self.vit = CloudViTBranch(gdepth=CLOUDVIT_GDEPTH, drop=CLOUDVIT_DROP)
            self.vit_input = "raw"                   # CloudViT selalu baca citra mentah
            extra_dim = self.vit.out_dim             # 256
        elif use_vit and vit_arch == "mobilevitv2":
            # MobileViTv2 pretrained ImageNet (timm). vit_pretrained_weights=False hanya
            # untuk membangun ulang dari checkpoint proyek (bobot dari checkpoint).
            # Pembekuannya diatur finetune.py mengikuti jadwal PHASES.
            self.vit = MobileViTv2Branch(MOBILEVIT_NAME, pretrained=vit_pretrained_weights)
            self.vit_input = "raw"                   # baca citra mentah, jalur independen
            extra_dim = self.vit.out_dim             # 512 untuk mobilevitv2_100
        elif use_vit:                                # vit_arch == "simple"
            self.vit_input = vit_input
            if vit_input == "raw":
                self.vit = ViTBranch(in_channels=3, patch=VIT_PATCH_RAW, input_size=IMG_SIZE)
            elif vit_input == "stem":
                self.vit = ViTBranch(in_channels=64, patch=VIT_PATCH, input_size=IMG_SIZE // 4)
            else:
                raise ValueError(f"VIT_INPUT tak dikenal: {vit_input!r} (pakai 'stem' atau 'raw')")
            extra_dim = VIT_DIM
        else:
            self.vit = None
            self.vit_input = None
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

        # --- Cabang B: ViT (opsional) ---
        # Sumber input tergantung mode: "raw" -> citra mentah x (jalur independen),
        # "stem" -> peta stem s (berbagi konvolusi awal).
        # vec_vit = None bila ViT nonaktif -> Top Block berperilaku lama.
        if self.vit is not None:
            vit_in = x if self.vit_input == "raw" else s
            vec_vit = self.vit(vit_in)                            # (B, VIT_DIM)
        else:
            vec_vit = None

        # --- Fusi: Top Block meng-GAP 'out' lalu concat dgn vec_vit ---
        logits = self.top(out, extra=vec_vit)                     # (B, n_cls)
        if return_all:
            return {"stem": s, "block1": b1, "trans1": t1, "block2": b2, "trans2": t2,
                    "block3": b3, "trans3": t3, "block4": b4, "out": out,
                    "vit": vec_vit, "logits": logits}
        return logits
