"""Cabang ViT alternatif: MobileViTv2 dengan bobot pretrained ImageNet (via timm).

Dipakai bila `VIT_ARCH = "mobilevitv2"` (atau `train.py --vit-pretrained`). Menggantikan
CloudViT (dilatih dari nol) sebagai **Cabang B** yang berjalan paralel dengan DenseNet;
keluarannya (vektor `num_features`, 512 untuk lebar 1.0) di-concat di `TopBlock`
persis seperti CloudViT. Head klasifikasi ImageNet dibuang (`num_classes=0`).

MobileViTv2 (Mehta & Rastegari, TMLR 2022) memakai *separable self-attention* O(N)
yang sama dengan G_Module CloudViT, jadi ini "kerabat pretrained" dari CloudViT.

Normalisasi: bobot cvnets dilatih dengan input [0,1] TANPA mean/std
(`pretrained_cfg`: mean=0, std=1). Pipeline data proyek menormalisasi ImageNet
(`NORM_MODE="imagenet"`), maka cabang ini membalik normalisasi itu lalu menerapkan
mean/std milik bobot -> `data.py` tidak perlu diubah.

Pembekuan mengikuti jadwal fine-tuning bertahap (lihat `finetune.py`):
  * fase "frozen"  : seluruh `vit.*` beku (seperti bobot ImageNet DenseNet).
  * fase "partial" : stage terakhir (`VIT_PRETRAINED_BLOK_FT`) ikut dibuka.
  * fase "full"    : seluruh cabang dibuka.
  Dengan `VIT_PRETRAINED_FT = "beku"` cabang tetap beku di SEMUA fase.

BatchNorm yang bobotnya beku dipaksa mode eval (running stats tak berubah) lewat
override `train()` — hanya berlaku di dalam cabang ini.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import MEAN, NORM_MODE, STD


class MobileViTv2Branch(nn.Module):
    """Backbone MobileViTv2 (timm) sebagai cabang paralel; keluaran vektor `out_dim`."""

    def __init__(self, nama: str, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover - pesan bantuan saja
            raise ImportError(
                "Cabang MobileViTv2 butuh paket 'timm' (pip install timm)."
            ) from e

        # pretrained=False dipakai saat membangun ulang dari checkpoint proyek
        # (bobot diambil dari checkpoint, tak perlu unduh).
        self.backbone = timm.create_model(nama, pretrained=pretrained, num_classes=0)
        self.nama = nama
        self.out_dim = self.backbone.num_features            # 512 untuk *_100

        cfg = self.backbone.pretrained_cfg
        # Statistik normalisasi data proyek (dibalik) & milik bobot (diterapkan).
        self.register_buffer("in_mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("w_mean", torch.tensor(cfg.get("mean", (0.0, 0.0, 0.0)))
                             .view(1, 3, 1, 1), persistent=False)
        self.register_buffer("w_std", torch.tensor(cfg.get("std", (1.0, 1.0, 1.0)))
                             .view(1, 3, 1, 1), persistent=False)
        self.balik_imagenet = NORM_MODE == "imagenet"

    def train(self, mode: bool = True):
        """Seperti nn.Module.train, tetapi BN yang bobotnya beku tetap eval."""
        super().train(mode)
        if mode:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d) and m.weight is not None \
                        and not m.weight.requires_grad:
                    m.eval()
        return self

    def forward(self, x):
        if self.balik_imagenet:
            x = x * self.in_std + self.in_mean                # -> rentang [0,1]
        x = (x - self.w_mean) / self.w_std                    # normalisasi milik bobot
        return self.backbone(x)                               # (B, out_dim)


def is_state_mobilevit(state: dict) -> bool:
    """True bila state_dict memuat cabang MobileViTv2 (kunci `vit.backbone.*`)."""
    return any(k.startswith("vit.backbone.") for k in state)
