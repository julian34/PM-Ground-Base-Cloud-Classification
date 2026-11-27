"""Model ablasi "ViT saja": cabang CloudViT berdiri sendiri sebagai pengklasifikasi.

Dipakai untuk mengukur kemampuan CloudViT TANPA backbone DenseNet
(train.py --arch vit-only), sebagai pembanding di tabel ablasi:
DenseNet saja (--no-vit) vs ViT saja (--arch vit-only) vs fusi (default).

Antarmuka sengaja disamakan dengan CloudDenseNet agar engine, jadwal fase,
laporan, Grad-CAM, dan pemuat checkpoint bisa dipakai apa adanya:
  * forward(x, lam=None, return_all=False) — `lam` (peta λ B/R) diabaikan.
  * head bernama `top` (kunci `top.fc.*`) -> selalu ikut dilatih di fase apa pun.
  * `imagenet_keys` kosong -> fase "frozen"/"partial" tidak membekukan apa pun;
    seluruh model dilatih dari nol sepanjang jadwal (hanya LR per fase yang berlaku).
  * `gradcam_target` -> peta fitur spasial terakhir CloudViT (g3, 4x4).
"""
from __future__ import annotations

import torch.nn as nn

from .cloudvit_branch import CloudViTBranch
from .config import CLOUDVIT_DROP, CLOUDVIT_GDEPTH, NUM_CLASSES, VIT_ONLY_DROPOUT
from .losses import init_new_layers


class ViTOnlyHead(nn.Module):
    """Head linear: Dropout -> Linear(in_dim -> n_cls). `hidden=()` utk kompatibilitas info."""

    def __init__(self, in_dim: int, n_cls: int = NUM_CLASSES, dropout: float = VIT_ONLY_DROPOUT):
        super().__init__()
        self.hidden = ()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, v):
        return self.fc(self.drop(v))


class CloudViTOnly(nn.Module):
    """CloudViT -> GAP (256) -> Dropout -> Linear(n_cls)."""

    arch = "vit-only"

    def __init__(self, n_cls: int = NUM_CLASSES, gdepth: int = CLOUDVIT_GDEPTH,
                 drop: float = CLOUDVIT_DROP, dropout: float = VIT_ONLY_DROPOUT):
        super().__init__()
        self.vit = CloudViTBranch(gdepth=gdepth, drop=drop)
        self.top = ViTOnlyHead(self.vit.out_dim, n_cls=n_cls, dropout=dropout)
        self.top.apply(init_new_layers)            # LeCun-uniform, sama spt TopBlock

        # Atribut yang dibaca finetune.py / engine.py (tak ada bobot ImageNet).
        self.imagenet_keys: set[str] = set()
        self.bn_beku, self.bn_buka = False, ()
        self.use_vit, self.vit_arch, self.vit_input = True, "cloudvit", "raw"

    @property
    def gradcam_target(self):
        return self.vit.g3

    def forward(self, x, lam=None, return_all=False):
        vec = self.vit(x)                                          # (B, 256)
        logits = self.top(vec)                                     # (B, n_cls)
        if return_all:
            return {"vit": vec, "logits": logits}
        return logits


def is_state_vit_only(state: dict) -> bool:
    """True bila state_dict berasal dari CloudViTOnly (ada `vit.*`, tanpa backbone DenseNet)."""
    return (any(k.startswith("vit.") for k in state)
            and not any(k.startswith(("block1.", "conv.")) for k in state))
