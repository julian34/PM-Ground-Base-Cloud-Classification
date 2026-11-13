"""Cabang Vision Transformer (ViT) sederhana untuk CloudDenseNet.

Cabang ini berjalan PARALEL dengan backbone DenseNet. Input-nya BUKAN citra
mentah, melainkan peta fitur hasil stem / konvolusi pertama, yaitu tensor
berukuran (B, 64, 56, 56). Alur singkatnya:

    peta stem (B,64,56,56)
      -> patch-embed (satu Conv) -> deretan token
      -> tambah token CLS + embedding posisi
      -> TransformerEncoder (bawaan PyTorch)
      -> ambil token CLS -> vektor ringkasan (B, VIT_DIM)

Vektor keluaran lalu digabung (concat) dengan vektor DenseNet di dalam Top Block.
Dibuat ringkas memakai `nn.TransformerEncoder` agar mudah dibaca dan tanpa
dependensi tambahan.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    IMG_SIZE, VIT_DEPTH, VIT_DIM, VIT_DROP, VIT_HEADS, VIT_MLP_RATIO, VIT_PATCH,
)


class ViTBranch(nn.Module):
    """ViT kecil: ubah peta fitur stem menjadi satu vektor ringkasan (B, dim)."""

    def __init__(self, in_channels=64, dim=VIT_DIM, depth=VIT_DEPTH, heads=VIT_HEADS,
                 patch=VIT_PATCH, mlp_ratio=VIT_MLP_RATIO, drop=VIT_DROP):
        super().__init__()
        self.dim = dim

        # Ukuran peta stem = IMG_SIZE / 4 (conv7x7 stride2 lalu maxpool stride2).
        # Contoh: 224 -> 56. Jumlah token = (56 / patch)^2, mis. (56/8)^2 = 49.
        stem_size = IMG_SIZE // 4
        grid = stem_size // patch
        num_patches = grid * grid

        # 1) Patch embedding: satu Conv ber-stride=patch memecah peta jadi token.
        #    (B, C, 56, 56) -> (B, dim, grid, grid). Tiap sel grid = 1 token.
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

        # 2) Token CLS: satu token belajar yang akan meringkas seluruh gambar.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # 3) Embedding posisi: 1 untuk token CLS + 1 untuk tiap patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # Inisialisasi kecil (truncated normal) seperti lazim pada ViT.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.drop = nn.Dropout(drop)

        # 4) Encoder Transformer standar. batch_first=True -> bentuk (B, token, dim).
        hidden = int(dim * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=hidden,
            dropout=drop, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: peta fitur stem (B, C, 56, 56).
        B = x.shape[0]

        t = self.patch_embed(x)                   # (B, dim, grid, grid)
        t = t.flatten(2).transpose(1, 2)          # (B, num_patches, dim)

        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, dim) — salin per batch
        t = torch.cat([cls, t], dim=1)            # (B, 1+num_patches, dim)
        t = self.drop(t + self.pos_embed)         # tambahkan info posisi

        t = self.encoder(t)                       # (B, token, dim)
        t = self.norm(t)
        return t[:, 0]                            # token CLS -> (B, dim)
