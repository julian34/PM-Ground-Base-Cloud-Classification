"""Modul atensi: Squeeze-and-Excitation (channel) dan Spatial Attention.

- SEBlock        : atensi channel (Hu et al., 2018), diinisialisasi ≈ identitas.
- SpatialAttn    : atensi spasial dari rata-rata + maks channel.
- SpatialAttnBR  : SpatialAttn + 1 channel tambahan berisi peta λ rasio B/R.
- EdgeLayer      : menambahkan 3 channel tepi (Sobel x, Sobel y, Laplacian).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LAM_SCALE, SE_INIT_BIAS, SE_REDUCTION


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=SE_REDUCTION, init_bias=SE_INIT_BIAS):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, in_channels // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=True), nn.Sigmoid())
        nn.init.zeros_(self.excitation[2].weight)                 # awal ≈ identitas
        nn.init.constant_(self.excitation[2].bias, init_bias)     # sigmoid(3) ≈ 0.953

    def forward(self, x):
        return x * self.excitation(self.global_pool(x))


class SpatialAttn(nn.Module):
    def __init__(self, k=7, init_bias=3.0, keep_map=False):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, padding_mode="reflect")
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, init_bias)
        self.keep_map = keep_map          # True hanya saat visualisasi
        self.map = None

    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        a = torch.sigmoid(self.conv(m))
        if self.keep_map:
            self.map = a.detach()         # tanpa graf autograd
        return x * a


class SpatialAttnBR(SpatialAttn):
    """SpatialAttn + 1 channel λ (rasio B/R) -> atensi sadar-fisika."""

    def __init__(self, k=7, init_bias=3.0, keep_map=False):
        super().__init__(k, init_bias, keep_map)
        self.conv = nn.Conv2d(3, 1, k, padding=k // 2, padding_mode="reflect")  # +1 channel λ
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, init_bias)

    def forward(self, x, lam=None):
        if lam is None:                                    # tanpa λ -> perilaku = SpatialAttn biasa
            lam = x.new_zeros(x.shape[0], 1, *x.shape[-2:])
        else:
            lam = F.interpolate(lam, size=x.shape[-2:], mode="area").to(x.dtype) * LAM_SCALE
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True), lam], 1)
        a = torch.sigmoid(self.conv(m))
        if self.keep_map:
            self.map = a.detach()
        return x * a


class EdgeLayer(nn.Module):
    """Menambahkan 3 channel tepi (Sobel x, Sobel y, Laplacian) dari grayscale."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        lap = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.register_buffer("k", torch.stack([sx, sx.T, lap]).unsqueeze(1))   # (3,1,3,3)

    def forward(self, x):
        g = x.mean(1, keepdim=True)                                           # grayscale
        return torch.cat([x, F.conv2d(F.pad(g, (1,) * 4, mode="reflect"), self.k)], 1)
