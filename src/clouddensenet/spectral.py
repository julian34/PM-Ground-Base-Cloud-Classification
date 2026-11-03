"""Peta λ rasio Biru/Merah (B/R) untuk spatial attention yang sadar-fisika.

λ relatif > 0  -> piksel lebih biru dari biasanya (langit)
λ relatif < 0  -> piksel lebih putih (awan)
Dihitung pada citra RGB [0,1] yang BELUM dinormalisasi.
"""
from __future__ import annotations

import torch

from .config import EPS


def br_ratio_lambda(x, eps=EPS):
    """x: [...,3,H,W] RGB [0,1] (BELUM dinormalisasi) -> λ [...,H,W]."""
    r, b = x[..., 0, :, :], x[..., 2, :, :]
    return (b - r) / (b + r + eps)


def valid_mask(x, dark=0.10, sat=0.97):
    """True = piksel layak dinilai (bukan gelap, bukan silau)."""
    return (x.amax(-3) > dark) & (x.amin(-3) < sat)


def lambda_relatif(x01, eps=EPS):
    """λ dikurangi median per gambar, lalu soft-clamp |λ| < 0.5."""
    lam = br_ratio_lambda(x01, eps)                     # [...,H,W]
    valid = valid_mask(x01)
    med = torch.stack([l[v].median() if v.any() else l.new_tensor(0.0)
                       for l, v in zip(lam.reshape(-1, *lam.shape[-2:]),
                                       valid.reshape(-1, *valid.shape[-2:]))])
    med = med.reshape(*lam.shape[:-2], 1, 1)
    return 0.5 * torch.tanh((lam - med) / 0.5) * valid      # soft clamp, |λ| < 0.5
