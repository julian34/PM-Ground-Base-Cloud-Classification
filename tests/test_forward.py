"""Smoke test: bangun CloudDenseNet dan jalankan satu forward pass.

Tidak butuh dataset atau GPU. Jalankan:
    python tests/test_forward.py      # mode skrip
    pytest tests/                     # mode pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import CloudDenseNet  # noqa: E402
from clouddensenet import config as C  # noqa: E402


def test_forward_shape():
    model = CloudDenseNet(growth_rate=32, n_cls=C.NUM_CLASSES).eval()
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, C.NUM_CLASSES), logits.shape


def test_return_all_keys():
    model = CloudDenseNet().eval()
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        out = model(x, return_all=True)
    for key in ("stem", "block1", "block2", "block3", "block4", "out", "logits"):
        assert key in out, key


def test_forward_vit_branch():
    # Aktifkan cabang ViT lewat parameter (tanpa mengubah config global).
    # Output harus tetap (N, 7) walau ada cabang + fusi tambahan.
    model = CloudDenseNet(use_vit=True).eval()
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, C.NUM_CLASSES), logits.shape


if __name__ == "__main__":
    test_forward_shape()
    test_return_all_keys()
    m = CloudDenseNet()
    total = sum(p.numel() for p in m.parameters())
    print(f"[OK] forward pass sukses | total parameter: {total:,}")
