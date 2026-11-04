"""Regresi: init LeCun-uniform untuk lapisan BARU harus benar-benar diterapkan.

`init_new_layers` / `lecun_uniform_` didokumentasikan sebagai skema init lapisan
baru (Top Block + cabang ND), tetapi sempat tidak pernah dipanggil di pipeline —
sehingga Top Block memakai init default PyTorch (bias acak), menyimpang diam dari
desain. Test ini mengunci dua hal sekaligus:

  1. Lapisan baru (Top Block + cabang ND `pq`) ter-init LeCun-uniform:
     bias di-nol-kan dan bobot berada dalam batas ±sqrt(3/fan_in).
  2. SEBlock & SpatialAttnBR TIDAK ikut ditimpa: keduanya sengaja di-init
     mendekati identitas (bobot nol + bias 3.0) di konstruktornya.

Tidak butuh dataset/GPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import CloudDenseNet  # noqa: E402


def _batas_lecun(weight):
    fan_in = weight[0].numel()
    return math.sqrt(3.0 / fan_in)


@pytest.fixture(scope="module")
def model():
    # Model berat (skala DenseNet-121); bangun sekali, dipakai semua test (tak ada mutasi).
    return CloudDenseNet()


def test_top_block_terinit_lecun(model):
    """Top Block: bias di-nol-kan, bobot dalam batas LeCun-uniform."""
    linear1 = model.top.bottleneck[0]
    fc = model.top.fc

    assert torch.count_nonzero(linear1.bias) == 0, "bias bottleneck harus nol (init_new_layers)"
    assert torch.count_nonzero(fc.bias) == 0, "bias fc harus nol (init_new_layers)"

    assert linear1.weight.abs().max() <= _batas_lecun(linear1.weight) + 1e-6
    assert fc.weight.abs().max() <= _batas_lecun(fc.weight) + 1e-6

    # BatchNorm1d Top Block: gamma=1, beta=0 (init_new_layers untuk BN).
    bn = model.top.bottleneck[1]
    assert torch.allclose(bn.weight, torch.ones_like(bn.weight))
    assert torch.count_nonzero(bn.bias) == 0


def test_cabang_nd_terinit_lecun(model):
    """Cabang ND (pq) tiap DenseLayer: bobot dalam batas LeCun-uniform."""
    pq_terlihat = False
    for name in ("block1", "block2", "block3", "block4"):
        for layer in getattr(model, name).layers:
            if layer.pq is not None:
                pq_terlihat = True
                assert layer.pq.weight.abs().max() <= _batas_lecun(layer.pq.weight) + 1e-6
    assert pq_terlihat, "ND_CH>0 diharapkan membuat cabang pq; tidak ada yang teruji"


def test_atensi_tidak_ditimpa(model):
    """SE & SpatialAttnBR tetap mendekati identitas (tidak ikut di-LeCun-kan)."""
    for se in (model.se1, model.se2, model.se3, model.se4):
        assert torch.count_nonzero(se.excitation[2].weight) == 0, "bobot SE harus tetap nol"
        assert torch.allclose(se.excitation[2].bias, torch.full_like(se.excitation[2].bias, 3.0))
    for sa in (model.sa, model.sa2, model.sa3, model.sa4):
        assert torch.count_nonzero(sa.conv.weight) == 0, "bobot SpatialAttnBR harus tetap nol"
        assert torch.allclose(sa.conv.bias, torch.full_like(sa.conv.bias, 3.0))


if __name__ == "__main__":
    m = CloudDenseNet()
    test_top_block_terinit_lecun(m)
    test_cabang_nd_terinit_lecun(m)
    test_atensi_tidak_ditimpa(m)
    print("[OK] init lapisan baru terpasang & atensi terjaga")
