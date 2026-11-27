"""Smoke test mode ablasi "ViT saja" (CloudViTOnly / train.py --arch vit-only).

Tanpa dataset/GPU. Memverifikasi: shape keluaran, kompatibilitas dengan jadwal
fase (tak ada yang terbeku), round-trip checkpoint lewat bangun_dari_checkpoint,
dan Grad-CAM memakai `gradcam_target` (model ini tak punya norm5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import (  # noqa: E402
    CloudDenseNet, CloudViTOnly, apply_phase, bangun_dari_checkpoint, plot_gradcam,
    set_bn_beku_eval,
)
from clouddensenet import config as C  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CloudViTOnly().eval()


def test_forward_shape(model):
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        out = model(x)                      # lam opsional & diabaikan
        out_lam = model(x, torch.rand(2, 1, C.IMG_SIZE, C.IMG_SIZE))
    assert out.shape == (2, C.NUM_CLASSES)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_lam)     # λ tak memengaruhi model ViT-saja


def test_return_all(model):
    with torch.no_grad():
        res = model(torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE), return_all=True)
    assert set(res) == {"vit", "logits"}
    assert res["vit"].shape == (2, model.vit.out_dim)
    assert res["logits"].shape == (2, C.NUM_CLASSES)


def test_tanpa_densenet(model):
    nama = {n for n, _ in model.named_parameters()}
    assert all(n.startswith(("vit.", "top.")) for n in nama)


def test_semua_fase_melatih_seluruh_model():
    # Tak ada bobot ImageNet -> fase frozen/partial/full semuanya melatih semua parameter.
    m = CloudViTOnly()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for fase in range(1, len(C.PHASES) + 1):
        apply_phase(m, opt, fase)
        assert all(p.requires_grad for p in m.parameters()), f"fase {fase}"
        assert opt.param_groups[0]["lr"] == C.PHASES[fase - 1][1]
        m.train()
        set_bn_beku_eval(m)                 # tak boleh error walau tanpa norm5/imagenet
        assert m.training


def test_backward_jalan():
    m = CloudViTOnly().train()
    out = m(torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE))
    out.sum().backward()
    assert m.top.fc.weight.grad is not None
    assert next(m.vit.i_block.parameters()).grad is not None


def test_roundtrip_checkpoint(tmp_path, model):
    path = tmp_path / "vitonly.pth"
    torch.save({"epoch": 1, "model": model.state_dict(), "best_f1": 0.5}, path)
    muat, info = bangun_dari_checkpoint(path)
    assert isinstance(muat, CloudViTOnly)
    assert info["top_hidden"] == []
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(x), muat.eval()(x), atol=1e-6)


def test_checkpoint_cloud_tetap_cloud(tmp_path):
    # Regresi: checkpoint CloudDenseNet (fusi) tidak boleh salah dikenali sbg vit-only.
    m = CloudDenseNet(use_vit=True)
    path = tmp_path / "cloud.pth"
    torch.save({"model": m.state_dict()}, path)
    muat, _ = bangun_dari_checkpoint(path)
    assert isinstance(muat, CloudDenseNet)


def test_gradcam(tmp_path, model):
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    simpan = tmp_path / "cam.png"
    plot_gradcam(model, x.clamp(0, 1), x, None, "cpu", simpan=str(simpan))
    assert simpan.exists()
