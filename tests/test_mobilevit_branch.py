"""Uji cabang ViT pretrained MobileViTv2 (train.py --vit-pretrained).

Memakai vit_pretrained_weights=False agar tak perlu unduh bobot (cukup arsitektur):
  * forward fusi -> (N, 7), Top Block melebar 1024 + 512.
  * pembekuan mengikuti jadwal PHASES (mode "jadwal" & "beku").
  * BN cabang yang beku tetap eval setelah model.train().
  * CloudViT (default) tidak terpengaruh: tetap dilatih sejak fase frozen.
  * round-trip checkpoint lewat bangun_dari_checkpoint.
"""
import pytest
import torch
import torch.nn as nn

pytest.importorskip("timm")

from clouddensenet import CloudDenseNet, bangun_dari_checkpoint  # noqa: E402
from clouddensenet import config as C  # noqa: E402
from clouddensenet.finetune import bekukan_imagenet, buka_blok_3_4, buka_semua  # noqa: E402


def _model():
    m = CloudDenseNet(use_vit=True, vit_arch="mobilevitv2", vit_pretrained_weights=False)
    m.imagenet_keys = set()          # cukup untuk uji pembekuan cabang ViT
    return m


def _vit_grad(m, prefiks="vit."):
    return [p.requires_grad for n, p in m.named_parameters() if n.startswith(prefiks)]


def test_forward_fusi_mobilevit():
    m = _model().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, C.NUM_CLASSES)
    assert m.vit.out_dim == 512
    assert m.top.extra_dim == 512


def test_jadwal_bertahap(monkeypatch):
    monkeypatch.setattr(C, "VIT_PRETRAINED_FT", "jadwal")
    m = _model()

    bekukan_imagenet(m, verbose=False)
    assert not any(_vit_grad(m)), "fase frozen: seluruh vit.* harus beku"

    buka_blok_3_4(m, verbose=False)
    terbuka = [n for n, p in m.named_parameters() if n.startswith("vit.") and p.requires_grad]
    assert terbuka, "fase partial: stage terakhir harus terbuka"
    assert all(n.startswith(tuple(C.VIT_PRETRAINED_BLOK_FT)) for n in terbuka)
    assert not any(_vit_grad(m, "vit.backbone.stem.")), "stem tetap beku di partial"

    buka_semua(m, verbose=False)
    assert all(_vit_grad(m)), "fase full: seluruh vit.* terbuka"


def test_mode_beku_semua_fase(monkeypatch):
    monkeypatch.setattr(C, "VIT_PRETRAINED_FT", "beku")
    m = _model()
    for fn in (bekukan_imagenet, buka_blok_3_4, buka_semua):
        fn(m, verbose=False)
        assert not any(_vit_grad(m)), f"{fn.__name__}: vit.* harus tetap beku"
    # Top Block tetap dilatih
    assert all(p.requires_grad for n, p in m.named_parameters() if n.startswith("top."))


def test_bn_beku_tetap_eval():
    m = _model()
    bekukan_imagenet(m, verbose=False)
    m.train()
    bn = [x for x in m.vit.backbone.modules() if isinstance(x, nn.BatchNorm2d)]
    assert bn and all(not x.training for x in bn)


def test_cloudvit_default_tak_terpengaruh():
    m = CloudDenseNet(use_vit=True)            # default: CloudViT, latih dari awal
    m.imagenet_keys = set()
    bekukan_imagenet(m, verbose=False)
    assert all(_vit_grad(m)), "CloudViT harus tetap dilatih sejak fase frozen"


def test_roundtrip_checkpoint(tmp_path):
    m = _model().eval()
    path = tmp_path / "mvit.pth"
    torch.save({"epoch": 1, "model": m.state_dict(), "best_f1": 0.0}, path)
    muat, _ = bangun_dari_checkpoint(path)
    assert muat.vit_arch == "mobilevitv2"
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(m(x), muat.eval()(x), atol=1e-5)
