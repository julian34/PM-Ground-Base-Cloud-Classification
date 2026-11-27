"""Strategi pembekuan bertahap (transfer learning 5 tahap).

Tahap 1-2 : bobot ImageNet beku, hanya Top Block + lapisan baru yang dilatih.
Tahap 3-4 : buka Dense Block 3-4 (+ trans3, norm5).
Tahap 5   : full defrost (seluruh backbone end-to-end).
"""
from __future__ import annotations

import torch.nn as nn

from . import config as _C
from .config import USE_SCCONV

BLOK_FT = ("block3.", "se3.", "sa3.", "trans3.", "block4.", "se4.", "sa4.", "norm5.")


def _vit_pretrained(model) -> bool:
    """True bila cabang ViT berbobot pretrained (MobileViTv2) -> ikut aturan pembekuan.

    CloudViT (dilatih dari awal) TIDAK terpengaruh: ia tetap dilatih sejak fase 1
    seperti lapisan baru lain (perilaku lama).
    """
    return getattr(model, "vit_arch", None) == "mobilevitv2" and getattr(model, "vit", None) is not None


def _atur_vit(model, latih: bool, prefiks=("vit.",)):
    """Set requires_grad parameter cabang ViT pretrained yang berprefiks `prefiks`."""
    for n, p in model.named_parameters():
        if n.startswith(prefiks):
            p.requires_grad = latih


def _lapor(model, judul):
    lat = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"[{judul}] dilatih {lat:,} / {tot:,} ({100 * lat / tot:.2f}%)")


def bekukan_imagenet(model, latih_baru=True, verbose=True):
    """Tahap 1-2: bobot ImageNet beku; Top Block (+ lapisan baru) dilatih."""
    for n, p in model.named_parameters():
        if n.startswith("top."):
            p.requires_grad = True
        elif n in model.imagenet_keys:
            p.requires_grad = False
        else:   # SE, SA, log_q, log_tau. pq dilewati: pembacanya (h.2) beku = 0
            p.requires_grad = latih_baru and ".pq." not in n
    # Cabang ViT pretrained (MobileViTv2): beku seluruhnya, sama seperti bobot ImageNet.
    if _vit_pretrained(model):
        _atur_vit(model, False)
    # SCConv mengubah distribusi -> BN harus adaptasi. Baca flag dari instance model
    # (di-set konstruktor) agar ablasi --no-scconv konsisten, bukan dari global config.
    model.bn_beku = not getattr(model, "use_scconv", USE_SCCONV)
    model.bn_buka = ()
    if verbose:
        _lapor(model, "Beku ImageNet")


def buka_blok_3_4(model, verbose=True):
    """Tahap 3-4: Dense Block 3-4 + trans3 + norm5 ikut dilatih."""
    bekukan_imagenet(model, verbose=False)
    for n, p in model.named_parameters():
        if n.startswith(BLOK_FT):
            p.requires_grad = True
    # Cabang ViT pretrained: stage terakhir ikut dibuka (setara Dense Block 3-4),
    # kecuali mode "beku" (tetap beku di semua fase).
    if _vit_pretrained(model) and _C.VIT_PRETRAINED_FT == "jadwal":
        _atur_vit(model, True, tuple(_C.VIT_PRETRAINED_BLOK_FT))
    model.bn_buka = BLOK_FT
    if verbose:
        _lapor(model, "Buka Block 3-4")


def buka_semua(model, verbose=True):
    """Tahap 5: full defrost."""
    for p in model.parameters():
        p.requires_grad = True
    # Mode "beku": cabang ViT pretrained tetap beku walau full defrost.
    if _vit_pretrained(model) and _C.VIT_PRETRAINED_FT == "beku":
        _atur_vit(model, False)
    model.bn_beku, model.bn_buka = False, ()
    if verbose:
        _lapor(model, "Full defrost")


def set_bn_beku_eval(model):
    """Setelah model.train(): BN ImageNet yang beku tidak update running stats."""
    if not getattr(model, "bn_beku", False):
        return
    for n, m in model.named_modules():
        if (isinstance(m, nn.BatchNorm2d) and f"{n}.weight" in model.imagenet_keys
                and not n.startswith(getattr(model, "bn_buka", ()))):
            m.eval()
