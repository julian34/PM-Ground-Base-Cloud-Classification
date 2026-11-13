"""Pemuatan checkpoint yang sadar-varian Top Block.

Checkpoint training hanya menyimpan {epoch, model, best_f1} — tanpa metadata
arsitektur. Karena Top Block pernah diubah antar-run (mis. 07/08/11-11-2026:
1024->512->128->32->16->7; 12-11-2026: 1024->512->32->7), dimensi head dibaca
langsung dari bentuk tensor `top.bottleneck.*.weight` & `top.fc.weight` di
state_dict, lalu model dibangun dengan head yang cocok.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from .model import CloudDenseNet

_POLA_LINEAR = re.compile(r"^top\.bottleneck\.(\d+)\.weight$")


def ambil_state_dict(ck) -> dict:
    """Ambil state_dict dari berbagai format checkpoint; buang prefiks 'module.' (DDP)."""
    if isinstance(ck, dict):
        state = (ck.get("model") or ck.get("model_state_dict")
                 or ck.get("state_dict") or ck)
    else:
        state = ck
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def dimensi_top(state: dict) -> tuple[tuple[int, ...], int] | None:
    """Baca (lebar_hidden, n_cls) Top Block dari state_dict.

    Linear di `bottleneck` dikenali sebagai weight 2-D (BN punya weight 1-D).
    Mengembalikan None bila state_dict tidak memuat kunci Top Block.
    """
    linear = []
    for k, v in state.items():
        m = _POLA_LINEAR.match(k)
        if m and getattr(v, "ndim", 0) == 2:
            linear.append((int(m.group(1)), int(v.shape[0])))
    fc = state.get("top.fc.weight")
    if fc is None or getattr(fc, "ndim", 0) != 2:
        return None
    hidden = tuple(d for _, d in sorted(linear))
    return hidden, int(fc.shape[0])


def bangun_dari_checkpoint(ckpt_path, device="cpu") -> tuple[CloudDenseNet, dict]:
    """Bangun CloudDenseNet dengan Top Block sesuai checkpoint, lalu muat bobotnya.

    Melempar ValueError (dengan detail lengkap) bila bobot tetap tidak cocok,
    termasuk bila ada kunci model yang hilang dari checkpoint — agar model
    setengah-acak tidak diam-diam dipakai untuk prediksi.
    """
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ambil_state_dict(ck)

    dims = dimensi_top(state)
    if dims is None:
        model = CloudDenseNet()
    else:
        hidden, n_cls = dims
        model = CloudDenseNet(n_cls=n_cls, top_hidden=hidden)
    model = model.to(device)

    try:
        hasil = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise ValueError(
            f"Checkpoint '{ckpt_path.name}' tidak cocok dengan arsitektur model "
            f"(head terdeteksi: {dims}).\n{e}"
        ) from e

    if hasil.missing_keys:
        raise ValueError(
            f"Checkpoint '{ckpt_path.name}' tidak memuat {len(hasil.missing_keys)} "
            "kunci yang dibutuhkan model (arsitektur berbeda):\n  "
            + "\n  ".join(hasil.missing_keys)
        )

    info = {
        "top_hidden": list(model.top.hidden),
        "unexpected_keys": list(hasil.unexpected_keys),
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "best_f1": ck.get("best_f1") if isinstance(ck, dict) else None,
    }
    return model, info
