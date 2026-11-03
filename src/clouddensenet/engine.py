"""Loop training & evaluasi, penjadwalan tahap, dan pengaman NaN.

Fungsi di sini di-parameterkan (tidak mengandalkan global) agar dapat diuji
dan dipanggil dari `scripts/train.py`.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .config import (
    DEBUG, DEBUG_BATCHES, GRAD_CLIP, MIXUP_ALPHA, MIXUP_P, PHASES,
)
from .data import siapkan_batch
from .finetune import bekukan_imagenet, buka_blok_3_4, buka_semua, set_bn_beku_eval


def mixup_batch(x01, y):
    w = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))          # bobot MixUp (bukan λ B/R)
    perm = torch.randperm(x01.size(0), device=x01.device)
    return w * x01 + (1 - w) * x01[perm], y[perm], w


def phase_of(epoch):
    return next((i for i, p in enumerate(PHASES, 1) if epoch <= p[0]), len(PHASES))


def apply_phase(model, optimizer, phase):
    strategi = PHASES[phase - 1][2]
    if   strategi == "frozen":  bekukan_imagenet(model)
    elif strategi == "partial": buka_blok_3_4(model)
    else:                       buka_semua(model)
    for grp in optimizer.param_groups:
        grp["lr"] = PHASES[phase - 1][1]  # LR = field ke-2


def _bn_buffers(model):
    return [b for n, b in model.named_buffers() if n.endswith(("running_mean", "running_var"))]


@torch.no_grad()
def lacak_nonfinite(model, x, lam, device):
    """Ulang batch yang gagal dengan hook -> modul pertama yang menghasilkan NaN/inf."""
    temuan, hs = [], []

    def hook(m, i, o):
        if not temuan and torch.is_tensor(o) and not torch.isfinite(o).all():
            temuan.append(f"{m._nama} ({type(m).__name__}, {o.dtype})")

    for n, m in model.named_modules():
        m._nama = n
        hs.append(m.register_forward_hook(hook))
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        model(x, lam)
    for h in hs:
        h.remove()
    return temuan[0] if temuan else "forward bersih (NaN di loss)"


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    writer=None, epoch=1, log_every=50):
    model.train()
    set_bn_beku_eval(model)
    bn = _bn_buffers(model)
    total_loss, correct, n, n_skip = 0.0, 0, 0, 0
    for i, (x01, y) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if DEBUG and i >= DEBUG_BATCHES:
            break
        x01, y = x01.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_b, w = y, 1.0
        if MIXUP_P > 0 and random.random() < MIXUP_P:
            x01, y_b, w = mixup_batch(x01, y)                      # MixUp pada gambar mentah
        x, lam = siapkan_batch(x01, device)                        # normalisasi + peta λ B/R
        backup = [b.clone() for b in bn]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(x, lam)
            loss = w * criterion(out, y) + (1 - w) * criterion(out, y_b)
        if not torch.isfinite(loss):
            sumber = lacak_nonfinite(model, x, lam, device)
            for b, s in zip(bn, backup):
                b.copy_(s)                                         # buang statistik BN tercemar
            n_skip += 1
            tqdm.write(f"[!] batch {i}: loss NaN dilewati | sumber: {sumber}")
            if n_skip > 20:
                raise RuntimeError("Terlalu banyak batch NaN — periksa modul sumbernya")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
        if writer is not None and i % log_every == 0:          # umpan balik cepat per-batch
            step = (epoch - 1) * len(loader) + i
            writer.add_scalar("Loss/train_step", loss.item(), step)
            writer.add_scalar("LR_step", optimizer.param_groups[0]["lr"], step)
            writer.flush()
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n, n_nan, yt, yp = 0.0, 0, 0, [], []
    for i, (x01, y) in enumerate(loader):
        if DEBUG and i >= DEBUG_BATCHES:
            break
        x, lam = siapkan_batch(x01, device)
        y = y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(x, lam).float()
        n_nan += int(not torch.isfinite(out).all())
        total_loss += criterion(out, y).item() * x.size(0)
        yt.append(y.cpu())
        yp.append(out.argmax(1).cpu())
        n += x.size(0)
    if n_nan:
        print(f"[!] {n_nan} batch validasi NaN — statistik BN kemungkinan tercemar")
    if n == 0:                                   # loader kosong / DEBUG_BATCHES=0
        return 0.0, 0.0, 0.0
    yt, yp = torch.cat(yt).numpy(), torch.cat(yp).numpy()
    return total_loss / n, (yt == yp).mean(), f1_score(yt, yp, average="macro")


def is_finite(*vals):
    return all(math.isfinite(v) for v in vals)
