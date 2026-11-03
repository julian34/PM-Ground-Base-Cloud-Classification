"""Stratified K-Fold Cross-Validation untuk CloudDenseNet.

Membangun loader train/val untuk SATU fold dari indeks `StratifiedKFold`
(scikit-learn). Pola mengikuti `data.build_loaders`: dua `ImageFolder` pada folder
`train/` yang sama — satu dengan augmentasi (train), satu polos (val) — sehingga
indeks kedua dataset sejajar (urutan ImageFolder = `sorted`, deterministik).

K-fold berjalan HANYA di pool `train/`. Folder `test/` tetap holdout final dan
tidak pernah tersentuh selama CV.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .config import (
    BATCH_SIZE, IMBALANCE, NUM_CLASSES, NUM_WORKERS, SAMPLES_PER_EPOCH, SEED,
)
from .data import build_transforms, seed_worker


def base_targets(data_root):
    """Label kelas seluruh pool train/ + nama kelas.

    Tanpa transform (hanya baca label) agar cepat. Dipakai `StratifiedKFold` untuk
    menentukan fold; indeks yang dihasilkan sejajar dengan ImageFolder ber-transform
    karena urutannya sama-sama `sorted`.
    """
    base = ImageFolder(root=str(data_root / "train"))
    return np.asarray(base.targets), base.classes


def build_fold_loaders(data_root, device, train_idx, val_idx):
    """Loader train & val untuk satu fold dari indeks StratifiedKFold.

    - train_tf (augmentasi) untuk subset latih, val_tf (polos) untuk subset validasi.
    - WeightedRandomSampler dihitung ULANG dari distribusi kelas train fold ini
      (jangan pakai bobot global — tiap fold distribusinya sedikit berbeda).
    """
    train_tf, val_tf = build_transforms()
    g = torch.Generator().manual_seed(SEED)
    pin = (device.type == "cuda")

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    base_train = ImageFolder(root=str(data_root / "train"), transform=train_tf)
    base_val = ImageFolder(root=str(data_root / "train"), transform=val_tf)

    train_ds = Subset(base_train, train_idx.tolist())
    val_ds = Subset(base_val, val_idx.tolist())
    val_ds.classes = base_val.classes
    train_targets = np.asarray(base_train.targets)[train_idx]

    if IMBALANCE == "sampler":
        cnt = np.bincount(train_targets, minlength=NUM_CLASSES)
        w = torch.as_tensor(1.0 / cnt[train_targets], dtype=torch.double)
        kw = dict(sampler=WeightedRandomSampler(w, SAMPLES_PER_EPOCH, replacement=True, generator=g))
    else:
        kw = dict(shuffle=True, generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True,
                              worker_init_fn=seed_worker, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS > 0, **kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)
    return train_loader, val_loader, train_ds, val_ds
