"""Stratified K-Fold Cross-Validation untuk CloudDenseNet.

Membangun loader train/val untuk SATU fold dari indeks `StratifiedKFold`
(scikit-learn). Pola mengikuti `data.build_loaders`: dua `ImageFolder` pada folder
`train/` yang sama — satu dengan augmentasi (train), satu polos (val) — sehingga
indeks kedua dataset sejajar (urutan ImageFolder = `sorted`, deterministik).

Kedua `ImageFolder` dibangun SEKALI (`build_base_datasets`) lalu dipakai ulang lintas
fold; tiap fold hanya mengganti indeks `Subset`, menghindari pemindaian direktori
berulang. K-fold berjalan HANYA di pool `train/`; folder `test/` tetap holdout final.
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


def build_base_datasets(data_root):
    """Bangun dua ImageFolder pada train/ SEKALI: (augmentasi, polos).

    Mengembalikan (base_train, base_val, targets, classes). `targets` dan urutan
    sampel identik untuk keduanya (root sama, `sorted`), jadi indeks fold sejajar.
    Dipakai `StratifiedKFold` (lewat `targets`) dan `build_fold_loaders` (lewat
    base_train/base_val) tanpa memindai ulang direktori tiap fold.
    """
    train_tf, val_tf = build_transforms()
    base_train = ImageFolder(root=str(data_root / "train"), transform=train_tf)
    base_val = ImageFolder(root=str(data_root / "train"), transform=val_tf)
    targets = np.asarray(base_train.targets)
    return base_train, base_val, targets, base_val.classes


def build_fold_loaders(base_train, base_val, device, train_idx, val_idx):
    """Loader train & val untuk satu fold dari dataset dasar yang sudah dibangun.

    - base_train (augmentasi) untuk subset latih, base_val (polos) untuk subset validasi.
    - WeightedRandomSampler dihitung ULANG dari distribusi kelas train fold ini
      (jangan pakai bobot global — tiap fold distribusinya sedikit berbeda).
    """
    g = torch.Generator().manual_seed(SEED)
    pin = (device.type == "cuda")

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

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
