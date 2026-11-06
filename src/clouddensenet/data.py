"""Dataset, DataLoader, augmentasi on-the-fly, dan persiapan batch.

Struktur folder dataset (torchvision ImageFolder):
    <DATA_ROOT>/train/<kelas>/*.jpg
    <DATA_ROOT>/val/<kelas>/*.jpg
    <DATA_ROOT>/test/<kelas>/*.jpg
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import (
    BATCH_SIZE, IMBALANCE, IMG_SIZE, MEAN, NORM_MODE, NUM_CLASSES, NUM_WORKERS,
    SAMPLES_PER_EPOCH, SEED, STD, STRONG_AUG, USE_BR, VAL_SPLIT,
)
from .spectral import lambda_relatif

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)

# Cache tensor mean/std per-device agar tidak disalin ulang ke device tiap batch.
_NORM_CACHE: dict = {}


def _norm_stats(device):
    stats = _NORM_CACHE.get(device)
    if stats is None:
        stats = (MEAN_T.to(device), STD_T.to(device))
        _NORM_CACHE[device] = stats
    return stats


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms():
    """Augmentasi train (on-the-fly) + transform val. Normalisasi dilakukan di siapkan_batch."""
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Pad(IMG_SIZE // 4, padding_mode="reflect"),
        transforms.RandomAffine(degrees=15, translate=(0.10, 0.10)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        *([transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
           transforms.RandomVerticalFlip(0.5)] if STRONG_AUG else []),
        transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.8, 1.4),
                               saturation=(0.9, 1.1), hue=0.08 if STRONG_AUG else 0),
        *([transforms.RandomGrayscale(0.2), transforms.RandomEqualize(0.2)] if STRONG_AUG else []),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    return train_tf, val_tf


def _stratified_val_indices(targets, val_frac, seed=SEED):
    """Pilih indeks validasi stratified per kelas (seeded, reproducible)."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_loaders(data_root, device):
    """Bangun train_loader & val_loader dari <data_root>/train (+ /val bila ada).

    Jika `<data_root>/val` tidak ada (kasus dataset GCD yang disediakan), validasi
    di-carve dari train secara stratified & seeded. Transform train/val berbeda,
    jadi subset dibuat dari dua ImageFolder pada folder train yang sama.
    """
    train_tf, val_tf = build_transforms()
    g = torch.Generator().manual_seed(SEED)
    pin = (device.type == "cuda")

    val_dir = data_root / "val"
    if val_dir.is_dir():
        train_dataset = ImageFolder(root=str(data_root / "train"), transform=train_tf)
        val_dataset = ImageFolder(root=str(val_dir), transform=val_tf)
        train_targets = train_dataset.targets
    else:
        # Tanpa folder val: carve dari train (transform train vs val tetap beda).
        base_train = ImageFolder(root=str(data_root / "train"), transform=train_tf)
        base_val = ImageFolder(root=str(data_root / "train"), transform=val_tf)
        tr_idx, va_idx = _stratified_val_indices(base_train.targets, VAL_SPLIT)
        print(f"[data] folder val/ tidak ada -> carve {len(va_idx)} val / "
              f"{len(tr_idx)} train (stratified, seed={SEED})")
        train_dataset = Subset(base_train, tr_idx)
        val_dataset = Subset(base_val, va_idx)
        val_dataset.classes = base_val.classes
        train_targets = [base_train.targets[i] for i in tr_idx]

    if IMBALANCE == "sampler":
        cnt = np.bincount(train_targets, minlength=NUM_CLASSES)
        w = torch.as_tensor(1.0 / cnt[train_targets], dtype=torch.double)
        kw = dict(sampler=WeightedRandomSampler(w, SAMPLES_PER_EPOCH, replacement=True, generator=g))
    else:
        kw = dict(shuffle=True, generator=g)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True,
                              worker_init_fn=seed_worker, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS > 0, **kw)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)
    return train_loader, val_loader, train_dataset, val_dataset


def _stratified_subset_indices(targets, per_class, seed=SEED):
    """Pilih `per_class` indeks per kelas secara stratified (seeded, reproducible).

    Bila suatu kelas punya < per_class citra, ambil semua yang ada (tidak error)
    dan beri peringatan. Dipakai untuk subset evaluasi test yang seimbang antar-kelas.
    """
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    idx_terpilih = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_ambil = min(per_class, len(idx))
        if n_ambil < per_class:
            print(f"[data] kelas {c}: hanya {len(idx)} citra (< {per_class}) -> ambil semua")
        idx_terpilih.extend(idx[:n_ambil].tolist())
    return sorted(idx_terpilih)


def build_test_loader(data_root, device, per_class=None):
    """Bangun test_loader dari <data_root>/test (ImageFolder). None bila folder tak ada.

    Dipakai untuk evaluasi akhir & confusion matrix pada data uji yang benar-benar
    held-out. Transform sama dengan val (tanpa augmentasi), normalisasi di siapkan_batch.

    Bila `per_class` diisi (mis. 300), test set disaring jadi subset stratified
    `per_class` citra per kelas (seeded `SEED`), seimbang antar-kelas. Default `None`
    memakai seluruh test set apa adanya.
    """
    test_dir = data_root / "test"
    if not test_dir.is_dir():
        print(f"[data] folder test/ tidak ada di {test_dir} -> lewati evaluasi test")
        return None, None
    _, val_tf = build_transforms()
    test_dataset = ImageFolder(root=str(test_dir), transform=val_tf)

    if per_class is not None:
        idx = _stratified_subset_indices(test_dataset.targets, per_class)
        classes = test_dataset.classes
        test_dataset = Subset(test_dataset, idx)
        test_dataset.classes = classes          # agar viz bisa baca nama kelas
        print(f"[data] subset test stratified: {len(idx)} citra "
              f"({per_class}/kelas, seed={SEED})")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                             persistent_workers=NUM_WORKERS > 0)
    return test_loader, test_dataset


@torch.no_grad()
def siapkan_batch(x01, device):
    """x01 [B,3,H,W] di [0,1] -> (x ternormalisasi, λ [B,1,H,W] atau None)."""
    x01 = x01.to(device, non_blocking=True).float()
    lam = lambda_relatif(x01).unsqueeze(1) if USE_BR else None
    if NORM_MODE == "imagenet":
        mean_t, std_t = _norm_stats(device)
        x = (x01 - mean_t) / std_t
    else:
        x = x01
    return x, lam
