"""Pembagian dataset GCD -> data/processed/{train,val,test} (stratified, seeded).

Sumber  : data/raw/GCD.zip  (struktur: GCD/{train,test}/<kelas>/*.jpg)
Hasil   : data/processed/train/<kelas>/*   (85% dari train asli)
          data/processed/val/<kelas>/*     (15% dari train asli, per-kelas)
          data/processed/test/<kelas>/*    (salinan test asli, utuh)

Split val diambil per-kelas (stratified) dengan RNG ber-seed, jadi hasilnya
reproducible dan proporsi tiap kelas tetap terjaga. Default sinkron dengan
config.py (SEED=42, VAL_SPLIT=0.15).

Contoh:
    python scripts/split_dataset.py                 # ekstrak zip + split (copy)
    python scripts/split_dataset.py --move          # pindahkan train (hemat disk)
    python scripts/split_dataset.py --val-frac 0.15 --seed 42 --force
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Split GCD -> data/processed/{train,val,test}")
    p.add_argument("--zip", type=Path, default=ROOT / "data" / "raw" / "GCD.zip")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "GCD",
                   help="Folder hasil ekstrak zip (berisi train/ dan test/)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--val-frac", type=float, default=0.15, help="Proporsi train -> val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move", action="store_true",
                   help="Pindahkan file train (bukan copy). test selalu di-copy.")
    p.add_argument("--force", action="store_true", help="Timpa folder out bila sudah ada")
    return p.parse_args()


def ensure_extracted(zip_path: Path, raw_dir: Path) -> Path:
    """Pastikan raw_dir berisi train/ & test/. Ekstrak zip bila perlu."""
    if (raw_dir / "train").is_dir() and (raw_dir / "test").is_dir():
        print(f"[ekstrak] sudah ada: {raw_dir}")
        return raw_dir
    if not zip_path.exists():
        sys.exit(f"[ERROR] zip tidak ditemukan: {zip_path}")
    print(f"[ekstrak] {zip_path} -> {raw_dir.parent} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir.parent)  # zip memuat prefix GCD/ -> jadi raw_dir
    if not (raw_dir / "train").is_dir():
        sys.exit(f"[ERROR] struktur tak terduga; 'train/' tak ada di {raw_dir}")
    print("[ekstrak] selesai")
    return raw_dir


def list_images(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def transfer(src: Path, dst_dir: Path, move: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(str(src), str(dst_dir / src.name))


def main():
    a = parse_args()
    raw = ensure_extracted(a.zip, a.raw_dir)
    train_src, test_src = raw / "train", raw / "test"

    if a.out.exists():
        if not a.force:
            sys.exit(f"[ERROR] {a.out} sudah ada. Pakai --force untuk menimpa.")
        print(f"[bersih] menghapus {a.out}")
        shutil.rmtree(a.out)

    classes = sorted(d.name for d in train_src.iterdir() if d.is_dir())
    rng = np.random.default_rng(a.seed)
    verb = "pindah" if a.move else "copy"
    print(f"\n{'kelas':<18}{'train':>8}{'val':>7}{'test':>8}")
    print("-" * 41)

    tot = {"train": 0, "val": 0, "test": 0}
    for c in classes:
        imgs = list_images(train_src / c)
        idx = np.arange(len(imgs))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(imgs) * a.val_frac)))
        val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:].tolist()

        for i in train_idx:
            transfer(imgs[i], a.out / "train" / c, a.move)
        for i in sorted(val_idx):
            transfer(imgs[i], a.out / "val" / c, a.move)
        # test: selalu copy (jaga raw/test tetap utuh)
        test_imgs = list_images(test_src / c) if (test_src / c).is_dir() else []
        for f in test_imgs:
            transfer(f, a.out / "test" / c, move=False)

        n_tr, n_va, n_te = len(train_idx), len(val_idx), len(test_imgs)
        tot["train"] += n_tr; tot["val"] += n_va; tot["test"] += n_te
        print(f"{c:<18}{n_tr:>8}{n_va:>7}{n_te:>8}")

    print("-" * 41)
    print(f"{'TOTAL':<18}{tot['train']:>8}{tot['val']:>7}{tot['test']:>8}")
    print(f"\n[selesai] metode train={verb}, test=copy, seed={a.seed}, "
          f"val_frac={a.val_frac}")
    print(f"[hasil]   {a.out}/{{train,val,test}}")


if __name__ == "__main__":
    main()
