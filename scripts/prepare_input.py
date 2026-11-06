"""Siapkan 1 gambar menjadi tensor .npy untuk di-feed ke Zetane Viewer / ONNX Runtime.

Model ONNX hasil `scripts/export_onnx.py` menerima input yang SUDAH dinormalisasi
(normalisasi imagenet terjadi di `data.siapkan_batch`, sebelum forward). Script ini
mereplikasi transform validasi + normalisasi itu persis:

    PIL(RGB) -> Resize(IMG_SIZE, IMG_SIZE) -> ToTensor([0,1]) -> (x - MEAN) / STD

Keluaran: array float32 shape (1, 3, IMG_SIZE, IMG_SIZE) di file .npy.

Opsi:
  --raw01   Simpan [0,1] TANPA normalisasi (mis. bila alat lain menormalkan sendiri,
            atau untuk inspeksi visual). Default: ternormalisasi (yang diminta model).

Contoh:
    python scripts/prepare_input.py data/processed/test/3_cirrus/xxx.jpg
    python scripts/prepare_input.py foto.png --output artifacts/onnx/input.npy
    python scripts/prepare_input.py foto.png --raw01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Siapkan gambar -> .npy untuk Zetane/ONNX")
    p.add_argument("image", type=Path, help="Path gambar sumber (jpg/png/...).")
    p.add_argument("--output", type=Path,
                   default=C.PROJECT_DIR / "artifacts" / "onnx" / "input.npy",
                   help="Path file .npy keluaran.")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    p.add_argument("--raw01", action="store_true",
                   help="Simpan [0,1] tanpa normalisasi imagenet (default: ternormalisasi).")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.image.exists():
        raise SystemExit(f"Gambar tidak ditemukan: {args.image}")

    # Transform validasi (sama seperti data.build_transforms -> val_tf).
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),                       # -> [0,1], shape (3,H,W)
    ])

    img = Image.open(args.image).convert("RGB")
    x01 = tf(img).unsqueeze(0)                       # (1,3,H,W) di [0,1]

    if args.raw01:
        x = x01
        mode = "[0,1] mentah (tanpa normalisasi)"
    elif C.NORM_MODE == "imagenet":
        mean = torch.tensor(C.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(C.STD).view(1, 3, 1, 1)
        x = (x01 - mean) / std                       # sama seperti siapkan_batch
        mode = "ternormalisasi imagenet (x-mean)/std"
    else:                                            # NORM_MODE == '01'
        x = x01
        mode = "[0,1] (NORM_MODE='01')"

    arr = x.numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)

    print(f"[OK] input disimpan -> {args.output}")
    print(f"     shape={arr.shape} dtype={arr.dtype} mode={mode}")
    print(f"     rentang nilai: min={arr.min():.3f} max={arr.max():.3f}")
    print("\nDi Zetane: load clouddensenet.onnx, lalu beri file .npy ini sebagai input "
          "untuk menjalankan inferensi & melihat aktivasi antar-layer.")


if __name__ == "__main__":
    main()
