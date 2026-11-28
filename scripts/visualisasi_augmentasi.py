"""Visualisasi augmentasi data training untuk laporan.

Skrip mandiri (tidak mengubah kode lain). Pipeline augmentasi diambil LANGSUNG
dari `build_transforms()` di `clouddensenet/data.py`, jadi gambar yang dihasilkan
selalu sesuai dengan augmentasi yang benar-benar dipakai saat training
(termasuk bila `STRONG_AUG` di config diubah).

Keluaran (default: laporan/gambar/agumentasi/):
    <kelas>/00_asli.png, 01_..., 02_...      tiap tahap augmentasi, terpisah (tanpa teks)
    <kelas>/pipeline_01.png ...              hasil pipeline lengkap (acak, seeded)
    <kelas>_tahapan.png                      grid: asli + tiap tahap (berjudul)
    ringkasan_tahapan.png                    baris = kelas, kolom = asli + tiap tahap
    ringkasan_pipeline.png                   baris = kelas, kolom = asli + N variasi pipeline

Contoh:
    python scripts/visualisasi_augmentasi.py
    python scripts/visualisasi_augmentasi.py --n-variasi 6 --dpi 300
    python scripts/visualisasi_augmentasi.py --data-root data/processed/dataset_b
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet.data import build_transforms, set_seed  # noqa: E402

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Visualisasi augmentasi data (untuk laporan)")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--split", default="train", help="subfolder sumber citra (default train)")
    p.add_argument("--out", type=Path, default=ROOT / "laporan" / "gambar" / "agumentasi")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    p.add_argument("--n-variasi", type=int, default=5,
                   help="jumlah variasi pipeline lengkap per kelas")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Tahapan augmentasi (diturunkan dari train_tf yang sebenarnya)
# --------------------------------------------------------------------------- #
def _paksa_aktif(t):
    """Salin transform acak ber-probabilitas `p` dengan p=1 agar efeknya terlihat."""
    if hasattr(t, "p"):
        t = copy.copy(t)
        t.p = 1.0
    return t


def _label(t):
    """Label Bahasa Indonesia + parameter singkat untuk satu transform."""
    if isinstance(t, T.RandomAffine):
        return f"Affine (rotasi ±{t.degrees[1]:.0f}°, geser {t.translate[0]:.0%})"
    if isinstance(t, T.RandomHorizontalFlip):
        return f"Flip horizontal (p={t.p})"
    if isinstance(t, T.RandomVerticalFlip):
        return f"Flip vertikal (p={t.p})"
    if isinstance(t, T.RandomChoice):
        return "Rotasi 0/90/180/270°"
    if isinstance(t, T.ColorJitter):
        b, c, s = t.brightness, t.contrast, t.saturation
        teks = f"Color jitter\n(b={b[0]:.1f}–{b[1]:.1f}, c={c[0]:.1f}–{c[1]:.1f}"
        if s is not None:
            teks += f", s={s[0]:.1f}–{s[1]:.1f}"
        if t.hue is not None:
            teks += f", h=±{t.hue[1]:.2f}"
        return teks + ")"
    if isinstance(t, T.RandomGrayscale):
        return f"Grayscale (p={t.p})"
    if isinstance(t, T.RandomEqualize):
        return f"Equalize (p={t.p})"
    return type(t).__name__


def susun_tahapan(train_tf, img_size):
    """Kembalikan (resize, list (slug, label, transform)) — tiap tahap diterapkan ke citra ter-resize.

    Pad(reflect) + RandomAffine + CenterCrop digabung jadi satu tahap karena
    Pad/CenterCrop hanya pembungkus agar tepi hasil affine tidak hitam.
    """
    ts = list(train_tf.transforms)
    tahap, i = [], 0
    while i < len(ts):
        t = ts[i]
        if isinstance(t, (T.Resize, T.ToTensor)):
            i += 1
            continue
        if isinstance(t, T.Pad) and i + 2 < len(ts) and isinstance(ts[i + 1], T.RandomAffine):
            grup = T.Compose(ts[i:i + 3])                      # Pad -> Affine -> CenterCrop
            tahap.append(("affine", _label(ts[i + 1]) + "\n+ pad reflect", grup))
            i += 3
            continue
        slug = type(t).__name__.lower().replace("random", "")
        tahap.append((slug, _label(t), _paksa_aktif(t)))
        i += 1
    return T.Resize((img_size, img_size)), tahap


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #
def pilih_citra_per_kelas(folder, rng):
    """Satu citra acak (seeded) per subfolder kelas, urut nama kelas (= urutan ImageFolder)."""
    hasil = []
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in EKSTENSI)
        if files:
            hasil.append((d.name, files[rng.randrange(len(files))]))
    if not hasil:
        raise SystemExit(f"Tidak ada citra di {folder}")
    return hasil


def ke_numpy(img):
    """PIL atau tensor [3,H,W] -> array HxWx3 di [0,1]."""
    if isinstance(img, torch.Tensor):
        return img.permute(1, 2, 0).clamp(0, 1).numpy()
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def simpan_polos(img, path):
    """Simpan citra tanpa sumbu/teks (siap disisipkan ke laporan)."""
    arr = (ke_numpy(img) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def grid(baris, judul_kolom, judul_baris, path, dpi, judul=None):
    """baris: list[list[img]] -> grid gambar berjudul."""
    nr, nc = len(baris), len(baris[0])
    fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.4 * nr), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            a = ax[r, c]
            a.imshow(ke_numpy(baris[r][c]))
            a.set_xticks([])
            a.set_yticks([])
            if r == 0 and judul_kolom:
                a.set_title(judul_kolom[c], fontsize=8)
            if c == 0 and judul_baris:
                a.set_ylabel(judul_baris[r], fontsize=9)
    if judul:
        fig.suptitle(judul, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    sumber = args.data_root / args.split
    if not sumber.is_dir():
        raise SystemExit(f"Folder sumber tidak ada: {sumber}")
    args.out.mkdir(parents=True, exist_ok=True)

    train_tf, _ = build_transforms(args.img_size)
    resize, tahap = susun_tahapan(train_tf, args.img_size)
    print(f"STRONG_AUG={C.STRONG_AUG} | {len(tahap)} tahap augmentasi:")
    for slug, lbl, _ in tahap:
        print(f"  - {slug}: {lbl.replace(chr(10), ' ')}")

    sampel = pilih_citra_per_kelas(sumber, rng)
    kolom_tahap = ["Asli (resize)"] + [lbl for _, lbl, _ in tahap]
    kolom_pipe = ["Asli (resize)"] + [f"Variasi {i + 1}" for i in range(args.n_variasi)]
    semua_tahap, semua_pipe, nama_kelas = [], [], []

    for kelas, path in sampel:
        print(f"[{kelas}] {path.name}")
        d = args.out / kelas
        d.mkdir(parents=True, exist_ok=True)
        asli = resize(Image.open(path).convert("RGB"))

        baris = [asli]
        simpan_polos(asli, d / "00_asli.png")
        for i, (slug, _, t) in enumerate(tahap, start=1):
            torch.manual_seed(args.seed + i)                    # tiap tahap reproducible
            hasil = t(asli)
            baris.append(hasil)
            simpan_polos(hasil, d / f"{i:02d}_{slug}.png")

        torch.manual_seed(args.seed)
        pipe = [asli]
        for v in range(args.n_variasi):
            hasil = train_tf(Image.open(path).convert("RGB"))   # pipeline LENGKAP (tensor)
            pipe.append(hasil)
            simpan_polos(hasil, d / f"pipeline_{v + 1:02d}.png")

        grid([baris], kolom_tahap, None, args.out / f"{kelas}_tahapan.png", args.dpi,
             judul=f"Tahapan augmentasi — {kelas} ({path.name})")
        semua_tahap.append(baris)
        semua_pipe.append(pipe)
        nama_kelas.append(kelas)

    grid(semua_tahap, kolom_tahap, nama_kelas, args.out / "ringkasan_tahapan.png", args.dpi,
         judul="Tahapan augmentasi per kelas")
    grid(semua_pipe, kolom_pipe, nama_kelas, args.out / "ringkasan_pipeline.png", args.dpi,
         judul="Hasil pipeline augmentasi lengkap (acak, on-the-fly)")
    print(f"[OK] tersimpan di {args.out}")


if __name__ == "__main__":
    main()
