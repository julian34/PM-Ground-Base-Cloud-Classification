"""Entrypoint Grad-CAM CloudDenseNet: visualisasi wilayah dasar prediksi.

Contoh:
    python scripts/gradcam.py                          # 1 sampel/kelas dari split test
    python scripts/gradcam.py --split val --n 8        # 8 sampel pertama dari val
    python scripts/gradcam.py --checkpoint artifacts/checkpoints/cloud_Versi01.pth

Hasil PNG -> artifacts/results/<run>_gradcam_<split>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    CloudDenseNet, build_loaders, build_test_loader, plot_gradcam, set_seed,
)
from clouddensenet.data import siapkan_batch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM CloudDenseNet 7-kelas awan")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--checkpoint", type=Path,
                   default=C.CKPT_DIR / f"{C.RUN_NAME}{C.VERSION}.pth")
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--n", type=int, default=None,
                   help="jumlah sampel; default None = 1 sampel per kelas")
    p.add_argument("--acak", action="store_true",
                   help="ambil sampel acak (seeded), bukan urutan pertama")
    p.add_argument("--hanya-benar", dest="hanya_benar", action="store_true",
                   help="hanya sampel yang prediksinya benar")
    p.add_argument("--show", action="store_true", help="tampilkan jendela (default simpan saja)")
    return p.parse_args()


def pilih_dataset(args, device):
    """Kembalikan (dataset, classes) untuk split terpilih (test/val/train)."""
    if args.split == "test":
        loader, ds = build_test_loader(args.data_root, device)
        if loader is None:
            raise SystemExit(f"folder test/ tidak ada di {args.data_root}")
        return ds, getattr(ds, "classes", C.CLASS_NAMES)
    _, _, train_ds, val_ds = build_loaders(args.data_root, device)
    ds = val_ds if args.split == "val" else train_ds
    return ds, getattr(val_ds, "classes", C.CLASS_NAMES)


@torch.no_grad()
def ambil_sampel(loader, n_kelas, model, device, n_total=None, hanya_benar=False):
    """Kumpulkan sampel dari loader: n_total sampel, atau 1 per kelas.

    `hanya_benar=True` hanya menerima gambar yang prediksi model-nya cocok label.
    Urutan acak vs tidak ditentukan oleh `shuffle` pada loader (lihat main).
    """
    imgs, labels, seen = [], [], {}
    for x, y in loader:
        pred = None
        if hanya_benar:
            xb, lamb = siapkan_batch(x, device)
            pred = model(xb, lamb).argmax(1).cpu()
        for i in range(x.size(0)):
            c = int(y[i])
            if hanya_benar and int(pred[i]) != c:
                continue
            if n_total is None and seen.get(c, 0) >= 1:
                continue
            imgs.append(x[i])
            labels.append(c)
            seen[c] = seen.get(c, 0) + 1
            if n_total is not None and len(imgs) >= n_total:
                break
        cukup = (n_total is not None and len(imgs) >= n_total) or \
                (n_total is None and len(seen) >= n_kelas)
        if cukup:
            break
    if not imgs:
        raise SystemExit("Tidak ada sampel yang memenuhi kriteria (coba tanpa --hanya-benar).")
    return torch.stack(imgs), torch.tensor(labels)


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    set_seed(C.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | split: {args.split}")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    model = CloudDenseNet().to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()  # WAJIB: samakan mode BatchNorm (eval) antara filter sampel & Grad-CAM
    print(f"Muat {args.checkpoint.name} (f1={ck.get('best_f1', float('nan')):.4f})")

    dataset, classes = pilih_dataset(args, device)
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=args.acak,
                        num_workers=C.NUM_WORKERS)
    imgs01, y_true = ambil_sampel(loader, n_kelas=len(classes), model=model, device=device,
                                  n_total=args.n, hanya_benar=args.hanya_benar)
    mode = "acak" if args.acak else "pertama"
    if args.hanya_benar:
        mode += "+hanya-benar"
    print(f"Grad-CAM untuk {imgs01.size(0)} sampel ({mode})")

    x, lam = siapkan_batch(imgs01, device)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{mode.replace('+', '_').replace('-', '')}" if (args.acak or args.hanya_benar) else ""
    out = C.RESULT_DIR / f"{args.run_name}_gradcam_{args.split}{suffix}.png"
    plot_gradcam(model, imgs01, x, lam, device, classes=classes, y_true=y_true,
                 simpan=str(out))
    print(f"[OK] tersimpan: {out}")


if __name__ == "__main__":
    main()
