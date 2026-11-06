"""Entrypoint evaluasi CloudDenseNet dari checkpoint (tanpa training ulang).

Muat bobot dari sebuah checkpoint lalu hitung confusion matrix + Akurasi/F1 pada
data test held-out. Secara default mengevaluasi test PENUH; bila --test-per-class
diisi, subset stratified N/kelas juga dievaluasi sebagai pembanding (test penuh
tetap dijalankan).

Contoh:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint "artifacts/checkpoints/06-11-2026/cloud_Versi01.pth"
    python scripts/evaluate.py --checkpoint "artifacts/checkpoints/06-11-2026/cloud_Versi01.pth" --test-per-class 300

Hasil PNG -> artifacts/results/<run>_cm_test.png (+ _cm_test<N>.png bila subset).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import torch

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import CloudDenseNet, build_test_loader, set_seed  # noqa: E402
from clouddensenet.viz import plot_confusion_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi CloudDenseNet dari checkpoint")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--checkpoint", type=Path,
                   default=C.CKPT_DIR / f"{C.RUN_NAME}{C.VERSION}.pth")
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--test-per-class", type=int, default=None,
                   help="Selain test penuh, evaluasi juga subset stratified N citra "
                        "per kelas (mis. 300) untuk pembanding. Default: hanya penuh.")
    p.add_argument("--show", action="store_true",
                   help="tampilkan jendela (default simpan saja)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    set_seed(C.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SEED: {C.SEED}")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint tidak ditemukan: {args.checkpoint}")
    if not args.data_root.exists():
        raise SystemExit(f"Dataset tidak ditemukan di {args.data_root}")

    model = CloudDenseNet().to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Muat {args.checkpoint.name} (f1={ck.get('best_f1', float('nan')):.4f})")

    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrix pada data TEST penuh (held-out <data_root>/test).
    test_loader, test_ds = build_test_loader(args.data_root, device)
    if test_loader is None:
        raise SystemExit(f"folder test/ tidak ada di {args.data_root}")
    classes = getattr(test_ds, "classes", C.CLASS_NAMES)
    print("[eval] confusion matrix — test penuh")
    plot_confusion_matrix(model, test_loader, device, classes=classes,
                          simpan=str(C.RESULT_DIR / f"{args.run_name}_cm_test.png"))

    # 2. (opsional) Subset test stratified N/kelas — pembanding, bukan pengganti.
    if args.test_per_class is not None:
        sub_loader, _ = build_test_loader(args.data_root, device,
                                          per_class=args.test_per_class)
        print(f"[eval] confusion matrix — test subset ({args.test_per_class}/kelas)")
        plot_confusion_matrix(
            model, sub_loader, device, classes=classes,
            simpan=str(C.RESULT_DIR / f"{args.run_name}_cm_test{args.test_per_class}.png"))


if __name__ == "__main__":
    main()
