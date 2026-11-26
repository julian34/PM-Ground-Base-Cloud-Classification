"""Entrypoint laporan metrik CloudDenseNet dari sebuah checkpoint (berdiri sendiri).

Membuat laporan yang sama seperti yang dihasilkan otomatis di akhir
``scripts/train.py``, tetapi dari sebuah checkpoint yang sudah ada — berguna
untuk mencetak ulang laporan tanpa melatih dari awal. Semua keluaran (Markdown,
CSV metrik, CM validasi/test, kurva loss & akurasi terpisah, Grad-CAM benar/salah)
ditulis ke SATU folder per laporan.

Contoh:
    python scripts/report.py
    python scripts/report.py --checkpoint "artifacts/checkpoints/06-11-2026/cloud_Versi01.pth"
    python scripts/report.py --dataset-b --checkpoint "artifacts/checkpoints/dataset-repartitioned_Versi01.pth" --run-name datasetB
    python scripts/report.py --out-dir "artifacts/results/cloud_laporan_ulang"

Kurva loss/akurasi membutuhkan riwayat (history) training. Diambil dari
checkpoint ``..._last.pth`` (berisi 'history') atau dari CSV log per-epoch.
Bila keduanya tak tersedia, kurva dilewati (bagian lain tetap dibuat).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    bangun_dari_checkpoint, buat_laporan_lengkap, build_loaders, build_test_loader,
    kumpulkan_info_hardware, set_seed,
)

HIST_KOLOM = ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]


def parse_args():
    p = argparse.ArgumentParser(description="Laporan metrik CloudDenseNet dari checkpoint")
    p.add_argument("--data-root", type=Path, default=None,
                   help=f"Root dataset ImageFolder (train/val/test). Default: {C.DATA_ROOT}")
    p.add_argument("--dataset-b", action="store_true",
                   help="Pakai config.DATASET_B_ROOT (data/processed/dataset_b) "
                        "bila --data-root tidak diisi eksplisit.")
    p.add_argument("--checkpoint", type=Path,
                   default=C.CKPT_DIR / f"{C.RUN_NAME}{C.VERSION}.pth")
    p.add_argument("--history-checkpoint", type=Path, default=None,
                   help="Checkpoint berisi 'history' untuk kurva (default: sibling ..._last.pth).")
    p.add_argument("--log-csv", type=Path, default=None,
                   help="CSV log per-epoch untuk kurva/waktu (fallback bila history checkpoint tak ada).")
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE,
                   help=f"Ukuran batch inferensi val/test (default {C.BATCH_SIZE} dari config). "
                        "Hanya memengaruhi throughput/VRAM, bukan hasil.")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE,
                   help=f"Sisi citra persegi H=W piksel (default {C.IMG_SIZE} dari config). "
                        "WAJIB sama dengan nilai saat checkpoint dilatih.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Folder keluaran laporan (default: artifacts/results/<run>_laporan_<tanggal>).")
    p.add_argument("--n-gradcam", type=int, default=4,
                   help="Jumlah sampel Grad-CAM per kategori (benar & salah). Default 4.")
    p.add_argument("--no-val", action="store_true",
                   help="Lewati evaluasi validasi (hanya test).")
    p.add_argument("--show", action="store_true", help="tampilkan jendela (default simpan saja)")
    return p.parse_args()


def _history_dari_checkpoint(path, device):
    """Ambil dict history dari sebuah checkpoint (atau None)."""
    if path is None or not path.exists():
        return None, {}
    ck = torch.load(path, map_location=device, weights_only=False)
    hist = ck.get("history")
    meta = {k: ck.get(k) for k in ("best_f1", "best_epoch", "epoch", "laporan")}
    return hist, meta


def _history_dari_csv(path):
    """Rekonstruksi history + rentang waktu + Σ detik dari CSV log per-epoch."""
    path = Path(path)
    if not path.exists():
        return None, None, None, None
    hist = {k: [] for k in HIST_KOLOM}
    waktu, detik = [], 0.0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in HIST_KOLOM:
                    if row.get(k):
                        hist[k].append(float(row[k]))
                if row.get("waktu"):
                    waktu.append(row["waktu"])
                if row.get("detik"):
                    detik += float(row["detik"])
    except (OSError, ValueError) as e:
        print(f"[report] gagal membaca CSV log ({e})")
        return None, None, None, None
    if not hist["train_loss"]:
        return None, None, None, None
    mulai = waktu[0] if waktu else None
    selesai = waktu[-1] if waktu else None
    return hist, mulai, selesai, (detik or None)


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    set_seed(C.SEED)

    if args.data_root is None:
        args.data_root = C.DATASET_B_ROOT if args.dataset_b else C.DATA_ROOT
    elif args.dataset_b:
        print("[report] --data-root eksplisit diberikan; --dataset-b diabaikan.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {args.data_root} | Device: {device} | SEED: {C.SEED} | "
          f"batch {args.batch_size} | img {args.img_size}")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint tidak ditemukan: {args.checkpoint}")
    if not args.data_root.exists():
        raise SystemExit(f"Dataset tidak ditemukan di {args.data_root}")

    model, info = bangun_dari_checkpoint(args.checkpoint, device)
    model.eval()
    f1 = info.get("best_f1")
    print(f"Muat {args.checkpoint.name} "
          + (f"(f1={f1:.4f})" if f1 is not None else f"(top_hidden={info.get('top_hidden')})"))

    # Loader: test penuh + (opsional) val dari train/val.
    test_loader, test_ds = build_test_loader(
        args.data_root, device, batch_size=args.batch_size, img_size=args.img_size)
    classes = getattr(test_ds, "classes", C.CLASS_NAMES) if test_ds is not None else C.CLASS_NAMES
    val_loader = None
    if not args.no_val:
        _, val_loader, _, val_ds = build_loaders(
            args.data_root, device, batch_size=args.batch_size, img_size=args.img_size)
        classes = getattr(val_ds, "classes", classes)

    # --- History untuk kurva (checkpoint _last.pth lalu fallback CSV) -------- #
    hist_ckpt = args.history_checkpoint
    if hist_ckpt is None:
        stem = args.checkpoint.stem
        cand = (args.checkpoint if stem.endswith("_last")
                else args.checkpoint.with_name(f"{stem}_last{args.checkpoint.suffix}"))
        hist_ckpt = cand if cand.exists() else None
    history, hmeta = _history_dari_checkpoint(hist_ckpt, device)
    laporan_sebelum = hmeta.get("laporan") or {}

    mulai = selesai = detik_komputasi = None
    log_csv = args.log_csv
    if log_csv is None and laporan_sebelum.get("run_dir"):
        kand = Path(laporan_sebelum["run_dir"]) / f"{args.run_name}_log.csv"
        log_csv = kand if kand.exists() else None
    if log_csv is not None:
        hist_csv, mulai, selesai, detik_komputasi = _history_dari_csv(log_csv)
        if history is None or not history.get("train_loss"):
            history = hist_csv

    # --- Folder keluaran (satu folder per laporan) -------------------------- #
    if args.out_dir is not None:
        run_dir = args.out_dir
    else:
        run_dir = C.RESULT_DIR / f"{args.run_name}_laporan_{datetime.now():%d-%m-%Y_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folder keluaran laporan: {run_dir}")

    # --- Meta laporan ------------------------------------------------------- #
    best_f1 = hmeta.get("best_f1") if hmeta.get("best_f1") is not None else f1
    try:
        ckpt_label = str(Path(args.checkpoint).resolve().relative_to(C.PROJECT_DIR))
    except ValueError:
        ckpt_label = str(args.checkpoint)

    meta = {
        "checkpoint": ckpt_label,
        "best_f1": best_f1,
        "best_epoch": hmeta.get("best_epoch"),
        "total_epoch": len(history["train_loss"]) if history and history.get("train_loss") else None,
        "started_at": laporan_sebelum.get("started_at") or mulai,
        "finished_at": selesai,
        "detik_komputasi": detik_komputasi,
        "resume": len(laporan_sebelum.get("sesi", []) or []) > 1,
        "sesi": laporan_sebelum.get("sesi"),
        "log_csv": str(log_csv) if log_csv else None,
        "hardware": kumpulkan_info_hardware(device),
        "catatan": "Laporan dibuat ulang (post-hoc) dari checkpoint via "
                   "`scripts/report.py` — waktu/komputasi/resume berasal dari log "
                   "pelatihan asli bila tersedia.",
    }

    buat_laporan_lengkap(
        model, device, classes, run_dir, args.run_name, meta,
        val_loader=val_loader, test_loader=test_loader, history=history,
        n_gradcam=args.n_gradcam)
    print(f"[OK] laporan selesai di: {run_dir}")


if __name__ == "__main__":
    main()
