"""Jalankan training CloudDenseNet untuk BEBERAPA seed berurutan + ringkasan mean ± std.

Tiap seed dijalankan sebagai proses `train.py` terpisah dengan run-name unik
`<run-name>_s<seed>`, jadi checkpoint/log/resume tidak saling timpa. Argumen lain
yang tidak dikenal script ini diteruskan apa adanya ke `train.py`.

Contoh:
    python scripts/run_seeds.py --seeds 42 123 2024 --run-name cloud --epochs 80
    python scripts/run_seeds.py --seeds 1 2 3 --run-name abl_base --no-se --no-vit

Hasil:
    artifacts/results/<run-name>_s<seed>_<tgl>/...          (per seed, dari train.py)
    artifacts/results/<run-name>_seeds_ringkasan.csv        (per seed + mean, std)

Berurutan (bukan paralel) karena satu GPU: paralel hanya merebut VRAM.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402

KOLOM = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Training multi-seed CloudDenseNet (argumen lain diteruskan ke train.py)")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="Daftar seed, mis. 42 123 2024")
    p.add_argument("--run-name", default=C.RUN_NAME, help="Basis run-name; tiap seed -> <basis>_s<seed>")
    p.add_argument("--split", choices=("test", "val"), default="test",
                   help="Metrik split mana yang diringkas (default test)")
    p.add_argument("--hanya-ringkas", action="store_true",
                   help="Lewati training; hanya baca hasil yang sudah ada & buat ringkasan")
    return p.parse_known_args()


def baca_metrik(run_name: str, split: str) -> dict | None:
    """Baca `<run>_metrik_<split>.csv` dari folder hasil TERBARU milik run tersebut."""
    nama = f"{run_name}_metrik_{split}.csv"
    kandidat = sorted(C.RESULT_DIR.glob(f"{run_name}_*/{nama}"),
                      key=lambda p: p.stat().st_mtime)
    if not kandidat:
        return None
    baris = {}
    with kandidat[-1].open(encoding="utf-8") as f:
        for row in csv.reader(line for line in f if not line.startswith("#")):
            if row:
                baris[row[0]] = row
    try:
        m, w = baris["macro avg"], baris["weighted avg"]
        return {
            "accuracy": float(baris["accuracy"][3]),
            "macro_precision": float(m[1]), "macro_recall": float(m[2]), "macro_f1": float(m[3]),
            "weighted_f1": float(w[3]),
            "sumber": str(kandidat[-1].relative_to(C.PROJECT_DIR)),
        }
    except (KeyError, IndexError, ValueError):
        print(f"[ringkas] format tak dikenali: {kandidat[-1]}")
        return None


def main():
    args, diteruskan = parse_args()
    gagal = []

    if not args.hanya_ringkas:
        for i, s in enumerate(args.seeds, 1):
            run = f"{args.run_name}_s{s}"
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
                   "--seed", str(s), "--run-name", run, *diteruskan]
            print(f"\n{'=' * 70}\n[seed {i}/{len(args.seeds)}] {' '.join(cmd)}\n{'=' * 70}")
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                print(f"[seed {s}] GAGAL — lanjut ke seed berikutnya")
                gagal.append(s)

    # Ringkasan per seed + mean ± std.
    hasil = {}
    for s in args.seeds:
        m = baca_metrik(f"{args.run_name}_s{s}", args.split)
        if m is None:
            print(f"[ringkas] seed {s}: metrik {args.split} tidak ditemukan")
        else:
            hasil[s] = m
    if not hasil:
        print("[ringkas] tidak ada hasil untuk diringkas.")
        sys.exit(1)

    stat = {}
    for k in KOLOM:
        nilai = [h[k] for h in hasil.values()]
        stat[k] = (statistics.mean(nilai), statistics.stdev(nilai) if len(nilai) > 1 else 0.0)

    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULT_DIR / f"{args.run_name}_seeds_ringkasan.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", *KOLOM, "sumber"])
        for s, h in hasil.items():
            w.writerow([s, *(f"{h[k]:.6f}" for k in KOLOM), h["sumber"]])
        w.writerow(["mean", *(f"{stat[k][0]:.6f}" for k in KOLOM), ""])
        w.writerow(["std", *(f"{stat[k][1]:.6f}" for k in KOLOM), ""])

    print(f"\nRingkasan {len(hasil)} seed (split {args.split}):")
    print(f"{'seed':>9} " + " ".join(f"{k:>16}" for k in KOLOM))
    for s, h in hasil.items():
        print(f"{s:>9} " + " ".join(f"{h[k]:>16.4f}" for k in KOLOM))
    # "+/-" (bukan ±) agar aman di konsol Windows non-UTF-8.
    print(f"{'mean+/-sd':>9} " + " ".join(f"{f'{m:.4f}+/-{sd:.4f}':>16}" for m, sd in stat.values()))
    print(f"Tersimpan: {out}")
    if gagal:
        print(f"[peringatan] seed gagal: {gagal}")
        sys.exit(1)


if __name__ == "__main__":
    main()
