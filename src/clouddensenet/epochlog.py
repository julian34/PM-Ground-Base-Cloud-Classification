"""Log training per-epoch ke SATU file CSV.

Melengkapi TensorBoard (`tblog.py`): selain event file untuk dasbor, tiap epoch
juga dicatat sebagai satu baris pada file CSV teks biasa — mudah dibuka di Excel,
pandas, atau sekadar diintip saat training berjalan.

File: `artifacts/results/<run_name>_log.csv` (satu file per run).

Perilaku aman-gagal (fail-safe) seperti `tblog.py`: kegagalan menulis log TIDAK
boleh menggagalkan training — hanya mencetak peringatan. Mode append + header
ditulis hanya saat file belum ada, sehingga resume training menambah baris tanpa
menimpa riwayat sebelumnya.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

# Urutan kolom file log (sekaligus header CSV).
KOLOM = [
    "waktu", "epoch", "phase", "lr",
    "train_loss", "train_acc",
    "val_loss", "val_acc", "val_f1",
    "detik",
]


class EpochLogger:
    """Penulis log per-epoch ke satu file CSV (append, aman-gagal)."""

    def __init__(self, path):
        self.path = Path(path)
        self.aktif = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Tulis header hanya bila file baru (agar resume tidak menduplikasi header).
            if not self.path.exists() or self.path.stat().st_size == 0:
                with self.path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(KOLOM)
            print(f"[LOG] log per-epoch -> {self.path}")
        except OSError as e:  # noqa: BLE001 — logging tak boleh menggagalkan training
            print(f"[LOG] gagal menyiapkan file log ({e}) — pencatatan CSV dinonaktifkan")
            self.aktif = False

    def catat(self, epoch, phase, tr_loss, tr_acc, va_loss, va_acc, va_f1, lr, detik):
        """Tambahkan satu baris epoch ke file. No-op bila logger nonaktif."""
        if not self.aktif:
            return
        baris = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch, phase, f"{lr:g}",
            f"{tr_loss:.6f}", f"{tr_acc:.6f}",
            f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1:.6f}",
            f"{detik:.1f}",
        ]
        try:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(baris)
        except OSError as e:  # noqa: BLE001
            print(f"[LOG] gagal menulis baris epoch {epoch} ({e})")
