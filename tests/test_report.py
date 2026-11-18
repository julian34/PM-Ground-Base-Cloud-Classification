"""Uji modul laporan metrik (`clouddensenet.report`).

Tidak butuh dataset/GPU: metrik diuji terhadap sklearn, dan laporan lengkap
dibuat dari model + loader tensor acak kecil di CPU. Jalankan:
    pytest tests/test_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: tak membuka jendela saat plot

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import CloudDenseNet  # noqa: E402
from clouddensenet import config as C  # noqa: E402
from clouddensenet import report as R  # noqa: E402


def test_hitung_metrik_cocok_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, C.NUM_CLASSES, size=300)
    y_pred = y_true.copy()
    flip = rng.random(300) < 0.3
    y_pred[flip] = rng.integers(0, C.NUM_CLASSES, size=int(flip.sum()))

    rows, summ = R.hitung_metrik(y_true, y_pred, C.CLASS_NAMES)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(C.NUM_CLASSES), zero_division=0)
    assert len(rows) == C.NUM_CLASSES
    for i, row in enumerate(rows):
        assert abs(row["precision"] - p[i]) < 1e-9
        assert abs(row["recall"] - r[i]) < 1e-9
        assert abs(row["f1"] - f1[i]) < 1e-9
        assert row["support"] == int(sup[i])
    assert abs(summ["accuracy"] - accuracy_score(y_true, y_pred)) < 1e-9
    assert summ["macro"]["support"] == len(y_true)


def test_simpan_csv_format(tmp_path):
    rows = [{"kelas": c, "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 10}
            for c in C.CLASS_NAMES]
    summ = {"accuracy": 0.5,
            "macro": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 70},
            "weighted": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 70}}
    p = R.simpan_metrik_csv(tmp_path / "m.csv", "Test (held-out)", rows, summ)
    baris = p.read_text(encoding="utf-8").splitlines()
    assert baris[0] == "# Test (held-out)"
    assert baris[1] == "kelas,precision,recall,f1_score,support"
    assert baris[-2].startswith("macro avg,")
    assert baris[-1].startswith("weighted avg,")


def test_kurva_terpisah_dibuat(tmp_path):
    hist = {"train_loss": [2, 1, 0.5], "val_loss": [2.1, 1.2, 0.7],
            "train_acc": [0.3, 0.6, 0.9], "val_acc": [0.25, 0.55, 0.85],
            "val_f1": [0.2, 0.5, 0.8]}
    R.plot_kurva_loss(hist, simpan=str(tmp_path / "loss.png"))
    R.plot_kurva_akurasi(hist, simpan=str(tmp_path / "acc.png"))
    assert (tmp_path / "loss.png").exists()
    assert (tmp_path / "acc.png").exists()


def test_laporan_lengkap_end_to_end(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = CloudDenseNet().eval()

    # Loader tensor acak kecil (citra [0,1], label 0..6).
    x = torch.rand(6, 3, C.IMG_SIZE, C.IMG_SIZE)
    y = torch.arange(6) % C.NUM_CLASSES
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    hist = {"train_loss": [2.0, 1.0, 0.5], "val_loss": [2.1, 1.1, 0.6],
            "train_acc": [0.3, 0.6, 0.9], "val_acc": [0.25, 0.55, 0.85],
            "val_f1": [0.2, 0.5, 0.8]}
    meta = {"checkpoint": "artifacts/checkpoints/uji.pth", "best_f1": 0.8,
            "best_epoch": 3, "total_epoch": 3,
            "started_at": "2026-11-19T08:00:00", "finished_at": "2026-11-19T08:30:00",
            "resume": False, "sesi": [{"mulai": "2026-11-19T08:00:00",
                                       "start_epoch": 1, "resume": False}]}

    md = R.buat_laporan_lengkap(model, device, C.CLASS_NAMES, tmp_path, "uji", meta,
                                val_loader=loader, test_loader=loader, history=hist,
                                n_gradcam=2)
    txt = Path(md).read_text(encoding="utf-8")
    # Bagian wajib ada.
    for wajib in ["Waktu Pelatihan", "Spesifikasi Hardware",
                  "Konfigurasi Parameter Utama", "Metrik Klasifikasi & Confusion Matrix",
                  "Grafik Pelatihan"]:
        assert wajib in txt, f"HILANG: {wajib}"
    # File pendukung ditulis.
    assert (tmp_path / "uji_metrik_val.csv").exists()
    assert (tmp_path / "uji_metrik_test.csv").exists()
    assert (tmp_path / "uji_cm_val.png").exists()
    assert (tmp_path / "uji_cm_test.png").exists()
    assert (tmp_path / "uji_kurva_loss.png").exists()
    assert (tmp_path / "uji_kurva_akurasi.png").exists()


if __name__ == "__main__":
    test_hitung_metrik_cocok_sklearn()
    test_laporan_lengkap_end_to_end(Path("./_uji_report_tmp"))
    print("[OK] uji report sukses")
