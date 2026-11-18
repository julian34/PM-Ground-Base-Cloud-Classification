"""Laporan metrik otomatis pasca-pelatihan CloudDenseNet.

Merangkai SATU laporan Markdown + semua gambar pendukung ke satu folder per
pelatihan (`run_dir`), berisi:

  1. **Konfigurasi parameter utama** yang dipakai (ringkasan ``config.py`` + jadwal ``PHASES``).
  2. **Confusion matrix** validasi & test (PNG) beserta tautannya.
  3. **Kurva Loss** (train|val) dan **Kurva Akurasi** (train|val) pada DUA kanvas
     terpisah (khusus loss dan khusus akurasi) beserta tautannya.
  4. **Grad-CAM** untuk prediksi **BENAR** dan **SALAH**, menampilkan skor
     kepercayaan (confidence) model pada tiap sampel.

Catatan yang ikut dilaporkan: waktu mulai & selesai pelatihan, keterangan bila
pelatihan dilanjutkan (resume) dari checkpoint, dan spesifikasi hardware.

Dipakai oleh ``scripts/train.py`` (otomatis di akhir training) dan
``scripts/report.py`` (berdiri sendiri dari sebuah checkpoint).

Metrik dihitung dengan ``sklearn.metrics.precision_recall_fscore_support`` pada
prediksi ``argmax`` — pipeline (transform/normalisasi + ``siapkan_batch``)
identik dengan pembuatan confusion matrix agar angka konsisten.
"""
from __future__ import annotations

import csv
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from . import config as C
from .config import PHASES
from .data import siapkan_batch
from .gradcam import plot_gradcam
from .viz import plot_confusion_matrix

__all__ = [
    "kumpulkan_info_hardware", "ringkas_durasi",
    "hitung_metrik", "simpan_metrik_csv",
    "plot_kurva_loss", "plot_kurva_akurasi",
    "gradcam_benar_salah", "buat_laporan_lengkap",
]


# --------------------------------------------------------------------------- #
# Prediksi & metrik
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _kumpulkan_prediksi(model, loader, device):
    """Jalankan model pada seluruh loader -> (y_true, y_pred) sebagai np.ndarray.

    Memakai `siapkan_batch` + autocast persis seperti `evaluate`/`plot_confusion_matrix`.
    """
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="metrik", leave=False, dynamic_ncols=True):
        x, lam = siapkan_batch(x, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(x, lam).argmax(1)
        y_pred += pred.cpu().tolist()
        y_true += y.tolist()
    return np.asarray(y_true), np.asarray(y_pred)


def hitung_metrik(y_true, y_pred, classes):
    """Precision/recall/F1/support per kelas + accuracy, macro & weighted avg.

    Mengembalikan ``(rows, summary)``:
      * ``rows``  : list dict per kelas (kelas, precision, recall, f1, support).
      * ``summary``: {accuracy, macro:{...}, weighted:{...}} (support = total sampel).
    """
    idx = list(range(len(classes)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0)
    rows = [{"kelas": classes[i], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f1[i]), "support": int(sup[i])} for i in idx]

    total = int(len(y_true))
    acc = float((y_true == y_pred).mean()) if total else 0.0
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="weighted", zero_division=0)
    summary = {
        "accuracy": acc,
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf), "support": total},
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf), "support": total},
    }
    return rows, summary


def simpan_metrik_csv(path, judul, rows, summary):
    """Tulis metrik ke CSV dengan format yang sama seperti `cloud_metrik_*.csv`."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {judul}\n")
        w = csv.writer(f)
        w.writerow(["kelas", "precision", "recall", "f1_score", "support"])
        for row in rows:
            w.writerow([row["kelas"], f"{row['precision']:.6f}", f"{row['recall']:.6f}",
                        f"{row['f1']:.6f}", row["support"]])
        m, wt = summary["macro"], summary["weighted"]
        w.writerow(["accuracy", "", "", f"{summary['accuracy']:.6f}", m["support"]])
        w.writerow(["macro avg", f"{m['precision']:.6f}", f"{m['recall']:.6f}",
                    f"{m['f1']:.6f}", m["support"]])
        w.writerow(["weighted avg", f"{wt['precision']:.6f}", f"{wt['recall']:.6f}",
                    f"{wt['f1']:.6f}", wt["support"]])
    return path


def _tabel_metrik_md(rows, summary):
    out = ["| Kelas | Precision | Recall | F1-score | Support |",
           "|---|---:|---:|---:|---:|"]
    for row in rows:
        out.append(f"| {row['kelas']} | {row['precision']:.4f} | {row['recall']:.4f} "
                   f"| {row['f1']:.4f} | {row['support']} |")
    m, w = summary["macro"], summary["weighted"]
    out.append(f"| **Accuracy** |  |  | **{summary['accuracy']:.4f}** | {m['support']} |")
    out.append(f"| **Macro avg** | {m['precision']:.4f} | {m['recall']:.4f} "
               f"| {m['f1']:.4f} | {m['support']} |")
    out.append(f"| **Weighted avg** | {w['precision']:.4f} | {w['recall']:.4f} "
               f"| {w['f1']:.4f} | {w['support']} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Kurva pelatihan — DUA kanvas terpisah (loss & akurasi)
# --------------------------------------------------------------------------- #
def _garis_fase(ax, n_ep):
    """Garis putus-putus vertikal di batas antar-tahap `PHASES` (orientasi)."""
    for batas, *_ in PHASES[:-1]:
        if 0 < batas < n_ep:
            ax.axvline(batas + 0.5, color="gray", ls=":", alpha=0.7)


def plot_kurva_loss(history, simpan=None):
    """Satu kanvas KHUSUS Loss: train_loss vs val_loss sepanjang epoch."""
    tr = history.get("train_loss", []) or []
    va = history.get("val_loss", []) or []
    ep = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(tr):
        ax.plot(ep, tr, "o-", color="#1f77b4", label="train_loss")
    if len(va):
        ax.plot(ep[:len(va)], va, "o--", color="#d62728", label="val_loss")
    _garis_fase(ax, len(ep))
    ax.set(title="Kurva Loss — Train vs Validasi", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if simpan:
        fig.savefig(simpan, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return simpan


def plot_kurva_akurasi(history, simpan=None):
    """Satu kanvas KHUSUS Akurasi: train_acc vs val_acc sepanjang epoch."""
    tr = history.get("train_acc", []) or []
    va = history.get("val_acc", []) or []
    ep = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(tr):
        ax.plot(ep, tr, "o-", color="#2ca02c", label="train_acc")
    if len(va):
        ax.plot(ep[:len(va)], va, "o--", color="#ff7f0e", label="val_acc")
    _garis_fase(ax, len(ep))
    ax.set(title="Kurva Akurasi — Train vs Validasi", xlabel="Epoch", ylabel="Akurasi")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if simpan:
        fig.savefig(simpan, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return simpan


# --------------------------------------------------------------------------- #
# Grad-CAM: prediksi benar & salah (dengan kepercayaan model)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _pisah_benar_salah(model, loader, device, n_benar, n_salah):
    """Kumpulkan sampel yang prediksinya BENAR dan SALAH dari loader.

    Mengembalikan ``((img_benar, label_benar), (img_salah, label_salah))`` —
    tiap `img_*` list tensor [3,H,W] di [0,1]; `label_*` list int label asli.
    """
    model.eval()
    bi, by, si, sy = [], [], [], []
    for x01, y in loader:
        x, lam = siapkan_batch(x01, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(x, lam).argmax(1).cpu()
        for i in range(x01.size(0)):
            c, pcl = int(y[i]), int(pred[i])
            if pcl == c and len(bi) < n_benar:
                bi.append(x01[i]); by.append(c)
            elif pcl != c and len(si) < n_salah:
                si.append(x01[i]); sy.append(c)
        if len(bi) >= n_benar and len(si) >= n_salah:
            break
    return (bi, by), (si, sy)


def gradcam_benar_salah(model, loader, device, classes, run_dir, run_name, split,
                        n_benar=4, n_salah=4):
    """Buat dua gambar Grad-CAM: sampel prediksi BENAR dan SALAH.

    Memanfaatkan `plot_gradcam` yang sudah menampilkan skor kepercayaan
    (confidence/softmax) dan penanda OK/SALAH pada judul tiap sampel.
    Mengembalikan dict {'benar': basename|None, 'salah': basename|None}.
    """
    (bi, by), (si, sy) = _pisah_benar_salah(model, loader, device, n_benar, n_salah)
    run_dir = Path(run_dir)
    hasil = {"benar": None, "salah": None}
    for nama, imgs, ys in [("benar", bi, by), ("salah", si, sy)]:
        if not imgs:
            print(f"[report] tidak ada sampel '{nama}' untuk Grad-CAM ({split}) — dilewati")
            continue
        imgs01 = torch.stack(imgs)
        x, lam = siapkan_batch(imgs01, device)
        out = run_dir / f"{run_name}_gradcam_{split}_{nama}.png"
        plot_gradcam(model, imgs01, x, lam, device, classes=classes,
                     y_true=torch.tensor(ys), simpan=str(out))
        hasil[nama] = out.name
    return hasil


# --------------------------------------------------------------------------- #
# Hardware & waktu
# --------------------------------------------------------------------------- #
def kumpulkan_info_hardware(device):
    """Spesifikasi hardware & lingkungan runtime (best-effort, aman-gagal)."""
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:  # noqa: BLE001
        tv = "tidak terdeteksi"

    info = {
        "os": f"{platform.system()} {platform.release()}",
        "mesin": platform.machine(),
        "prosesor": platform.processor() or "tidak terdeteksi",
        "cpu_logis": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": tv,
        "device": str(device),
        "cuda_versi": torch.version.cuda or "-",
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        "gpu": None,
        "gpu_mem_gb": None,
        "ram_gb": None,
    }
    try:
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001 — psutil opsional
        pass
    if info["ram_gb"] is None:  # fallback tanpa psutil (Windows/Unix)
        try:
            if platform.system() == "Windows":
                import ctypes

                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong),
                                ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

                ms = _MS()
                ms.dwLength = ctypes.sizeof(_MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
                info["ram_gb"] = round(ms.ullTotalPhys / 1024 ** 3, 1)
            else:
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                info["ram_gb"] = round(total / 1024 ** 3, 1)
        except Exception:  # noqa: BLE001
            pass
    return info


def ringkas_durasi(detik):
    """Format detik -> '2j 13m 05d' (jam/menit/detik)."""
    detik = int(round(detik))
    j, sisa = divmod(detik, 3600)
    m, d = divmod(sisa, 60)
    if j:
        return f"{j}j {m:02d}m {d:02d}d"
    if m:
        return f"{m}m {d:02d}d"
    return f"{d}d"


def _baca_detik_epoch(log_csv):
    """Jumlahkan kolom 'detik' dari CSV log per-epoch -> total waktu komputasi."""
    log_csv = Path(log_csv)
    if not log_csv.exists():
        return None
    total, n = 0.0, 0
    try:
        with log_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = row.get("detik") or row.get("dt")
                if val:
                    total += float(val)
                    n += 1
    except (OSError, ValueError):
        return None
    return total if n else None


# --------------------------------------------------------------------------- #
# Snapshot konfigurasi (parameter utama)
# --------------------------------------------------------------------------- #
def _ringkas_konfig_md():
    """Section Markdown 'Konfigurasi Parameter Utama' dari `config.py` aktif."""
    optimizer = f"AdamW (lr awal {C.LR:g}, weight_decay {C.WEIGHT_DECAY:g})"
    loss = (f"Focal Loss (γ={C.FOCAL_GAMMA:g})" if C.FOCAL_GAMMA > 0
            else "CrossEntropyLoss")
    utama = [
        ("Seed", f"`{C.SEED}`"),
        ("Ukuran citra", f"`{C.IMG_SIZE}×{C.IMG_SIZE}` RGB"),
        ("Normalisasi", f"`{C.NORM_MODE}` (mean/std ImageNet)"),
        ("Batch size", f"`{C.BATCH_SIZE}`"),
        ("Total epoch", f"`{C.EPOCHS}`"),
        ("Optimizer", optimizer),
        ("Loss", loss),
        ("Grad clip", f"`{C.GRAD_CLIP:g}`"),
        ("Penanganan imbang kelas", f"`{C.IMBALANCE}`"),
        ("Early stopping", f"`{C.EARLY_STOP}` (patience `{C.PATIENCE}`)"),
    ]
    arsitektur = [
        ("SEBlock (`USE_SE`)", f"`{C.USE_SE}`"),
        ("SCConv (`USE_SCCONV`)", f"`{C.USE_SCCONV}`"),
        ("Atensi Biru/Merah (`USE_BR`)", f"`{C.USE_BR}`"),
        ("Cabang ViT (`USE_VIT`)", f"`{C.USE_VIT}` — arch `{C.VIT_ARCH}`"),
        ("Top Block hidden (`TOP_HIDDEN`)", f"`{C.TOP_HIDDEN}`"),
        ("Dropout Top Block (`P_DROP_TOP`)", f"`{C.P_DROP_TOP:g}`"),
    ]

    lines = ["## Konfigurasi Parameter Utama", "",
             "> Sumber kebenaran: `src/clouddensenet/config.py`.", "",
             "**Pelatihan & optimasi**", "",
             "| Parameter | Nilai |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in utama]
    lines += ["", "**Arsitektur (modifikasi vs DenseNet-121)**", "",
              "| Komponen | Nilai |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in arsitektur]

    lines += ["", "**Jadwal training bertahap (`PHASES`)**", "",
              "| Tahap | Epoch akhir | LR | Mode | Deskripsi |",
              "|---|---:|---|---|---|"]
    for i, (ep, lr, strategi, desc) in enumerate(PHASES, 1):
        lines.append(f"| {i} | {ep} | `{lr:g}` | `{strategi}` | {desc} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orkestrator utama
# --------------------------------------------------------------------------- #
def buat_laporan_lengkap(model, device, classes, run_dir, run_name, meta,
                         val_loader=None, test_loader=None, history=None,
                         n_gradcam=4):
    """Hasilkan laporan lengkap (CSV + PNG + Markdown) ke `run_dir`.

    Parameter
    ---------
    meta : dict
        Kunci yang dipakai (semua opsional): ``checkpoint`` (str/Path),
        ``best_f1`` (float), ``best_epoch`` (int), ``total_epoch`` (int),
        ``started_at`` / ``finished_at`` (datetime | ISO str),
        ``resume`` (bool), ``sesi`` (list dict), ``log_csv`` (path),
        ``catatan`` (str tambahan).
    Mengembalikan path file Markdown yang ditulis.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tanggal = datetime.now().strftime("%d-%m-%Y")

    metrik = {}   # 'val'/'test' -> (rows, summary)
    gambar = {}   # label -> basename

    # --- 1. Metrik + confusion matrix: validasi lalu test ------------------- #
    for split, loader in (("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        judul = "Validasi" if split == "val" else "Test (held-out)"
        print(f"[report] metrik — {judul}")
        yt, yp = _kumpulkan_prediksi(model, loader, device)
        rows, summ = hitung_metrik(yt, yp, classes)
        metrik[split] = (rows, summ)
        simpan_metrik_csv(run_dir / f"{run_name}_metrik_{split}.csv", judul, rows, summ)
        cm = run_dir / f"{run_name}_cm_{split}.png"
        print(f"[report] confusion matrix — {judul}")
        plot_confusion_matrix(model, loader, device, classes=classes, simpan=str(cm))
        gambar[f"cm_{split}"] = cm.name

    # --- 2. Kurva loss & akurasi (dua kanvas terpisah) --------------------- #
    if history and history.get("train_loss"):
        loss_png = run_dir / f"{run_name}_kurva_loss.png"
        acc_png = run_dir / f"{run_name}_kurva_akurasi.png"
        plot_kurva_loss(history, simpan=str(loss_png))
        plot_kurva_akurasi(history, simpan=str(acc_png))
        gambar["kurva_loss"] = loss_png.name
        gambar["kurva_akurasi"] = acc_png.name

    # --- 3. Grad-CAM benar & salah (pakai test bila ada, else val) --------- #
    gc_loader = test_loader or val_loader
    gc_split = "test" if test_loader is not None else "val"
    if gc_loader is not None:
        print("[report] Grad-CAM — prediksi benar & salah")
        gc = gradcam_benar_salah(model, gc_loader, device, classes, run_dir,
                                 run_name, gc_split, n_benar=n_gradcam, n_salah=n_gradcam)
        gambar["gradcam_benar"] = gc.get("benar")
        gambar["gradcam_salah"] = gc.get("salah")
        gambar["gradcam_split"] = gc_split

    # --- 4. Rakit Markdown -------------------------------------------------- #
    md = _render_markdown(tanggal, run_name, device, classes, metrik, gambar, meta)
    out_md = run_dir / f"{run_name}_metrik.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[report] laporan tersimpan -> {out_md}")
    return out_md


def _fmt_waktu(w):
    if w is None:
        return "—"
    if isinstance(w, datetime):
        return w.strftime("%Y-%m-%d %H:%M:%S")
    return str(w)


def _as_dt(w):
    if isinstance(w, datetime):
        return w
    try:
        return datetime.fromisoformat(str(w))
    except (TypeError, ValueError):
        return None


def _render_markdown(tanggal, run_name, device, classes, metrik, gambar, meta):
    meta = meta or {}
    L = []
    L.append("# Metrik Klasifikasi — CloudDenseNet (7 kelas awan GCD)")
    L.append("")
    ckpt = meta.get("checkpoint", "—")
    best_f1 = meta.get("best_f1")
    best_ep = meta.get("best_epoch")
    ket_ckpt = f"`{ckpt}`"
    if best_ep is not None and best_f1 is not None:
        ket_ckpt += f" (epoch {best_ep}, best val F1-macro = {best_f1:.4f})"
    elif best_f1 is not None:
        ket_ckpt += f" (best val F1-macro = {best_f1:.4f})"
    L.append(f"**Tanggal:** {tanggal}  ")
    L.append(f"**Run:** `{run_name}`  ")
    L.append(f"**Checkpoint:** {ket_ckpt}  ")
    L.append(f"**Device evaluasi:** {device}  ")
    L.append("**Metrik:** precision, recall, F1-score per kelas + accuracy "
             "(macro & weighted average).")
    L.append("")
    L.append("> Dihitung dengan `sklearn.metrics.precision_recall_fscore_support` pada "
             "prediksi `argmax`. Pipeline (transform/normalisasi) identik dengan "
             "pembuatan confusion matrix.")
    L.append("")
    L.append("---")
    L.append("")

    # --- Waktu pelatihan --------------------------------------------------- #
    mulai = _as_dt(meta.get("started_at"))
    selesai = _as_dt(meta.get("finished_at"))
    L.append("## Waktu Pelatihan")
    L.append("")
    L.append("| Keterangan | Nilai |")
    L.append("|---|---|")
    L.append(f"| Waktu mulai | {_fmt_waktu(mulai or meta.get('started_at'))} |")
    L.append(f"| Waktu selesai | {_fmt_waktu(selesai or meta.get('finished_at'))} |")
    if mulai and selesai:
        L.append(f"| Durasi (rentang mulai→selesai) | {ringkas_durasi((selesai - mulai).total_seconds())} |")
    komputasi = meta.get("detik_komputasi")
    if komputasi is None and meta.get("log_csv"):
        komputasi = _baca_detik_epoch(meta["log_csv"])
    if komputasi:
        L.append(f"| Total waktu komputasi (Σ per-epoch) | {ringkas_durasi(komputasi)} |")
    if meta.get("total_epoch") is not None:
        L.append(f"| Total epoch dijalankan | {meta['total_epoch']} |")
    resumed = bool(meta.get("resume"))
    L.append(f"| Dilanjutkan dari checkpoint (resume) | {'Ya' if resumed else 'Tidak'} |")
    L.append("")

    sesi = meta.get("sesi") or []
    if resumed or len(sesi) > 1:
        L.append("**Keterangan resume:** pelatihan sempat berhenti lalu dilanjutkan "
                 "dari checkpoint terakhir (`..._last.pth`). Bobot, optimizer, "
                 "`GradScaler`, dan riwayat (history) dipulihkan sehingga kurva & "
                 "epoch berlanjut tanpa terputus.")
        if sesi:
            L.append("")
            L.append("| Sesi | Waktu mulai sesi | Mulai dari epoch | Resume |")
            L.append("|---:|---|---:|---|")
            for i, s in enumerate(sesi, 1):
                L.append(f"| {i} | {_fmt_waktu(_as_dt(s.get('mulai')) or s.get('mulai'))} "
                         f"| {s.get('start_epoch', '—')} "
                         f"| {'Ya' if s.get('resume') else 'Tidak'} |")
        L.append("")
    if meta.get("catatan"):
        L.append(f"> {meta['catatan']}")
        L.append("")
    L.append("---")
    L.append("")

    # --- Hardware ---------------------------------------------------------- #
    hw = meta.get("hardware") or kumpulkan_info_hardware(device)
    L.append("## Spesifikasi Hardware & Lingkungan")
    L.append("")
    L.append("| Komponen | Spesifikasi |")
    L.append("|---|---|")
    L.append(f"| Sistem operasi | {hw.get('os', '—')} ({hw.get('mesin', '—')}) |")
    L.append(f"| Prosesor (CPU) | {hw.get('prosesor', '—')} — {hw.get('cpu_logis', '—')} core logis |")
    if hw.get("ram_gb"):
        L.append(f"| RAM total | {hw['ram_gb']} GB |")
    if hw.get("gpu"):
        L.append(f"| GPU | {hw['gpu']} ({hw.get('gpu_mem_gb', '—')} GB VRAM) |")
    else:
        L.append("| GPU | — (evaluasi/latih di CPU) |")
    L.append(f"| CUDA | {hw.get('cuda_versi', '—')} (cuDNN {hw.get('cudnn', '—')}) |")
    L.append(f"| Python | {hw.get('python', '—')} |")
    L.append(f"| PyTorch | {hw.get('torch', '—')} + torchvision {hw.get('torchvision', '—')} |")
    L.append("")
    L.append("---")
    L.append("")

    # --- Konfigurasi parameter utama (item 1) ------------------------------ #
    L.append(_ringkas_konfig_md())
    L.append("")
    L.append("---")
    L.append("")

    # --- Metrik + CM (item 2) ---------------------------------------------- #
    if metrik:
        L.append("## Metrik Klasifikasi & Confusion Matrix")
        L.append("")
    peta = [("val", "Validasi", "cm_val"), ("test", "Test (held-out)", "cm_test")]
    for split, judul, key_cm in peta:
        if split not in metrik:
            continue
        rows, summ = metrik[split]
        L.append(f"### {judul}")
        L.append("")
        L.append(_tabel_metrik_md(rows, summ))
        L.append("")
        if gambar.get(key_cm):
            fn = gambar[key_cm]
            L.append(f"**Confusion matrix {judul.lower()}** (kiri: jumlah, kanan: "
                     f"ternormalisasi/recall):")
            L.append("")
            L.append(f"![Confusion matrix {judul}]({fn})")
            L.append("")
            L.append(f"[Buka gambar CM {judul}]({fn})")
            L.append("")
        L.append("---")
        L.append("")

    # --- Kurva loss & akurasi (item 3) ------------------------------------- #
    if gambar.get("kurva_loss") or gambar.get("kurva_akurasi"):
        L.append("## Grafik Pelatihan")
        L.append("")
        L.append("> Dua kanvas terpisah: satu khusus **Loss** (train vs val), satu "
                 "khusus **Akurasi** (train vs val). Garis putus-putus vertikal = "
                 "batas antar-tahap `PHASES`.")
        L.append("")
        if gambar.get("kurva_loss"):
            fn = gambar["kurva_loss"]
            L.append("### Loss — Train vs Validasi")
            L.append("")
            L.append(f"![Kurva Loss Train vs Val]({fn})")
            L.append("")
            L.append(f"[Buka grafik Loss]({fn})")
            L.append("")
        if gambar.get("kurva_akurasi"):
            fn = gambar["kurva_akurasi"]
            L.append("### Akurasi — Train vs Validasi")
            L.append("")
            L.append(f"![Kurva Akurasi Train vs Val]({fn})")
            L.append("")
            L.append(f"[Buka grafik Akurasi]({fn})")
            L.append("")
        L.append("---")
        L.append("")

    # --- Grad-CAM benar & salah (item 4) ----------------------------------- #
    if gambar.get("gradcam_benar") or gambar.get("gradcam_salah"):
        sp = gambar.get("gradcam_split", "test")
        L.append("## Grad-CAM — Prediksi Benar & Salah")
        L.append("")
        L.append(f"> Sampel dari split **{sp}**. Tiap baris: citra asli + overlay "
                 "Grad-CAM. Judul menampilkan label asli, kelas prediksi, dan "
                 "**skor kepercayaan (confidence)** model, serta penanda "
                 "`[OK]`/`[SALAH]`.")
        L.append("")
        if gambar.get("gradcam_benar"):
            fn = gambar["gradcam_benar"]
            L.append("### Prediksi Benar")
            L.append("")
            L.append(f"![Grad-CAM prediksi benar]({fn})")
            L.append("")
            L.append(f"[Buka Grad-CAM benar]({fn})")
            L.append("")
        if gambar.get("gradcam_salah"):
            fn = gambar["gradcam_salah"]
            L.append("### Prediksi Salah")
            L.append("")
            L.append(f"![Grad-CAM prediksi salah]({fn})")
            L.append("")
            L.append(f"[Buka Grad-CAM salah]({fn})")
            L.append("")
        L.append("---")
        L.append("")

    L.append(f"_Laporan dibuat otomatis oleh `clouddensenet.report` pada "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}._")
    L.append("")
    return "\n".join(L)
