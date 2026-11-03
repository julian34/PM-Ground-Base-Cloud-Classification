# Cara Menjalankan — CloudDenseNet

Runbook ringkas untuk menjalankan pipeline: persiapan, training biasa, Stratified
K-Fold CV, ablasi otomatis, model final, dan troubleshooting.

> Penjelasan konsep ada di `docs/STRATIFIED_KFOLD_CV.md`; laporan penerapan di
> `docs/2026-11-04/LAPORAN_KFOLD_DAN_GRADIENT_ACCUMULATION.md`. Sumber kebenaran
> hyperparameter: `src/clouddensenet/config.py`.

---

## 0. Prasyarat (sekali saja)

```powershell
# Aktifkan virtual env
.\.venv\Scripts\Activate.ps1

# (bila belum) pasang dependensi + paket
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Cek GPU terdeteksi (harus True untuk training penuh)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Siapkan dataset (ekstrak `GCD.zip` + split stratified → `data/processed/`):

```powershell
python scripts/split_dataset.py            # hasil: data/processed/{train,val,test}/<kelas>/
```

Cek cepat tanpa GPU/dataset (membangun model + 1 forward pass):

```powershell
pytest
```

---

## 1. Pilih cara menjalankan

| Tujuan | Perintah inti | Waktu* |
|--------|---------------|--------|
| **A. Satu model cepat** (holdout 15%) | `python scripts/train.py` | ~24 jam |
| **B. CV satu konfigurasi** (dapat `mean ± std`) | `python scripts/train_kfold.py --folds 3` | ~27 jam |
| **C. Ablasi otomatis** (BR × STRONG_AUG, 3 run + banding) | `.\scripts\run_ablation.ps1` | ~81 jam |

\* Perkiraan pada asumsi ~18 menit/epoch; sesuaikan dengan GPU Anda. Pastikan
`DEBUG = False` di `config.py` untuk training penuh (True = uji cepat 1–3 batch).

---

## 2. A — Training biasa (holdout)

```powershell
python scripts/train.py                         # default dari config.py (80 epoch)
python scripts/train.py --run-name cloud --epochs 80
```

- Checkpoint terbaik → `artifacts/checkpoints/<run>_Versi01.pth`
- Checkpoint terakhir (auto-resume) → `<run>_Versi01_last.pth` (`--no-resume` untuk mematikan)
- Kurva + confusion matrix → `artifacts/results/`

---

## 3. B — Stratified K-Fold CV (satu konfigurasi)

```powershell
python scripts/train_kfold.py --folds 3                 # 3-fold, jadwal 30 epoch/fold
python scripts/train_kfold.py --folds 5 --accum-steps 4 # 5-fold + gradient accumulation
```

Keluaran:
- `artifacts/checkpoints/<run>_fold{k}_Versi01.pth` — model terbaik tiap fold
- `artifacts/results/kfold_<run>.json` — `mean_f1`, `std_f1`, `oof_macro_f1`, flag arsitektur
- `artifacts/results/kfold_<run>_oof.npz` — `y_true`, `y_pred` out-of-fold

Angka kunci di akhir: **`F1 = mean ± std`** (estimasi + ketidakpastian). CV berjalan
hanya di pool `train/`; `test/` tetap holdout final.

---

## 4. C — Ablasi otomatis (disarankan untuk eksperimen)

Membandingkan `USE_BR` (Biru/Merah) dan `STRONG_AUG` lewat 3 run CV berurutan, lalu
otomatis membandingkan. Flag diatur via environment variable (tanpa menulis-ulang
`config.py`).

```powershell
.\scripts\run_ablation.ps1                    # baseline, br_off, aug_on -> compare
.\scripts\run_ablation.ps1 -AccumSteps 4      # bila CUDA out of memory
.\scripts\run_ablation.ps1 -Folds 5
```

Bila skrip diblokir Windows (*execution policy*):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Run yang dijalankan:

| Run | `USE_BR` | `STRONG_AUG` | Guna |
|-----|:--------:|:------------:|------|
| `baseline` | on | off | acuan |
| `br_off` | **off** | off | isolasi kontribusi Biru/Merah |
| `aug_on` | on | **on** | isolasi efek augmentasi kuat |

Log per run: `artifacts/results/ablation_<nama>.log`.

---

## 5. Membaca & membandingkan hasil

```powershell
python scripts/compare_kfold.py                       # semua kfold_*.json
python scripts/compare_kfold.py baseline br_off aug_on # run tertentu; pertama = baseline
```

Aturan keputusan (otomatis dicetak):
- **NYATA** bila `|Δmean| > (std_a + std_b)` → aman memilih yang mean lebih tinggi.
- **DERAU** bila rentang ±std tumpang-tindih → perbedaan belum tentu nyata; pilih yang
  lebih sederhana/stabil/murah.

Contoh keputusan: `br_off` NYATA lebih buruk → pertahankan `USE_BR=True`;
`aug_on` hanya DERAU → biarkan `STRONG_AUG=False`.

---

## 6. Model final (produk deployment)

Setelah menentukan konfigurasi pemenang, set nilainya di `src/clouddensenet/config.py`
(sebagai default), lalu latih **sekali** di seluruh `train/` dengan jadwal penuh:

```powershell
python scripts/train.py --run-name cloud_final        # 80 epoch, seluruh train/
```

Terakhir, evaluasi **sekali** di `test/` (9.000 citra) — itu angka final yang
dilaporkan (bukan angka CV). Jangan menukar angka CV dengan angka test.

---

## 7. Gradient Accumulation (VRAM terbatas)

Batch efektif = `BATCH_SIZE × ACCUM_STEPS` tanpa menaikkan memori puncak. Naikkan saat
muncul *CUDA out of memory*, terutama di fase `full` (defrost).

```powershell
# Global (train.py & train_kfold.py): edit config.py -> ACCUM_STEPS = 4
# Khusus CV / ablasi: lewat argumen
python scripts/train_kfold.py --accum-steps 4
.\scripts\run_ablation.ps1 -AccumSteps 4
```

`ACCUM_STEPS = 1` (default) = perilaku lama (step tiap batch).

---

## 8. Troubleshooting

| Gejala | Penyebab / solusi |
|--------|-------------------|
| `CUDA out of memory` | naikkan `ACCUM_STEPS` (2→4→8); **jangan** naikkan `BATCH_SIZE` |
| `loss NaN/Inf` | ditangani otomatis (batch di-skip, BN dipulihkan; fold di-skip). >20 batch NaN → `RuntimeError`; periksa SCConv (fp32) |
| Skrip `.ps1` diblokir | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Teks `±`/`Δ` error di konsol | skrip sudah paksa UTF-8 stdout; pakai Windows Terminal bila perlu |
| `ModuleNotFoundError: clouddensenet` | jalankan `pip install -e .`, atau pastikan dijalankan dari root repo |
| Dataset tak ditemukan | jalankan `python scripts/split_dataset.py` dulu (lihat `data/README.md`) |
| Training sangat lambat / CPU | `DEBUG = False` + pastikan `torch.cuda.is_available()` True |

---

## 9. Dokumen terkait

- `docs/STRATIFIED_KFOLD_CV.md` — konsep & desain K-Fold CV.
- `docs/2026-11-04/LAPORAN_KFOLD_DAN_GRADIENT_ACCUMULATION.md` — laporan penerapan.
- `docs/EDA_AUGMENTASI.md` — distribusi kelas & augmentasi.
- `docs/ARSITEKTUR_CLOUDDENSENET.md` — arsitektur model.
- `README.md` · `data/README.md` — ringkasan proyek & dataset.
