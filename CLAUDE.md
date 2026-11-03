# CLAUDE.md

Panduan untuk Claude Code (dan agen lain) saat bekerja di repo ini.
Ditulis dalam Bahasa Indonesia mengikuti gaya kode & dokumentasi proyek.

## Ringkasan proyek

**CloudDenseNet** — DenseNet-121 yang dimodifikasi untuk klasifikasi citra langit
ke **7 kelas awan** (dataset GCD). Repo ini adalah refaktor lokal dari notebook
Colab `Copy_of_CloudDenseNet.ipynb` menjadi paket Python `clouddensenet` yang bisa
dijalankan di mesin lokal (dibangun untuk GPU NVIDIA / CUDA 12.4, dengan fallback CPU).

Modifikasi arsitektur vs DenseNet-121 standar: **SEBlock** (squeeze-and-excitation),
**SCConv** (konvolusi cosine ternormalisasi, dipaksa fp32 agar stabil di AMP),
**SpatialAttnBR** (atensi spasial + channel λ rasio Biru/Merah, sadar-fisika),
Top Block ringkas, **Focal Loss** (γ=2.0), dan fine-tuning ImageNet bertahap.

## Stack

- **Bahasa:** Python ≥ 3.10 (venv terpasang di `.venv/`, interpreter cpython-3.11).
- **Deep learning:** PyTorch `2.5.1` + torchvision `0.20.1` (build CUDA 12.4; `cpu` juga didukung).
- **Numerik/data:** numpy (`>=1.26,<2.1`), pandas, scikit-learn.
- **Plot:** matplotlib, seaborn. **Citra:** Pillow. **Lain:** tqdm, PyYAML.
- **Kemasan:** paket `clouddensenet` dengan src-layout (`src/clouddensenet/`),
  installable via `pip install -e .` (`pyproject.toml`, setuptools).
- **Testing:** pytest (`[project.optional-dependencies] dev`).
- **Platform dev utama:** Windows 11, shell PowerShell. Tool Bash (POSIX) juga tersedia.

## Struktur folder utama

```
.
├── src/clouddensenet/        # paket inti (sumber kebenaran arsitektur)
│   ├── config.py             # SEMUA hyperparameter, flag arsitektur, & path
│   ├── model.py              # CloudDenseNet (arsitektur utama)
│   ├── blocks.py             # DenseLayer, DenseBlock1-4, Transition1-3, TopBlock
│   ├── attention.py          # SEBlock, SpatialAttn(BR), EdgeLayer
│   ├── conv.py               # SCConv
│   ├── spectral.py           # peta λ rasio Biru/Merah (fitur sadar-fisika)
│   ├── losses.py             # FocalLoss + init (LeCun uniform)
│   ├── pretrained.py         # muat bobot ImageNet + nol-kan channel ND
│   ├── finetune.py           # pembekuan/unfreeze bertahap
│   ├── data.py               # transform, ImageFolder DataLoader, siapkan_batch
│   ├── engine.py             # loop train/eval, jadwal tahap, pengaman NaN
│   └── viz.py                # kurva training & confusion matrix
├── scripts/
│   ├── train.py              # entrypoint training (baca config.py)
│   └── split_dataset.py      # ekstrak GCD.zip + split train/val/test (stratified)
├── tests/test_forward.py     # smoke test (tanpa dataset/GPU)
├── data/                     # dataset — TIDAK di-commit (lihat .gitignore)
│   ├── raw/GCD.zip           # arsip asli (JANGAN dipindah/hapus)
│   ├── raw/GCD/{train,test}  # hasil ekstrak (cadangan, jangan diubah)
│   └── processed/{train,val,test}/<kelas>/   # dataset siap latih (split jadi)
├── artifacts/{checkpoints,results}/   # keluaran training — tidak di-commit
├── configs/                  # konfigurasi tersimpan (opsional, saat ini kosong)
├── Backup/                   # SENGAJA TIDAK DISENTUH / tidak di-commit
├── Copy_of_CloudDenseNet.ipynb   # notebook asal — referensi, JANGAN diubah
├── pyproject.toml · requirements.txt · README.md · .gitignore
```

7 kelas (urutan ImageFolder, indeks 0–6):
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.

## Perintah penting

Aktifkan venv dulu:

```powershell
.\.venv\Scripts\Activate.ps1          # PowerShell
```
```bash
source .venv/Scripts/activate         # Git Bash
```

Instalasi dependensi (PyTorch build CUDA 12.4; ganti ke `.../whl/cpu` bila tanpa GPU):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .                      # pasang paket clouddensenet (disarankan)
```

Siapkan dataset (ekstrak + split stratified 15% val, seed 42 → `data/processed/`):

```bash
python scripts/split_dataset.py                  # copy (raw tetap utuh)
python scripts/split_dataset.py --move --force   # pindahkan train, timpa output
```

Training:

```bash
python scripts/train.py                                   # default dari config.py
python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud
```

- Checkpoint terbaik → `artifacts/checkpoints/<run>_Versi01.pth`
- Checkpoint terakhir (resume) → `..._last.pth` (resume otomatis; `--no-resume` untuk matikan)
- Kurva & confusion matrix → `artifacts/results/`

## Testing

- Konfigurasi di `pyproject.toml` → `[tool.pytest.ini_options]`:
  `pythonpath = ["src"]`, `testpaths = ["tests"]`. Jadi cukup jalankan `pytest` dari root.
- Smoke test `tests/test_forward.py` membangun model & menjalankan satu forward pass
  (tanpa dataset/GPU) — memverifikasi shape output `(N, 7)` dan key `return_all=True`.

```bash
pytest                       # memakai config pyproject (src di path otomatis)
python tests/test_forward.py # mode skrip; cetak "forward pass sukses" + jumlah parameter
```

Jalankan `pytest` sebagai cek cepat setelah mengubah kode di `src/clouddensenet/`.

## Konvensi & aturan main

- **Sumber kebenaran hyperparameter:** `src/clouddensenet/config.py`. Ubah di sini,
  bukan hard-code di script. Set `DEBUG = True` untuk uji cepat 1–3 batch/epoch di CPU.
- **Path dataset default:** `DATA_ROOT = data/processed` (ImageFolder: `train/val/test`).
  Jika `val/` tersedia, itu yang dipakai; jika tidak, `data.py` carve val dari train
  (stratified, seeded) sesuai `VAL_SPLIT = 0.15`.
- **Bahasa:** komentar, docstring, dan sebagian nama fungsi memakai Bahasa Indonesia
  (mis. `muat_imagenet`, `bekukan_imagenet`, `siapkan_batch`). Pertahankan gaya ini.
- **JANGAN sentuh:** folder `Backup/`, file `GCD.zip`, dan notebook
  `Copy_of_CloudDenseNet.ipynb` (dibiarkan sebagai referensi).
- **JANGAN commit** data & artefak besar: `data/raw/`, `data/processed/`,
  `artifacts/checkpoints/`, `artifacts/results/`, `*.pth`, `*.pt`, `Backup/`
  (sudah di `.gitignore`).
- Reproducibility dijaga lewat `SEED = 42` (`set_seed` di `data.py`).
