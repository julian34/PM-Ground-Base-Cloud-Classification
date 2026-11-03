# Data — Dataset GCD (7 kelas awan)

Folder ini **tidak** ikut di-commit (lihat `.gitignore`). Dataset diekstrak di sini
secara manual.

## Dataset yang tersedia

`data/raw/GCD.zip` sudah ada di repo lokal. Isinya diekstrak menjadi:

```
data/raw/GCD/
├── train/   (7 subfolder kelas, ~10.017 gambar)
└── test/    (7 subfolder kelas)
```

> Catatan: zip ini **hanya** berisi `train/` dan `test/` — **tidak ada `val/`**.
> `scripts/train.py` otomatis meng-carve validasi dari `train/` secara
> *stratified* & *seeded* (default `VAL_SPLIT = 0.15` di `config.py`). Jika Anda
> menyediakan folder `data/raw/GCD/val/`, itu yang dipakai.

## 7 kelas (urutan ImageFolder)

| idx | folder |
|----:|--------|
| 0 | `1_cumulus` |
| 1 | `2_altocumulus` |
| 2 | `3_cirrus` |
| 3 | `4_clearsky` |
| 4 | `5_stratocumulus` |
| 5 | `6_cumulonimbus` |
| 6 | `7_mixed` |

## Cara ekstrak

```powershell
# PowerShell
Expand-Archive data\raw\GCD.zip -DestinationPath data\raw\
```
```bash
# Git Bash
unzip data/raw/GCD.zip -d data/raw/
```

Hasil akhir harus: `data/raw/GCD/train/<kelas>/*.jpg` dan `data/raw/GCD/test/...`.

- `data/raw/`       — dataset mentah (jangan diubah).
- `data/processed/` — turunan opsional (split ulang, cache, dll).
