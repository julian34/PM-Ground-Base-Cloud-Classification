# Panduan Pemasangan Awal — CloudDenseNet

*Langkah menyiapkan proyek dari hasil clone (untuk pengguna baru)*

Dokumen ini menjelaskan cara memasang proyek CloudDenseNet dari awal ketika seseorang
baru saja meng-clone repository. Beberapa folder (lingkungan virtual, dataset, dan hasil
training) sengaja **TIDAK** ikut di-commit ke Git, sehingga harus dibuat atau disiapkan
sendiri setelah clone — ini dijelaskan di bawah.

## Prasyarat

| Komponen | Keterangan |
|---|---|
| Git | Untuk melakukan clone repository. |
| Python ≥ 3.10 | Disarankan 3.11 (interpreter yang dipakai proyek). |
| GPU NVIDIA + CUDA 12.4 | Opsional. Tanpa GPU, pakai build CPU (lihat Langkah 3). |
| Koneksi internet | Untuk mengunduh PyTorch & dependensi dari PyPI. |

## Langkah 1 — Clone repository

Buka terminal (PowerShell atau Git Bash), lalu clone dan masuk ke folder proyek:

```bash
git clone https://github.com/julian34/PM-Ground-Base-Cloud-Classification.git
cd PM-Ground-Base-Cloud-Classification
```

## Langkah 2 — Buat & aktifkan virtual environment

> ⚠️ Folder `.venv/` **TIDAK** ikut di-clone (ada di `.gitignore`). Jadi setelah clone,
> lingkungan virtual harus dibuat sendiri — jangan mencari `.venv` yang sudah jadi.

Buat venv dengan Python 3.10+ lalu aktifkan sesuai shell yang dipakai:

**PowerShell (Windows):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Setelah aktif, nama `(.venv)` akan muncul di awal baris terminal. Perbarui pip dulu
(disarankan):

```bash
python -m pip install --upgrade pip
```

## Langkah 3 — Pasang dependensi

Instal dependensi dari `requirements.txt`. Pilih salah satu sesuai ketersediaan GPU.

**A. Dengan GPU NVIDIA (build CUDA 12.4):**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

**B. Tanpa GPU (build CPU):**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

`pip install -e .` memasang paket `clouddensenet` secara editable (disarankan, agar import
dan skrip berjalan dari mana saja).

## Langkah 4 — Uji cepat (tanpa dataset)

Pastikan pemasangan berhasil dengan smoke test — tidak butuh dataset maupun GPU:

```bash
pytest
# atau
python tests/test_forward.py
```

Keluaran harus memuat `forward pass sukses` beserta jumlah parameter model. Jika ini
berhasil, instalasi inti sudah benar.

## Langkah 5 — Siapkan dataset

> ⚠️ Dataset `data/raw/GCD.zip` dan folder `data/` **TIDAK** ikut di-clone (ada di
> `.gitignore`). Pengguna baru harus memperoleh `GCD.zip` secara terpisah dari pemilik
> proyek, lalu meletakkannya di `data/raw/GCD.zip`.

Setelah file `GCD.zip` tersedia di `data/raw/`, ekstrak dan split menjadi train/val/test:

**Ekstrak (PowerShell):**

```powershell
Expand-Archive data\raw\GCD.zip -DestinationPath data\raw\
```

**Split stratified (15% val, seed 42) ke `data/processed/`:**

```bash
python scripts/split_dataset.py
```

Dataset punya 7 kelas (indeks 0–6):

- `1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`

## Langkah 6 — Jalankan training

Jalankan training dengan konfigurasi default dari `config.py`, atau atur argumen sendiri:

```bash
python scripts/train.py
# atau atur argumen:
python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud
```

- Checkpoint terbaik → `artifacts/checkpoints/<run>_Versi01.pth`
- Checkpoint terakhir (bisa di-resume) → `..._last.pth`
- Kurva & confusion matrix → `artifacts/results/`

TensorBoard aktif otomatis saat training; buka dasbor di:

```
http://localhost:6006
```

Semua hyperparameter ada di `src/clouddensenet/config.py`. Set `DEBUG = True` untuk uji
cepat 1–3 batch/epoch di CPU.

## Ringkasan cepat

1. Clone repo dan `cd` ke foldernya.
2. Buat `.venv` (Python 3.10+) lalu aktifkan.
3. `pip install -r requirements.txt` (pilih index cu124 atau cpu) + `pip install -e .`
4. Jalankan `pytest` untuk verifikasi.
5. Dapatkan `GCD.zip`, ekstrak, lalu `python scripts/split_dataset.py`.
6. `python scripts/train.py` dan pantau di http://localhost:6006.

## Troubleshooting ringkas

- **`torch` tidak mendeteksi GPU:** cek dengan `python -c "import torch; print(torch.cuda.is_available())"`.
  Jika `False` padahal ada GPU, pastikan terpasang build cu124, bukan cpu.
- **`Activate.ps1` ditolak di PowerShell:** jalankan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  sekali, lalu aktifkan ulang.
- **`ModuleNotFoundError: clouddensenet`:** pastikan sudah `pip install -e .` di dalam venv yang aktif.
- **Training tidak menemukan data:** pastikan `data/processed/{train,val,test}` sudah terbentuk dari
  Langkah 5, atau arahkan `--data-root` ke lokasi yang benar.
