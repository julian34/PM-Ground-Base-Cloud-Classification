# Panduan Pemasangan Awal — CloudDenseNet

> Panduan operasional untuk **eksperimen baru yang terpisah**. Ini bukan catatan
> instalasi historis run datasetB. Hasil dan koreksi perangkat penelitian ada di
> [HASIL_DATASET_B.md](HASIL_DATASET_B.md). Konfigurasi aktif dapat berbeda dari run,
> khususnya ukuran citra: laporan 224×224, kode saat revisi 512.

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

## Langkah 5 — Siapkan salinan datasetB untuk run baru

Dataset dan artefak tidak disertakan pada clone. Gunakan arsip `data/raw/dataset_b.zip`
yang sudah diverifikasi identitasnya. Untuk membaca hasil penelitian yang ada,
tidak perlu mengekstrak atau membagi ulang dataset.

Contoh berikut hanya untuk membuat **split baru di lokasi berbeda**; jangan
menimpa `data/processed/dataset_b` yang terkait hasil penelitian:

```bash
python scripts/split_dataset_experiment.py --zip data/raw/dataset_b.zip --out data/processed/dataset_b_replikasi --split 0.70 0.20 0.10 --seed 42
```

Simpan manifest anggota split beserta hash citra. Seed dan rasio yang sama saja
belum menjamin bahwa anggota split identik dengan run historis. Hitungan acuan
run ada pada [DATASET_SPLIT.md](DATASET_SPLIT.md).

## Langkah 6 — Jalankan eksperimen baru setelah konfigurasi diperiksa

Bandingkan semua pengaturan dengan [HASIL_DATASET_B.md](HASIL_DATASET_B.md),
terutama masukan **224×224**, cabang CloudViT dan head 1280→512→128→32→16→7.
`config.py` saat revisi memakai 512; menjalankan perintah berikut tanpa
menyesuaikan konfigurasi tidak mereproduksi pengaturan yang dicatat laporan.
Simpan salinan konfigurasi dan versi kode sebelum menjalankan training.

```bash
python scripts/train.py --data-root data/processed/dataset_b_replikasi --epochs 80 --seed 42 --run-name datasetB_replikasi_baru --no-resume
```

Nama run harus unik dan belum dipakai agar checkpoint penelitian asli tidak
tertimpa. Perintah ini adalah contoh operasional, tidak dijalankan saat revisi
dokumentasi, dan tidak menjamin hasil metrik identik.

Checkpoint dan keluaran baru berada di `artifacts/checkpoints` dan
`artifacts/results`. Jangan arahkan keluaran baru ke folder hasil acuan
`datasetB_laporan_19-11-2026_0251`.

## Ringkasan penggunaan

Untuk penulisan laporan, baca artefak yang sudah ada. Untuk eksperimen baru,
siapkan lingkungan, cocokkan konfigurasi, gunakan direktori/nama run berbeda,
dan catat manifest data serta perangkat yang benar-benar digunakan.

## Troubleshooting ringkas

- **`torch` tidak mendeteksi GPU:** cek dengan `python -c "import torch; print(torch.cuda.is_available())"`.
  Jika `False` padahal ada GPU, pastikan terpasang build cu124, bukan cpu.
- **`Activate.ps1` ditolak di PowerShell:** jalankan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  sekali, lalu aktifkan ulang.
- **`ModuleNotFoundError: clouddensenet`:** pastikan sudah `pip install -e .` di dalam venv yang aktif.
- **Training tidak menemukan data:** pastikan `data/processed/dataset_b_replikasi/{train,val,test}` sudah terbentuk dari
  Langkah 5, atau arahkan `--data-root` ke lokasi yang benar.
