# CloudDenseNet — Klasifikasi 7 Kelas Awan

DenseNet-121 yang dimodifikasi untuk mengklasifikasikan citra langit ke dalam
**7 kelas awan** (dataset GCD). Proyek ini merefaktor notebook Colab
`Copy_of_CloudDenseNet.ipynb` menjadi paket Python yang rapi dan bisa dijalankan
lokal (GPU NVIDIA terdeteksi: dibangun untuk CUDA 12.4).

## Apa yang membuatnya beda dari DenseNet-121 standar

| Komponen | Fungsi |
|---|---|
| **SEBlock** (Squeeze-and-Excitation) | Atensi antar-channel di tiap bottleneck & tiap block; diinisialisasi ≈ identitas. |
| **SCConv** | Konvolusi *cosine* ter-normalisasi dengan skala τ per-channel (dipaksa fp32, stabil di bawah AMP). |
| **SpatialAttnBR** | Atensi spasial + 1 channel **λ rasio Biru/Merah** → sadar-fisika (membedakan langit vs awan). |
| **Top Block** | GAP → FC(1024→32) → BN → Dropout → FC(32→7). |
| **Focal Loss** | Menangani ketidakseimbangan kelas (γ = 2.0). |
| **Fine-tuning 5 tahap** | Bobot ImageNet dibekukan lalu dibuka bertahap (lihat `PHASES`). |

## Struktur folder

```
.
├── README.md                 # file ini
├── requirements.txt          # dependensi (PyTorch CUDA dipisah)
├── pyproject.toml            # paket installable (pip install -e .)
├── .gitignore
├── Copy_of_CloudDenseNet.ipynb   # notebook asal (referensi, tidak diubah)
├── Backup/                   # TIDAK DISENTUH
├── src/clouddensenet/        # paket inti
│   ├── config.py             # semua hyperparameter & path
│   ├── model.py              # CloudDenseNet (arsitektur utama)
│   ├── blocks.py             # DenseLayer, DenseBlock1-4, Transition1-3, TopBlock
│   ├── attention.py          # SEBlock, SpatialAttn(BR), EdgeLayer
│   ├── conv.py               # SCConv
│   ├── spectral.py           # peta λ rasio Biru/Merah
│   ├── losses.py             # FocalLoss + init LeCun
│   ├── pretrained.py         # muat bobot ImageNet + nol-kan channel ND
│   ├── finetune.py           # pembekuan bertahap (5 tahap)
│   ├── data.py               # transform, DataLoader, persiapan batch
│   ├── engine.py             # loop train/eval, jadwal tahap, pengaman NaN
│   ├── tblog.py              # TensorBoard: pasang & aktif otomatis saat training
│   └── viz.py                # kurva training & confusion matrix
├── scripts/train.py          # entrypoint training
├── tests/test_forward.py     # smoke test (tanpa dataset/GPU)
├── configs/                  # (opsional) konfigurasi tersimpan
├── data/                     # dataset (lihat data/README.md) — tidak di-commit
│   ├── raw/ · processed/
└── artifacts/                # keluaran training — tidak di-commit
    ├── checkpoints/ · results/
```

## Dokumentasi

Dokumen penting untuk memahami & mengembangkan proyek ini:

| Dokumen | Isi |
|---|---|
| **[`docs/Panduan_Instalasi_Clone.md`](docs/Panduan_Instalasi_Clone.md)** | Langkah pasang dari hasil clone (venv, dependensi, dataset, training). |
| **[`CONTRIBUTING.md`](CONTRIBUTING.md)** | Aturan kolaborasi GitHub: branch, commit, PR, review, Issue. |
| **[`CLAUDE.md`](CLAUDE.md)** | Konvensi kode, aturan arsitektur, perintah penting. |
| **[`TRAINING.md`](TRAINING.md)** | Alur training lengkap, jadwal tahap, dan penyiapan data (§5). |
| **[`docs/ARSITEKTUR_CLOUDDENSENET.md`](docs/ARSITEKTUR_CLOUDDENSENET.md)** | Arsitektur CloudDenseNet secara menyeluruh. |
| **[`docs/PENJELASAN_MODEL_DETAIL.md`](docs/PENJELASAN_MODEL_DETAIL.md)** | Penjelasan detail tiap komponen model. |
| **[`docs/REFERENSI_KOMPONEN_DETAIL.md`](docs/REFERENSI_KOMPONEN_DETAIL.md)** | Referensi komponen (SEBlock, SCConv, SpatialAttnBR, dll). |
| **[`docs/DATASET_SPLIT.md`](docs/DATASET_SPLIT.md)** | Logika split train/val/test (stratified, seeded). |
| **[`docs/EDA_AUGMENTASI.md`](docs/EDA_AUGMENTASI.md)** | Analisis data (EDA) & strategi augmentasi. |
| **[`data/README.md`](data/README.md)** | Struktur folder dataset & angka per kelas. |
| **[`model_ui/README.md`](model_ui/README.md)** | Web UI uji model: upload gambar → prediksi + confidence + Grad-CAM. |

## Instalasi

Virtual environment khusus sudah dibuat di `.venv/`. Aktifkan:

```powershell
# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
```
```bash
# Git Bash
source .venv/Scripts/activate
```

Dependensi diinstal dari `requirements.txt` (PyTorch build CUDA 12.4):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .            # pasang paket clouddensenet (opsional tapi disarankan)
```

> Mesin tanpa GPU NVIDIA: ganti index URL menjadi
> `https://download.pytorch.org/whl/cpu`.

## Uji cepat (tanpa dataset)

```bash
python tests/test_forward.py
# atau
pytest
```

Harus mencetak `forward pass sukses` beserta jumlah parameter model.

## Dataset

Proyek ini memakai **GCD — TJNU Ground-based Cloud Dataset** (Tianjin Normal University):
citra langit berbasis-tanah yang dilabeli ke **7 kelas awan**.

- 🔗 **Sumber dataset:** <https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset>
- 📄 **Perjanjian penggunaan (WAJIB dibaca & dipatuhi):**
  [GCD Agreement.pdf](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset/blob/main/GCD%20Agreement.pdf)

> ⚠️ **Lisensi & etika.** GCD dirilis dengan *Agreement* tersendiri (umumnya untuk riset
> non-komersial). Baca dan patuhi `GCD Agreement.pdf` sebelum memakai/mendistribusikan data.
> Karena itu **dataset TIDAK ikut di-commit** ke repo (lihat `.gitignore`) — setiap pengguna
> memperolehnya sendiri dari sumber resmi di atas.

### Cara memasang dataset ke proyek

1. **Peroleh dataset** dari repo/sumber resmi GCD di atas (ikuti instruksi pada *Agreement*;
   bila perlu, hubungi pemilik proyek untuk salinan `GCD.zip` yang dipakai di sini).
2. **Letakkan arsipnya** tepat di path berikut (dibaca otomatis oleh skrip split):

   ```
   data/raw/GCD.zip
   ```

3. **Ekstrak + split** menjadi `train/val/test` (lihat subbagian di bawah).

Setelah langkah di atas, `data/raw/GCD.zip` berisi 7 kelas (hanya `train/` & `test/` —
tanpa `val/`):
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.
Pilih **salah satu** cara penyiapan di bawah.

### Cara ringkas (disarankan): split jadi `data/processed/`

`scripts/split_dataset.py` mengekstrak zip, membagi `train/` menjadi train/val
secara *stratified* & *seeded* (15 %, seed 42), lalu menyalin `test/` apa adanya
→ `data/processed/{train,val,test}`. Inilah layout default yang dibaca `train.py`
(`DATA_ROOT = data/processed` di `config.py`), jadi setelah langkah ini cukup
jalankan `python scripts/train.py` tanpa argumen.

```bash
python scripts/split_dataset.py                            # ekstrak + split (copy; data/raw tetap utuh)
python scripts/split_dataset.py --move --force             # pindahkan train (hemat disk), timpa output
python scripts/split_dataset.py --val-frac 0.15 --seed 42  # atur proporsi val & seed
```

Argumen utama: `--zip` (default `data/raw/GCD.zip`), `--raw-dir`
(`data/raw/GCD`), `--out` (`data/processed`), `--val-frac` (`0.15`), `--seed`
(`42`), `--move` (pindah, bukan copy; `test/` selalu di-copy), `--force` (timpa
`--out`). Skrip mencetak tabel jumlah train/val/test per kelas setelah selesai.

### Cara manual: ekstrak saja, biarkan val di-carve otomatis

```powershell
Expand-Archive data\raw\GCD.zip -DestinationPath data\raw\   # PowerShell
```

Lalu latih langsung dari `data/raw/GCD`. Karena tak ada `val/`, validasi di-carve
otomatis dari `train/` secara stratified (`VAL_SPLIT = 0.15`):

```bash
python scripts/train.py --data-root data/raw/GCD
```

Detail lengkap (angka nyata per kelas, logika split, reproducibility):
**[`data/README.md`](data/README.md)** · **[`docs/DATASET_SPLIT.md`](docs/DATASET_SPLIT.md)**
· **[`TRAINING.md`](TRAINING.md)** §5.

## Training

```bash
python scripts/train.py                                   # baca data/processed (default)
# atau atur argumen
python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud
```

- Checkpoint terbaik → `artifacts/checkpoints/<run>_Versi01.pth`
- Checkpoint terakhir (bisa di-resume) → `..._last.pth`
- Kurva & confusion matrix → `artifacts/results/`

### Pemantauan dengan TensorBoard

TensorBoard **terpasang & aktif otomatis** saat training berjalan — tak perlu
langkah manual. Begitu `train.py` mulai, server dijalankan sendiri dan dasbor
bisa dibuka di:

```
http://localhost:6006
```

- Event file (metrik per epoch: loss, accuracy, F1, learning rate) → `artifacts/runs/<run-name>/`
- Bila paket `tensorboard` belum ada, akan dipasang otomatis via pip (`TB_AUTO_INSTALL`).
- Atur lewat `config.py`: `USE_TENSORBOARD` (on/off), `TB_AUTO_LAUNCH` (tulis log
  tanpa membuka server), `TB_PORT`, `TB_LOG_DIR`.

Membuka dasbor secara manual (mis. untuk melihat run lama):

```bash
tensorboard --logdir artifacts/runs
```

Semua hyperparameter (ukuran gambar, batch, jadwal tahap, augmentasi, flag
arsitektur) ada di **`src/clouddensenet/config.py`**. Set `DEBUG = True` untuk
uji cepat 1 batch/epoch di CPU.

## Catatan

- Notebook asli (`Copy_of_CloudDenseNet.ipynb`) dibiarkan sebagai referensi dan
  **tidak** diubah.
- Folder **`Backup/`** sengaja tidak disentuh.
