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
| **Top Block** | GAP → FC(1024→512→128→32→16) + BN/ReLU tiap lapis → Dropout(0.8) → FC(16→7). |
| **Cabang CloudViT** | Cabang Transformer paralel (default aktif: `USE_VIT=True`, `VIT_ARCH="cloudvit"`); vektornya (256-dim) difusikan ke Top Block. Lihat [`docs/CABANG_VIT_DETAIL.md`](docs/CABANG_VIT_DETAIL.md). |
| **Focal Loss** | Menangani ketidakseimbangan kelas (γ = 2.0). |
| **Fine-tuning bertahap** | Bobot ImageNet dibekukan lalu dibuka bertahap lewat 7 fase (lihat `PHASES`). |

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
│   ├── vit_branch.py         # ViTBranch sederhana (VIT_ARCH="simple")
│   ├── cloudvit_branch.py    # CloudViTBranch (default, VIT_ARCH="cloudvit")
│   ├── data.py               # transform, DataLoader, persiapan batch
│   ├── engine.py             # loop train/eval, jadwal tahap, pengaman NaN
│   ├── tblog.py              # TensorBoard: pasang & aktif otomatis saat training
│   └── viz.py                # kurva training & confusion matrix
├── scripts/train.py          # entrypoint training
├── scripts/visualisasi_augmentasi.py  # gambar augmentasi -> laporan/gambar/agumentasi/
├── scripts/gradcam_perkelas.py        # Grad-CAM per kelas -> laporan/gambar/grad-map/
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
| **[`docs/Panduan_Pull.md`](docs/Panduan_Pull.md)** | Cara menarik pembaruan terbaru (git pull), stash, dan atasi konflik. |
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

### Penelitian B: dataset berbeda (split ulang terpisah)

Untuk penelitian dengan **dataset lain** tanpa mengubah dataset GCD asli, gunakan
`scripts/split_dataset_experiment.py`. Berbeda dari `split_dataset.py` (yang
meng-carve val dari train yang sudah ter-split), script ini:

1. **ekstrak** arsip dataset baru (struktur per-kelas),
2. **gabung** semua gambar jadi satu pool per-kelas (`_merged/<kelas>/`) — struktur
   bersarang apa pun (mis. `train/` + `test/`) dikelompokkan berdasarkan nama kelas,
3. **split ulang 3-arah** (train/val/test) *stratified* & *seeded*.

Hasilnya masuk ke **sub-folder terpisah** `data/processed/dataset_b/{train,val,test}`,
jadi `data/processed/{train,val,test}` asli **tidak tersentuh**. Script menolak
berjalan bila output bukan sub-folder di dalam `data/processed` (dataset asli aman).

```bash
# 1. letakkan arsip dataset baru di:
#    data/raw/dataset_b.zip
# 2. (opsional) atur rasio split di config.py -> DATASET_B_SPLIT = (train, val, test)
# 3. jalankan:
python scripts/split_dataset_experiment.py                 # ekstrak + gabung + split (copy)
python scripts/split_dataset_experiment.py --move          # pindah file (hemat disk)
python scripts/split_dataset_experiment.py --keep-merged   # simpan pool _merged
python scripts/split_dataset_experiment.py --force         # timpa data/processed/dataset_b
# 4. latih di dataset baru (tanpa ubah DATA_ROOT global):
python scripts/train.py --data-root data/processed/dataset_b
```

Konfigurasi di `config.py` (blok `DATASET_B_*`): `DATASET_B_NAME` (nama sub-folder),
`DATASET_B_ZIP` (arsip sumber), `DATASET_B_ROOT` (output), `DATASET_B_SPLIT`
(rasio train/val/test, harus = 1.0), `DATASET_B_SEED` (reproducible, default `SEED`).
Argumen CLI: `--zip`, `--out`, `--split TR VA TE`, `--seed`, `--move`,
`--keep-merged`, `--force`.

## Training

```bash
python scripts/train.py                                   # baca data/processed (default)
# atau atur argumen
python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud
```

- Checkpoint terbaik → `artifacts/checkpoints/<run>_Versi01.pth`
- Checkpoint terakhir (bisa di-resume) → `..._last.pth`
- Kurva & confusion matrix → `artifacts/results/`

Beberapa seed sekaligus (berurutan; run-name otomatis `<run>_s<seed>`, argumen lain
diteruskan ke `train.py`), lalu ringkasan mean ± std ke
`artifacts/results/<run>_seeds_ringkasan.csv`:

```bash
python scripts/run_seeds.py --seeds 42 123 2024 --run-name cloud --epochs 80
python scripts/run_seeds.py --seeds 42 123 2024 --run-name cloud --hanya-ringkas  # ringkas ulang saja
```

### Cabang ViT: CloudViT dari awal vs MobileViTv2 pretrained

Secara default cabang ViT memakai **CloudViT yang dilatih dari awal**. Dengan flag
`--vit-pretrained`, cabang itu diganti **MobileViTv2 berbobot ImageNet** (lewat `timm`,
model `config.MOBILEVIT_NAME = "mobilevitv2_100.cvnets_in1k"`, keluaran 512-dim). Model
ini dibekukan mengikuti jadwal fine-tuning bertahap `PHASES`.

```bash
python scripts/train.py --run-name fusi_cloudvit                                    # default: CloudViT dari awal
python scripts/train.py --vit-pretrained --run-name fusi_mvit2                      # MobileViTv2, ikut jadwal
python scripts/train.py --vit-pretrained --vit-ft beku --run-name fusi_mvit2_beku   # MobileViTv2, beku total
```

| Fase `PHASES` | `--vit-ft jadwal` (default) | `--vit-ft beku` |
|---|---|---|
| frozen (1–2) | seluruh `vit.*` beku | beku |
| partial (3–5) | stage terakhir dibuka (`VIT_PRETRAINED_BLOK_FT`) | beku |
| full (6–7) | seluruh cabang dibuka | beku |

Catatan:
- Butuh paket `timm` (sudah ada di `requirements.txt`). Run pertama perlu internet untuk
  mengunduh bobot. Setelah itu bobot diambil dari cache Hugging Face.
- Normalisasi ditangani di dalam cabang: bobot MobileViTv2 memakai input [0,1] tanpa
  mean/std, sehingga normalisasi ImageNet dari pipeline data dibalik otomatis.
- BatchNorm MobileViTv2 yang bobotnya beku tetap dalam mode eval (running stats tidak berubah).
- Selalu pakai `--run-name` berbeda. Checkpoint CloudViT dan MobileViTv2 tidak bisa
  saling di-resume. `evaluate.py`/`report.py` mengenali checkpoint MobileViTv2 secara otomatis.
- Bawaan permanen bisa diatur di `config.py`: `VIT_PRETRAINED`, `MOBILEVIT_NAME`,
  `VIT_PRETRAINED_FT`, `VIT_PRETRAINED_BLOK_FT`.

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

## Gambar untuk laporan: augmentasi & Grad-CAM per kelas

Dua skrip mandiri yang menghasilkan gambar siap pakai di laporan. Keduanya hanya
**memakai** modul yang ada (tidak mengubah kode training) dan memakai model yang
**sudah dilatih**.

### Visualisasi augmentasi

```bash
python scripts/visualisasi_augmentasi.py                        # default: 1 citra/kelas dari train
python scripts/visualisasi_augmentasi.py --n-variasi 6 --dpi 300
python scripts/visualisasi_augmentasi.py --data-root data/processed/dataset_b
```

Pipeline diambil langsung dari `build_transforms()` (`data.py`), jadi selalu
sama dengan augmentasi saat training (ikut berubah bila `STRONG_AUG` diubah).
Pada gambar per tahap, flip ditampilkan dalam keadaan aktif (p=1) agar efeknya
terlihat; saat training tetap p=0.5.

Keluaran → `laporan/gambar/agumentasi/`:

| File | Isi |
|---|---|
| `<kelas>/00_asli.png`, `01_affine.png`, … | tiap tahap augmentasi, terpisah, tanpa teks |
| `<kelas>/pipeline_01..N.png` | hasil pipeline lengkap (acak, seeded) |
| `<kelas>_tahapan.png` | grid tahapan satu kelas |
| `ringkasan_tahapan.png`, `ringkasan_pipeline.png` | gabungan 7 kelas |

### Grad-CAM per kelas (prediksi benar & salah + confidence)

```bash
python scripts/gradcam_perkelas.py                              # 4 benar + 4 salah per kelas (test)
python scripts/gradcam_perkelas.py --k 6 --pilih yakin          # confidence tertinggi
python scripts/gradcam_perkelas.py --pilih ragu                 # confidence terendah
python scripts/gradcam_perkelas.py --polos                      # + overlay tanpa teks
python scripts/gradcam_perkelas.py --checkpoint artifacts/checkpoints/cloud_Versi01.pth --split val
```

Opsi penting: `--checkpoint` (default: `.pth` terbaru di `artifacts/checkpoints/`,
selain `*_last.pth`), `--split test|val|train`, `--k` (jumlah sampel benar & salah
per kelas), `--pilih acak|yakin|ragu` (default `acak`, seed 42), `--alpha`
(transparansi heatmap), `--dpi`.

Keluaran → `laporan/gambar/grad-map/`:

| File | Isi |
|---|---|
| `<kelas>/benar/NN_<file>.png` | citra asli · Grad-CAM kelas prediksi · batang probabilitas 7 kelas |
| `<kelas>/salah/NN_<file>.png` | sama + Grad-CAM **label asli** sebagai pembanding |
| `<kelas>/<kelas>_benar.png`, `<kelas>_salah.png` | grid semua sampel kelas tsb |
| `<kelas>/polos/` | (dengan `--polos`) citra & overlay tanpa teks |
| `ringkasan_benar.png`, `ringkasan_salah.png` | 1 contoh per kelas |
| `prediksi_<split>.csv` | prediksi + confidence + probabilitas tiap kelas untuk semua citra |
| `ringkasan_perkelas.csv` | n, benar, salah, akurasi, rata-rata confidence per kelas |

Warna judul: hijau = BENAR, merah = SALAH; batang biru = label asli.
Grad-CAM diambil dari layer `norm5` (peta fitur 7×7 terakhir backbone DenseNet),
sehingga hanya mencerminkan jalur CNN, bukan cabang CloudViT paralel.

## Catatan

- Notebook asli (`Copy_of_CloudDenseNet.ipynb`) dibiarkan sebagai referensi dan
  **tidak** diubah.
- Folder **`Backup/`** sengaja tidak disentuh.
