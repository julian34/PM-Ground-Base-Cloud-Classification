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

`data/raw/GCD.zip` sudah tersedia. Ekstrak menjadi `data/raw/GCD/{train,test}`:

```powershell
Expand-Archive data\raw\GCD.zip -DestinationPath data\raw\   # PowerShell
```

Dataset ini hanya punya `train/` dan `test/` (tanpa `val/`); validasi otomatis
di-carve dari `train/` secara stratified (`VAL_SPLIT = 0.15`). 7 kelas:
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.
Detail: **[`data/README.md`](data/README.md)**.

## Training

```bash
python scripts/train.py
# atau atur argumen
python scripts/train.py --data-root data/raw/GCD --epochs 80 --run-name cloud
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
