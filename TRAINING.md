# Panduan Menjalankan Training — CloudDenseNet

Runbook langkah-demi-langkah untuk melatih **CloudDenseNet** (klasifikasi 7 kelas
awan) di mesin lokal. Semua hyperparameter ada di
[`src/clouddensenet/config.py`](src/clouddensenet/config.py) — ubah di sana, bukan
hard-code di script.

> Platform utama: **Windows 11 + PowerShell**. Perintah Git Bash juga disertakan.

---

## 1. Prasyarat

- Python ≥ 3.10 (venv sudah tersedia di `.venv/`, cpython-3.11).
- GPU NVIDIA + CUDA 12.4 **disarankan** (fallback CPU otomatis bila tak ada GPU).
- Dataset `data/raw/GCD.zip` sudah tersedia di repo.

## 2. Aktifkan virtual environment

```powershell
# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
```
```bash
# Git Bash
source .venv/Scripts/activate
```

## 3. Instalasi dependensi (sekali saja)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .            # pasang paket clouddensenet (disarankan)
```

> Mesin tanpa GPU NVIDIA: ganti index URL menjadi
> `https://download.pytorch.org/whl/cpu`.

`tensorboard` ikut terpasang dari `requirements.txt`. Kalaupun belum ada, ia akan
**dipasang otomatis** saat training mulai (lihat `TB_AUTO_INSTALL` di `config.py`).

## 4. Uji cepat (opsional, tanpa dataset/GPU)

Pastikan model bisa dibangun & forward pass jalan sebelum training penuh:

```bash
pytest
# atau
python tests/test_forward.py
```

Harus mencetak `forward pass sukses` + jumlah parameter.

## 5. Siapkan dataset

Ekstrak `GCD.zip` lalu split menjadi `data/processed/{train,val,test}`
(stratified, seed 42, 15% validasi):

```bash
python scripts/split_dataset.py                  # salin (data/raw tetap utuh)
python scripts/split_dataset.py --move --force   # pindahkan train & timpa output
```

Argumen lain: `--zip`, `--raw-dir`, `--out`, `--val-frac` (default 0.15),
`--seed` (default 42).

Hasil akhir berbentuk ImageFolder:

```
data/processed/
├── train/<kelas>/*.jpg
├── val/<kelas>/*.jpg
└── test/<kelas>/*.jpg
```

7 kelas (indeks 0–6):
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.

## 6. Jalankan training

```bash
python scripts/train.py
```

Dengan argumen kustom:

```bash
python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud
```

| Argumen | Default | Keterangan |
|---|---|---|
| `--data-root` | `data/processed` | Root dataset ImageFolder (`train/val/test`). |
| `--epochs` | `80` (dari `PHASES`) | Jumlah epoch total. |
| `--run-name` | `cloud` | Nama run (dipakai untuk checkpoint & folder TensorBoard). |
| `--resume` / `--no-resume` | resume **aktif** | Lanjut dari `_last.pth` bila ada; `--no-resume` untuk mulai dari nol. |

Saat mulai, script mencetak device, daftar kelas, jadwal tahap (`PHASES`), dan
jumlah parameter trainable per tahap.

## 7. Pemantauan dengan TensorBoard (otomatis)

TensorBoard **aktif sendiri** begitu training berjalan — tak perlu langkah manual.
Buka dasbor di:

```
http://localhost:6006
```

- Metrik per epoch: `Loss/train`, `Loss/val`, `Acc/train`, `Acc/val`, `F1/val`, `LR`.
- Event file tersimpan di `artifacts/runs/<run-name>/`.
- Membuka dasbor manual (mis. melihat run lama):

  ```bash
  tensorboard --logdir artifacts/runs
  ```

Pengaturan di `config.py`:

| Flag | Default | Fungsi |
|---|---|---|
| `USE_TENSORBOARD` | `True` | Aktif/nonaktifkan penulisan log. |
| `TB_AUTO_INSTALL` | `True` | Pasang `tensorboard` via pip bila belum ada. |
| `TB_AUTO_LAUNCH` | `True` | Jalankan server otomatis; `False` = tetap tulis log tanpa server. |
| `TB_PORT` | `6006` | Port server. |
| `TB_LOG_DIR` | `artifacts/runs` | Lokasi event file (subfolder per run-name). |

## 8. Keluaran training

| Berkas | Lokasi |
|---|---|
| Checkpoint **terbaik** (F1 val tertinggi) | `artifacts/checkpoints/<run>_Versi01.pth` |
| Checkpoint **terakhir** (untuk resume) | `artifacts/checkpoints/<run>_Versi01_last.pth` |
| Kurva training (PNG) | `artifacts/results/<run>_kurva.png` |
| Confusion matrix val (PNG) | `artifacts/results/<run>_cm_val.png` |
| Log TensorBoard | `artifacts/runs/<run>/` |

## 9. Melanjutkan (resume) training

Resume otomatis aktif: cukup jalankan ulang perintah yang sama, script membaca
`_last.pth` dan melanjutkan dari epoch berikutnya.

```bash
python scripts/train.py --run-name cloud              # lanjut otomatis
python scripts/train.py --run-name cloud --no-resume  # paksa mulai dari nol
```

## 10. Mode DEBUG (uji cepat di CPU)

Set `DEBUG = True` di `config.py` untuk melatih hanya 1–3 batch/epoch di CPU —
berguna memverifikasi pipeline tanpa menunggu training penuh. Kembalikan ke
`False` untuk training sungguhan.

## 11. Troubleshooting

| Gejala | Kemungkinan sebab & solusi |
|---|---|
| `Dataset tidak ditemukan di ...` | Jalankan langkah **5** (`split_dataset.py`) lebih dulu. |
| Device tercetak `cpu` padahal ada GPU | `DEBUG=True` memaksa CPU; atau build PyTorch CPU-only — reinstall dengan index URL CUDA. |
| `[!] batch N: loss NaN dilewati` | Normal sesekali (pengaman NaN). Bila >20 batch, training berhenti — periksa modul sumber yang dicetak. |
| Port 6006 sudah dipakai | Ubah `TB_PORT` di `config.py`, atau tutup proses TensorBoard lama. |
| TensorBoard tak mau jalan | Set `TB_AUTO_LAUNCH=False` dan buka manual: `tensorboard --logdir artifacts/runs`. |

---

Detail arsitektur & konvensi repo: lihat [`README.md`](README.md) dan
[`CLAUDE.md`](CLAUDE.md).
