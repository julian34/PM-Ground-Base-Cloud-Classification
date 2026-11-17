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

```text
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

| Argumen                    | Default              | Keterangan                                                            |
| -------------------------- | -------------------- | --------------------------------------------------------------------- |
| `--data-root`              | `data/processed`     | Root dataset ImageFolder (`train/val/test`).                          |
| `--epochs`                 | `80` (dari `PHASES`) | Jumlah epoch total.                                                   |
| `--run-name`               | `cloud`              | Nama run (dipakai untuk checkpoint & folder TensorBoard).             |
| `--resume` / `--no-resume` | resume **aktif**     | Lanjut dari `_last.pth` bila ada; `--no-resume` untuk mulai dari nol. |

Saat mulai, script mencetak device, daftar kelas, jadwal tahap (`PHASES`), dan
jumlah parameter trainable per tahap.

## 7. Pemantauan dengan TensorBoard (otomatis)

TensorBoard **aktif sendiri** begitu training berjalan — tak perlu langkah manual.
Buka dasbor di:

```text
http://localhost:6006
```

- Metrik per epoch: `Loss/train`, `Loss/val`, `Acc/train`, `Acc/val`, `F1/val`, `LR`.
- Event file tersimpan di `artifacts/runs/<run-name>/`.
- Membuka dasbor manual (mis. melihat run lama):

  ```bash
  tensorboard --logdir artifacts/runs
  ```

Pengaturan di `config.py`:

| Flag              | Default          | Fungsi                                                            |
| ----------------- | ---------------- | ----------------------------------------------------------------- |
| `USE_TENSORBOARD` | `True`           | Aktif/nonaktifkan penulisan log.                                  |
| `TB_AUTO_INSTALL` | `True`           | Pasang `tensorboard` via pip bila belum ada.                      |
| `TB_AUTO_LAUNCH`  | `True`           | Jalankan server otomatis; `False` = tetap tulis log tanpa server. |
| `TB_PORT`         | `6006`           | Port server.                                                      |
| `TB_LOG_DIR`      | `artifacts/runs` | Lokasi event file (subfolder per run-name).                       |

### 7.1. TensorBoard masih menampilkan data lama walau file sudah dihapus

Ini **bukan bug** — gabungan dua perilaku bawaan TensorBoard + desain `tblog.py`.
Menghapus *sebagian* file saja biasanya tidak cukup.

**Sebab 1 — file event menumpuk di folder run yang sama.** TensorBoard
**menggabungkan (merge) SEMUA** file `events.out.tfevents.*` dalam satu folder jadi
satu grafik. Tiap training ulang dengan `--run-name` yang sama membuat file event
**baru** (tidak menimpa yang lama), jadi kurva run-run lama ikut tertimpa:

```text
artifacts/runs/cloud/
├── events.out.tfevents.…32508.0   ← run 1
├── events.out.tfevents.…5352.0    ← run 2  (semua di-merge jadi satu kurva)
└── events.out.tfevents.…8092.0    ← run 3
```

**Sebab 2 — server TensorBoard sengaja dibiarkan hidup & nge-cache di RAM.**
`tblog.py` tidak mematikan server saat training selesai (agar dasbor tetap bisa
dibuka). Akibatnya: (a) server yang masih jalan menyimpan data lama di memori, jadi
**menghapus file di disk ≠ hilang dari server**; (b) saat training lagi, server
melihat port 6006 sudah dipakai lalu **memakai ulang server lama** (bukan start
fresh). Server dari sesi sebelumnya berjalan *detached*, jadi `hentikan_server()`
(yang hanya mematikan server proses Python yang sama) tidak menyentuhnya — harus
dimatikan manual.

**Membersihkan dengan benar — urutannya penting:**

1. **Matikan dulu** server TensorBoard yang masih hidup (langkah yang sering dilewati):

   ```powershell
   # PowerShell — cari & matikan pemilik port 6006
   Get-NetTCPConnection -LocalPort 6006 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   # alternatif kasar:
   taskkill /IM tensorboard.exe /F 2>$null
   ```

2. **Baru hapus** log lama (setelah server mati, file tidak lagi dipegang):

   ```powershell
   Remove-Item -Recurse -Force "artifacts\runs\*"
   ```

3. **Jalankan training lagi** → server fresh hanya membaca file yang ada sekarang.
   Di browser tekan **Ctrl+Shift+R** (hard refresh) atau klik tombol reload di
   pojok kanan atas TensorBoard.

**Cara menghindari penumpukan:** pakai `--run-name` unik tiap percobaan supaya tiap
run punya folder sendiri (bisa dipilih/dimatikan per-run di sidebar TensorBoard):

```powershell
python scripts/train.py --run-name "cloud_$(Get-Date -Format yyyyMMdd_HHmm)"
```

## 8. Keluaran training

| Berkas                                    | Lokasi                                         |
| ----------------------------------------- | ---------------------------------------------- |
| Checkpoint **terbaik** (F1 val tertinggi) | `artifacts/checkpoints/<run>_Versi01.pth`      |
| Checkpoint **terakhir** (untuk resume)    | `artifacts/checkpoints/<run>_Versi01_last.pth` |
| Kurva training (PNG)                      | `artifacts/results/<run>_kurva.png`            |
| Confusion matrix val (PNG)                | `artifacts/results/<run>_cm_val.png`           |
| Log TensorBoard                           | `artifacts/runs/<run>/`                        |

## 9. Evaluasi dari checkpoint (tanpa training ulang)

Untuk membuat confusion matrix dari checkpoint yang **sudah ada** — tanpa melatih
ulang — pakai `scripts/evaluate.py`. Script memuat bobot, menjalankan inferensi
pada **test set held-out** (`data/processed/test`), lalu menyimpan confusion
matrix (panel **Jumlah** + **Ternormalisasi/recall**) beserta Akurasi & F1 macro.

```bash
# checkpoint default (artifacts/checkpoints/cloud_Versi01.pth)
python scripts/evaluate.py

# checkpoint tertentu (mis. salah satu di folder list/)
python scripts/evaluate.py --checkpoint "artifacts/checkpoints/list/cloud_Versi01_51_09246.pth" --run-name cloud_51
```

| Argumen            | Default                                   | Keterangan                                                                                        |
| ------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `--checkpoint`     | `artifacts/checkpoints/cloud_Versi01.pth` | Path file `.pth` yang dievaluasi.                                                                 |
| `--data-root`      | `data/processed`                          | Root dataset ImageFolder (dipakai folder `test/`).                                                |
| `--run-name`       | `cloud`                                   | Prefiks nama berkas keluaran PNG.                                                                 |
| `--test-per-class` | `None`                                    | Selain test penuh, evaluasi juga subset stratified N citra/kelas (mis. `300`) sebagai pembanding. |
| `--show`           | nonaktif                                  | Tampilkan jendela plot; default hanya menyimpan PNG (`Agg`).                                      |

Keluaran:

| Berkas                                            | Lokasi                                   |
| ------------------------------------------------- | ---------------------------------------- |
| Confusion matrix test penuh (PNG)                 | `artifacts/results/<run>_cm_test.png`    |
| Confusion matrix subset (bila `--test-per-class`) | `artifacts/results/<run>_cm_test<N>.png` |

> **Varian Top Block berbeda.** Script membangun head sesuai bentuk tensor di
> checkpoint (lewat `bangun_dari_checkpoint`), jadi checkpoint dari varian head
> lama tetap bisa dimuat tanpa error `size mismatch`. Bila bobot benar-benar tak
> cocok dengan arsitektur, muncul `ValueError` dengan detail kunci yang hilang —
> bukan prediksi diam-diam dari bobot setengah-acak.

## 10. Melanjutkan (resume) training

Resume otomatis aktif: cukup jalankan ulang perintah yang sama, script membaca
`_last.pth` dan melanjutkan dari epoch berikutnya.

```bash
python scripts/train.py --run-name cloud              # lanjut otomatis
python scripts/train.py --run-name cloud --no-resume  # paksa mulai dari nol
```

## 11. Mode DEBUG (uji cepat di CPU)

Set `DEBUG = True` di `config.py` untuk melatih hanya 1–3 batch/epoch di CPU —
berguna memverifikasi pipeline tanpa menunggu training penuh. Kembalikan ke
`False` untuk training sungguhan.

## 12. Melihat jumlah parameter

Jalankan dari root proyek (venv aktif).

**Cara tercepat** — smoke test bawaan mencetak total parameter di baris akhir:

```bash
python tests/test_forward.py
# ... [OK] forward pass sukses | total parameter: 8,519,809
```

**Total vs trainable** (angka *trainable* berubah tiap tahap pembekuan; panggil
setelah `apply_phase` untuk melihat jumlah per tahap):

```python
import sys; sys.path.insert(0, "src")
from clouddensenet import CloudDenseNet

m = CloudDenseNet(use_vit=True)        # False = tanpa cabang ViT
total     = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Total     : {total:,}")
print(f"Trainable : {trainable:,}")
```

**Rincian per submodul:**

```python
import sys; sys.path.insert(0, "src")
from clouddensenet import CloudDenseNet

m = CloudDenseNet(use_vit=True)
for sub in ["conv", "block1", "block2", "block3", "block4", "top", "vit"]:
    mod = getattr(m, sub, None)
    if mod is not None:
        print(f"{sub:8s}: {sum(p.numel() for p in mod.parameters()):,}")
```

Hasil (per 2026-11-13):

| Submodul        | Tanpa ViT | Dengan ViT |
| --------------- | --------: | ---------: |
| `conv` (stem)   |     9.408 |      9.408 |
| `block1`        |   403.590 |    403.590 |
| `block2`        | 1.056.780 |  1.056.780 |
| `block3`        | 3.111.960 |  3.111.960 |
| `block4`        | 2.340.880 |  2.340.880 |
| `top`           |   596.615 |    694.919 |
| `vit`           |         — |  1.984.896 |
| **Total**       | **8.519.809** | **10.603.009** |

> `top` membesar saat ViT aktif karena Linear pertamanya melebar untuk menampung
> vektor cabang ViT (fusi concat). Total bertambah ±2,08 jt (+25%).

## 13. Troubleshooting

| Gejala                                                     | Kemungkinan sebab & solusi                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `Dataset tidak ditemukan di ...`                           | Jalankan langkah **5** (`split_dataset.py`) lebih dulu.                                                |
| `Checkpoint tidak ditemukan: ...` (evaluasi)               | Periksa path `--checkpoint`; lihat daftar `.pth` di `artifacts/checkpoints/`.                          |
| `ValueError: ... tidak cocok dengan arsitektur` (evaluasi) | Checkpoint dari arsitektur lain; bukan sekadar varian Top Block. Pakai checkpoint yang sesuai.         |
| Device tercetak `cpu` padahal ada GPU                      | `DEBUG=True` memaksa CPU; atau build PyTorch CPU-only — reinstall dengan index URL CUDA.               |
| `[!] batch N: loss NaN dilewati`                           | Normal sesekali (pengaman NaN). Bila >20 batch, training berhenti — periksa modul sumber yang dicetak. |
| Port 6006 sudah dipakai                                    | Ubah `TB_PORT` di `config.py`, atau tutup proses TensorBoard lama.                                     |
| TensorBoard tak mau jalan                                  | Set `TB_AUTO_LAUNCH=False` dan buka manual: `tensorboard --logdir artifacts/runs`.                     |
| TensorBoard masih tampilkan data run lama                  | File event menumpuk + server lama nge-cache. Matikan server dulu, baru hapus `artifacts/runs/` — lihat **§7.1**. |
| Kurva beberapa run tertumpuk jadi satu                     | Nama run (`--run-name`) dipakai ulang; file event di-merge. Pakai run-name unik tiap percobaan (**§7.1**). |

---

Detail arsitektur & konvensi repo: lihat [`README.md`](README.md) dan
[`CLAUDE.md`](CLAUDE.md).
