# Cara Menjalankan ALGA-DMA-Cloud-Classification (setup lokal)

Panduan menjalankan repo pihak ketiga `Tofu-Pang/ALGA-DMA-Cloud-Classification`
di mesin ini, memakai **venv & dataset dari projek CloudDenseNet** (`D:\citra awan\PM`)
**tanpa merusak** projek tersebut.

> Folder ini (`src/ALGA-DMA-Cloud-Classification/`) sudah masuk `.gitignore` projek,
> jadi repo ini **tidak akan ter-commit** ke repo CloudDenseNet. Semua output training
> (log, `.pth`, `.png`) juga tersimpan di sini dan ikut ter-ignore.

---

## 1. Prasyarat (sudah disiapkan)

- **Python & venv:** memakai venv projek di `D:\citra awan\PM\.venv` (Python 3.11).
- **Dependensi:** semua kebutuhan ALGA sudah ada di venv itu. Hanya 2 paket yang
  ditambahkan khusus untuk ALGA:
  - `opencv-python`
  - `collections-extended`
  (torch/torchvision/numpy dll. memakai yang sudah terpasang — tidak diubah.)

Jika suatu saat venv dibuat ulang, install lagi 2 paket itu:

```powershell
D:\citra awan\PM\.venv\Scripts\python.exe -m pip install opencv-python collections-extended
```

---

## 2. Dataset yang dipakai

Script memuat **langsung** split yang sudah ada di:

```
D:\citra awan\PM\data\processed\dataset_b\
├── train\   (13.299 gambar)  -> dengan augmentasi
├── val\     ( 3.800 gambar)  -> tanpa augmentasi
└── test\    ( 1.901 gambar)  -> tanpa augmentasi
```

`dataset_b` = gabungan GCD train (10k) + test (9k) = **19.000 gambar**, sudah di-split
**70/20/10 SEBELUM augmentasi** (metodologi paper ALGA). 7 kelas:
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.

> Script **tidak** melakukan split ulang dan **tidak** menghapus apa pun di folder data
> (fungsi `remove_hidden_folders` dinonaktifkan). Data hanya dibaca.

---

## 3. Menjalankan

Dari dalam folder ini (agar output nyasar ke sini, bukan ke projek Anda):

```powershell
cd "D:\citra awan\PM\src\ALGA-DMA-Cloud-Classification"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
D:\citra awan\PM\.venv\Scripts\python.exe -u alga_cloud_classification.py
```

Git Bash:

```bash
cd "D:/citra awan/PM/src/ALGA-DMA-Cloud-Classification"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "D:/citra awan/PM/.venv/Scripts/python.exe" -u alga_cloud_classification.py
```

GPU dipakai otomatis bila CUDA tersedia (`cuda:0`), jika tidak jatuh ke CPU.

---

## 4. Output yang dihasilkan (di folder ini)

- `training_log.log` — log training
- `ALGA_DenseNet_model.pth` — bobot model hasil training
- `train_val_metrics.png` — kurva loss/akurasi
- `confusion_matrix.png` — confusion matrix set test
- `ppp.png` — kurva precision-recall

---

## 5. Perubahan vs repo asli (hanya di file skrip, bukan projek Anda)

Edit pada `alga_cloud_classification.py`:

| Bagian | Asli | Sekarang | Alasan |
|---|---|---|---|
| Path dataset | `dataset_path = r'GCD'` (1 folder, split 70/20/10 internal) | `dataset_root = ...\dataset_b`, muat `train/val/test` langsung | Pakai split ALGA yang sudah dibuat, hindari double-split |
| `remove_hidden_folders()` | dipanggil (hapus folder berawalan `.` via `rmtree`) | **dinonaktifkan** | Lindungi data asli projek |
| `batch_size` | 32 / 16 / 32 | **32 / 16 / 32** (tetap) | Sesuai paper; butuh GPU VRAM besar (>= ~16 GB). Di GPU 8 GB OOM → turunkan ke 8 |
| `num_workers` | 4 | **0** | Aman di Windows (hindari masalah multiprocessing) |
| `num_epochs` | 10 | **10** (default) | Training penuh |

> **Catatan GPU:** di GPU 8 GB batch 32 = `CUDA OutOfMemory`, batch 8 jalan tapi ~8 jam/epoch.
> Script ini disetel batch 32 untuk dijalankan di **PC dengan GPU lebih besar**. Di PC kecil,
> ubah ketiga `batch_size` ke 8.
>
> **Catatan path dataset di PC lain:** `dataset_root` memakai path absolut
> `D:\citra awan\PM\data\processed\dataset_b`. Di PC lain, sesuaikan path ini dan pastikan
> folder `dataset_b\{train,val,test}` tersedia di sana.
| `print_class_distribution` | muat tiap gambar untuk hitung | pakai `.targets` | Startup jauh lebih cepat |

---

## 6. Menyesuaikan untuk training sebenarnya

Buka `alga_cloud_classification.py`, bagian `if __name__ == "__main__":`

- **Jumlah epoch** — ubah `num_epochs=1` menjadi lebih besar (mis. `num_epochs=10`
  atau sesuai paper) pada baris:
  ```python
  history = train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=1)
  ```
- **Batch size** — bila ingin lebih besar & GPU cukup, naikkan `batch_size` ketiga
  `DataLoader`. Untuk setia ke paper (batch efektif 32) di GPU 8 GB, pakai batch 8 +
  **gradient accumulation** 4 langkah (perlu sedikit modifikasi loop `train_model`).

---

## 7. Troubleshooting

- **`CUDA out of memory`** — turunkan `batch_size` (mis. 8 → 4) pada ketiga `DataLoader`,
  pastikan `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` diset. Tutup aplikasi lain
  yang memakai GPU.
- **Error multiprocessing DataLoader di Windows** — pastikan `num_workers=0`.
- **`ModuleNotFoundError: cv2` / `collections_extended`** — jalankan ulang perintah pip
  di bagian 1.
- **Jalankan via CPU saja** (lambat) — set `$env:CUDA_VISIBLE_DEVICES = ""` sebelum
  menjalankan.
