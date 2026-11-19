# Hasil penelitian datasetB — sumber acuan dokumentasi

Revisi 19 November 2026. Satu run yang dilaporkan: **fusi DenseNet lokal dan CloudViT**,
tujuh kelas citra langit. Folder acuan:
`artifacts/results/datasetB_laporan_19-11-2026_0251`.

## 1. Asal bukti dan urutan penggunaan

| Bukti | Informasi yang didukung |
|---|---|
| [Laporan metrik](../artifacts/results/datasetB_laporan_19-11-2026_0251/datasetB_metrik.md) | Evaluasi ulang, konfigurasi yang dicatat, dan metadata laporan |
| [CSV validasi](../artifacts/results/datasetB_laporan_19-11-2026_0251/datasetB_metrik_val.csv) / [CSV uji](../artifacts/results/datasetB_laporan_19-11-2026_0251/datasetB_metrik_test.csv) | Metrik per kelas dan agregat |
| [CSV pelatihan](../artifacts/results/datasetB_laporan_19-11-2026_0251/dataset-repartitioned_log.csv) | Epoch, tahap, LR, loss, akurasi, F1 validasi, timestamp dan durasi |
| PNG dalam folder hasil | Matriks kebingungan, kurva dan contoh Grad-CAM |
| Checkpoint `artifacts/checkpoints/dataset-repartitioned_Versi01.pth` | Epoch 70, skor seleksi dan struktur bobot yang diperiksa saat revisi jurnal |
| `data/processed/dataset_b` | Hitungan split lokal; bukan manifest historis yang terikat checkpoint |
| Kode di `src/clouddensenet` dan `scripts` | Penjelasan implementasi saat revisi; bukan rekaman semua pengaturan saat training |
| Konfirmasi penulis dan dua gambar perangkat pada percakapan 19 November 2026 | Koreksi GPU/RAM untuk pelatihan dan evaluasi |

Laporan dibuat otomatis pada **19 November 2026 02:54:53**. Nilai konfigurasi yang
dicetak post-hoc diambil dari konfigurasi aktif saat pelaporan. Checkpoint tidak
menyimpan snapshot konfigurasi historis lengkap. Karena itu, hasil numerik,
konfigurasi yang dicatat, dan penjelasan kode mempunyai status bukti berbeda.

## 2. Evaluasi ulang checkpoint terpilih

| Metrik | Validasi | Uji |
|---|---:|---:|
| Jumlah citra | 3.800 | 1.901 |
| Akurasi (%) | 88,03 | 86,74 |
| Precision macro | 0,8896 | 0,8720 |
| Recall macro | 0,8903 | 0,8795 |
| F1-macro | 0,8895 | 0,8747 |
| Precision weighted | 0,8784 | 0,8652 |
| Recall weighted | 0,8803 | 0,8674 |
| F1-weighted | 0,8787 | 0,8651 |

### Hasil uji per kelas

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 1_cumulus | 0,9799 | 0,9605 | 0,9701 | 152 |
| 2_altocumulus | 0,8820 | 0,9595 | 0,9191 | 148 |
| 3_cirrus | 0,8964 | 0,9058 | 0,9010 | 191 |
| 4_clearsky | 0,9787 | 0,9840 | 0,9813 | 374 |
| 5_stratocumulus | 0,7556 | 0,6456 | 0,6963 | 364 |
| 6_cumulonimbus | 0,8295 | 0,8785 | 0,8533 | 576 |
| 7_mixed | 0,7822 | 0,8229 | 0,8020 | 96 |

Pada matriks uji terdapat **1.649 prediksi benar dan 252 salah**. Sebanyak 94 citra
stratocumulus diprediksi sebagai cumulonimbus, sedangkan arah sebaliknya 60 citra.
Pasangan tersebut menyumbang **154/252 = 61,11%** kesalahan uji. Clear sky memiliki
F1 tertinggi (0,9813), stratocumulus terendah (0,6963).

![Matriks kebingungan uji](../artifacts/results/datasetB_laporan_19-11-2026_0251/datasetB_cm_test.png)

## 3. Log pelatihan 80 epoch

CSV memuat tepat epoch 1–80 tanpa duplikasi atau nomor terlewat. Nilai berikut
adalah metrik selama pelatihan, bukan hasil evaluasi ulang pada bagian 2.

| Epoch | Loss latih | Loss validasi | Akurasi validasi (%) | F1-macro validasi |
|---|---:|---:|---:|---:|
| 1 | 1,169144 | 0,572717 | 73,0263 | 0,723063 |
| 70 | 0,568848 | 0,178635 | 88,0526 | 0,890024 |
| 80 | 0,564160 | 0,184073 | 86,9211 | 0,881622 |

F1 validasi maksimum ada pada **epoch 70**. Checkpoint mencatat
`best_f1 = 0.8900238108451747`, sesuai pembulatan log menjadi 0,890024.
Nilai ini tidak diganti dengan F1 validasi evaluasi ulang **0,8895**. Selisih tersebut
tetap dicatat karena penyebabnya belum dipastikan.

| Tahap | Epoch | LR pada CSV | Mode menurut kode terdokumentasi |
|---|---|---|---|
| 1 | 1–10 | 0.001 | frozen |
| 2 | 11–20 | 0.0001 | frozen |
| 3 | 21–30 | 0.00001 | partial |
| 4 | 31–40 | 0.00005 | partial |
| 5 | 41–45 | 0.00001 | partial |
| 6 | 46–60 | 0.00001 | full |
| 7 | 61–80 | 0.000001 | full |

Nomor tahap, rentang epoch dan LR terkonfirmasi CSV. Nama mode dan perilaku
pembekuan dijelaskan dari kode; CSV tidak merekam daftar parameter trainable.
LR meningkat pada tahap 4, sehingga jadwal tidak menurun secara monoton.

### Durasi

- Jumlah kolom `detik`: **58.354,1 detik = 16 jam 12 menit 34,1 detik**.
- Rerata: **729,43 detik/epoch**; minimum 673,6 dan maksimum 859,8 detik.
- Timestamp akhir epoch 1: **17 November 2026 16:30:17**.
- Timestamp akhir epoch 80: **18 November 2026 22:26:34**.
- Selisih kedua timestamp: **29 jam 56 menit 17 detik**. Ini bukan durasi sejak
  awal pelatihan, karena timestamp pertama sudah berada di akhir epoch 1.
- Antara akhir epoch 21 dan 22 terdapat interval 14 jam 7 menit 54 detik.
  Setelah durasi epoch 22 dikurangi, tersisa **13 jam 56 menit 19,5 detik** di luar
  durasi epoch. Penyebab jeda dan status resume tidak dapat disimpulkan dari CSV.

Menurut batas timer pada kode, `detik` mencakup proses latih dan validasi per epoch,
tetapi tidak mencakup penyimpanan checkpoint sesudah timer dan jeda antarepoch.
Angka tersebut bukan waktu GPU murni atau latensi inferensi. Timestamp tanpa zona waktu.
Kolom `resume: Tidak` pada laporan post-hoc tidak dipakai sebagai bukti pasti
bahwa tidak pernah ada penghentian atau kelanjutan proses.

## 4. Konfigurasi yang dicatat pada laporan

| Parameter | Nilai |
|---|---|
| Masukan | RGB 224×224; normalisasi ImageNet |
| Seed | 42 (konfigurasi terdokumentasi) |
| Batch size | 32 |
| Optimizer | AdamW; weight decay 0,01 |
| Loss | Focal Loss; gamma 2; alpha 1 menurut implementasi |
| Gradient clipping | 5 |
| Penanganan ketidakseimbangan | Sampler |
| SE / SCConv / atensi B/R / ViT | Aktif menurut laporan |
| Arsitektur cabang global | CloudViT |
| Kepala klasifikasi | 1280 → 512 → 128 → 32 → 16 → 7 (struktur checkpoint) |
| Dropout kepala | 0,8 menurut laporan; tidak tersimpan sebagai bobot |
| Early stopping | Tidak aktif menurut laporan |

Pada pemeriksaan revisi, `config.py` sudah berisi `IMG_SIZE=512`, sedangkan laporan
run mencatat **224×224**. Dokumentasi run mempertahankan 224×224. Pengaturan kode
saat ini tidak boleh langsung digunakan untuk mengklaim reproduksi run ini.
Parameter tambahan seperti augmentasi, 21.000 penarikan sampler per epoch dan
opsi MixUp dijelaskan sebagai konfigurasi/implementasi terdokumentasi, bukan hasil
eksperimen pembanding.

## 5. Perangkat: koreksi penulis dan konflik metadata

Penulis mengonfirmasi bahwa perangkat pada gambar dipakai untuk **pelatihan dan
evaluasi** penelitian ini.

| Komponen | Keterangan terkonfirmasi dari penulis/gambar |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER |
| Memori GPU khusus | 8.006 MB pada gambar, sekitar 8 GB VRAM |
| RAM | Sekitar 32 GB; Task Manager menunjukkan 31,7 GB yang dapat digunakan |
| RAM tambahan pada gambar | 2400 MT/s, 2 dari 4 slot, DIMM, hardware reserved 311 MB |
| CPU, OS, versi driver dan perangkat lunak mesin ini | Belum dikonfirmasi oleh gambar atau pernyataan penulis |

`Approx. Total Memory 24.364 MB` pada gambar adalah gabungan memori display dan
shared memory; **bukan 24 GB VRAM khusus**. Shared memory tercantum 16.358 MB.
Angka pemakaian RAM saat gambar diambil bukan pengukuran puncak memori training.

Laporan otomatis mencantumkan **RTX 4060 Laptop GPU, RAM 15,7 GB, Windows 10,
CPU Intel 16 core logis, Python 3.11.9, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, CUDA 12.4, cuDNN 90100**. Ini bertentangan dengan koreksi
perangkat oleh penulis. Metadata tersebut dipertahankan pada artefak asli untuk
ketertelusuran, tetapi tidak diatribusikan sebagai spesifikasi mesin penelitian
yang telah terkonfirmasi. Asal perbedaan belum ditetapkan. Versi perangkat lunak
tersebut juga tidak dipindahkan otomatis ke mesin RTX 2060 SUPER.

## 6. Batas kesimpulan

Hasil menunjukkan performa satu checkpoint pada satu repartisi datasetB.
Belum tersedia baseline model tunggal, ablasi, pengulangan seed, interval
ketidakpastian, audit kemiripan antarsplit, latensi inferensi, atau puncak memori.
Jumlah parameter bukan bukti efisiensi waktu atau memori.

Grad-CAM dalam folder hasil menargetkan `norm5` pada cabang DenseNet. Empat contoh
benar dan empat contoh salah seluruhnya berlabel asli cumulus; contoh tersebut
bersifat ilustratif, bukan sampel representatif tujuh kelas atau visualisasi
seluruh jalur CloudViT. Tidak ada klaim sebab-akibat dari peta perhatian.
