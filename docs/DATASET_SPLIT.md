# Pembagian datasetB untuk penelitian yang dilaporkan

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Dataset dan lokasi

Run ini memakai repartisi lokal tujuh kelas GCD pada
`data/processed/dataset_b/{train,val,test}/<kelas>/`.
**DatasetB bukan dataset independen.** Dataset ini dibentuk dengan menggabungkan
**10.000 citra training resmi GCD** dan **9.000 citra test resmi GCD** menjadi
satu pool berisi **19.000 citra**, kemudian membaginya ulang menjadi training,
validation, dan test dengan rasio target **70%:20%:10%**. Nama `dataset_b`
hanya merupakan nama eksperimen lokal. Pemeriksaan sebelumnya menunjukkan arsip
`data/raw/dataset_b.zip` dan `data/raw/GCD.zip` identik secara SHA-256:

`8c1a1ab93dcac62247e34965f1f118089a4cf21ab12f7c626c54d13662950bd1`

## 2. Jumlah citra

| Kelas | Latih | Validasi | Uji | Total |
|---|---:|---:|---:|---:|
| 1_cumulus | 1068 | 305 | 152 | 1525 |
| 2_altocumulus | 1032 | 295 | 148 | 1475 |
| 3_cirrus | 1334 | 381 | 191 | 1906 |
| 4_clearsky | 2617 | 748 | 374 | 3739 |
| 5_stratocumulus | 2545 | 727 | 364 | 3636 |
| 6_cumulonimbus | 4035 | 1153 | 576 | 5764 |
| 7_mixed | 668 | 191 | 96 | 955 |
| **Total** | **13299** | **3800** | **1901** | **19000** |

Support validasi/uji berasal dari CSV hasil. Jumlah latih dicocokkan dengan
direktori lokal pada revisi; folder hasil sendiri tidak memuat manifest anggota
latih. Rasio target **70:20:10** diterapkan per kelas dengan pembulatan, sehingga
total aktual tidak harus tepat sama dengan perkalian rasio pada total dataset.

## 3. Prosedur yang terdokumentasi

Teknik pembagian mengikuti prosedur yang dilaporkan Tu dkk. pada
ALGA-DenseNet: split resmi 10.000/9.000 digabung terlebih dahulu dan pool
19.000 citra direpartisi 70:20:10 sebelum augmentasi. Implementasi lokal pada
`scripts/split_dataset_experiment.py` mengumpulkan citra menurut kelas,
mengacaknya dengan seed 42, lalu membagi secara terstratifikasi per kelas.
Pembulatan per kelas menghasilkan total aktual 13.299/3.800/1.901.

Rujukan: B. Tu dkk., “ALGA-DenseNet ground-based cloud classification network
based on multi-scale features,” *PLOS ONE*, 20(10):e0333999, 2025,
https://doi.org/10.1371/journal.pone.0333999.

Yang diadopsi adalah **teknik penggabungan split dan repartisi data**. Ukuran
masukan, augmentasi, arsitektur, optimizer, dan jadwal pelatihan datasetB tetap
mengikuti artefak run penelitian ini dan tidak disamakan dengan ALGA-DenseNet.

Pembagian lama **8.501/1.499/9.000** dan konstanta lama
`N_SPLIT=(8340,1471,9000)` **tidak berlaku untuk hasil datasetB ini**.
Uji 1.901 citra adalah held-out pada repartisi lokal; bukan test resmi GCD yang
dipertahankan utuh. Karena kedua split resmi digabung, citra dari test resmi
GCD dapat berada pada train datasetB. Checkpoint datasetB karena itu tidak boleh
diuji kembali pada 9.000 citra test resmi dan disebut sebagai evaluasi independen.

Folder validasi datasetB sudah tersedia. Jalur kode yang mengambil validasi
secara runtime bila folder val tidak ada bukan prosedur yang digunakan untuk
menjelaskan support 3.800 pada laporan ini. Evaluasi mencakup 1.901 citra uji
dengan support tidak seimbang, bukan subset berjumlah sama per kelas.

## 4. Batas reproduksibilitas

Seed dan prosedur merupakan dokumentasi kode, bukan pengganti manifest historis.
Belum tersedia daftar anggota split berhash yang disimpan bersama checkpoint.
Identitas dua arsip tidak membuktikan ketiadaan citra duplikat atau sangat mirip
antarsplit. Tidak ada klaim bahwa model telah teruji lintas lokasi atau sensor.

Pembagian ulang bukan bagian revisi dokumentasi ini. Untuk eksperimen baru,
gunakan output terpisah dan simpan manifest; panduan ada di
[Panduan_Instalasi_Clone.md](Panduan_Instalasi_Clone.md).
