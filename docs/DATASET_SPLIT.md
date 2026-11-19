# Pembagian datasetB untuk penelitian yang dilaporkan

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Dataset dan lokasi

Run ini memakai repartisi lokal tujuh kelas GCD pada
`data/processed/dataset_b/{train,val,test}/<kelas>/`.
Nama `dataset_b` adalah nama eksperimen lokal, bukan bukti bahwa sumber datanya
berbeda dari GCD. Pemeriksaan sebelumnya pada revisi jurnal menunjukkan arsip
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

`scripts/split_dataset_experiment.py` mengumpulkan citra per kelas dari arsip,
menggabungkannya ke pool kelas, mengacak dengan seed yang ditentukan, kemudian
membagi ke tiga split. Konfigurasi datasetB mencatat rasio `(0.70,0.20,0.10)`
dan seed 42. Ini berbeda dari prosedur lama yang mengambil validasi 15% hanya
dari folder train asal.

Pembagian lama **8.501/1.499/9.000** dan konstanta lama
`N_SPLIT=(8340,1471,9000)` **tidak berlaku untuk hasil datasetB ini**.
Uji 1.901 citra adalah held-out pada repartisi lokal; bukan test resmi GCD yang
dipertahankan utuh. Pool repartisi dapat mencakup train/test arsip asal.

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
