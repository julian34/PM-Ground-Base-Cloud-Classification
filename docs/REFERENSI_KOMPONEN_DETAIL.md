# Referensi dan batas interpretasi penelitian datasetB

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Referensi bukan hasil eksperimen lokal

Dokumen ini menghubungkan rancangan dengan referensi yang sudah digunakan pada
draf jurnal. Akurasi publikasi lain tidak dipindahkan ke hasil datasetB dan tidak
digunakan sebagai baseline eksperimen ini. Daftar bibliografi draf tersedia pada
[references.bib](../laporan/ai-writing-space/papers/klasifikasi-awan-ieee/references.bib).
Revisi docs ini tidak menambahkan klaim dari jurnal lain yang belum diperiksa.

| Rujukan lokal | Hubungan dengan penelitian ini | Batas perbandingan |
|---|---|---|
| [CloudDenseNet, Sensors 2023](../laporan/jurnal/10-3390s23187957.pdf) | Rujukan modifikasi DenseNet untuk klasifikasi awan | Protokol, resolusi dan penyeimbangan publikasi berbeda; bukan baseline lokal |
| [CloudViT, CMC 2025](../laporan/jurnal/TSP_CMC_61402.pdf) | Dasar adaptasi ekstraktor lokal/global pada cabang paralel | Head diganti untuk fusi; implementasi lokal tidak diklaim reproduksi persis |
| [ALGA-DenseNet, PLOS ONE 2025](../laporan/jurnal/journal.pone.0333999.pdf) | Penelitian terkait yang menggabungkan pemodelan lokal dan global | Jalur dan protokol berbeda; tidak ada implementasi pembanding dalam hasil run |

Ketiga rujukan tersebut telah dibaca pada revisi jurnal sebelumnya. Publikasi
tambahan yang ada di folder jurnal tidak otomatis menjadi sumber hasil atau
klaim baru pada dokumentasi ini.

## 2. Pemetaan komponen yang dijelaskan

| Komponen lokal | Dasar penjelasan | Hal yang dapat dinyatakan |
|---|---|---|
| DenseNet-121 | Huang dkk., CVPR 2017; `model.py`, `blocks.py` | Koneksi dense, susunan 6/12/24/16 dan pemetaan bobot ImageNet |
| SE | Hu dkk., CVPR 2018; `attention.py` | Pembobotan kanal menggunakan ringkasan global |
| SpatialAttnBR | Implementasi mean/max + kanal lambda pada `attention.py` | Atensi spasial lokal dengan tambahan petunjuk warna; bukan seluruh CBAM asli |
| SCConv | Rumus aktual `conv.py` | Konvolusi dengan normalisasi respons, penstabil dan pangkat terlatih |
| ND | `DenseLayer.pq` pada `blocks.py` | Rasio dua fitur positif yang dipelajari |
| CloudViT | Wei dkk., CMC 2025; `cloudvit_branch.py` | Ekstraktor lokal/global 256 fitur yang diadaptasi untuk fusi |
| Focal loss | Lin dkk., ICCV 2017; `losses.py` | Pembobotan loss berdasarkan probabilitas kelas benar |
| Grad-CAM | Selvaraju dkk.; implementasi pelapor | Visualisasi pada norm5 cabang DenseNet untuk contoh yang disimpan |

Peta lambda merupakan selisih warna ternormalisasi yang dihitung sebelum
normalisasi ImageNet. Penyebutan petunjuk fisika/warna menjelaskan motivasi,
bukan bukti bahwa model mengukur sifat atmosfer atau memisahkan awan/langit
secara benar pada setiap piksel.

SCConv lokal tidak boleh otomatis disamakan dengan publikasi lain hanya karena
memiliki singkatan sama. Sifat cosine ideal juga tidak menjadi jaminan invariansi
implementasi dengan q dan epsilon. Efek masing-masing komponen terhadap akurasi
belum diisolasi lewat ablasi.

## 3. Sumber dataset

Draf merujuk TJNU Ground-based Cloud Dataset dan artikel dataset yang tercantum
pada bibliografi. Penelitian ini memakai repartisi lokal, dengan support
3.800 validasi dan 1.901 uji yang tercatat pada artefak. Rujukan dataset tidak
membuat split lokal identik dengan benchmark publikasi.

## 4. Klaim yang dibatasi oleh bukti

| Klaim yang dapat ditulis | Klaim yang belum didukung |
|---|---|
| Model fusi mencapai akurasi uji 86,74% pada split datasetB ini | Fusi terbukti lebih baik dari seluruh model tunggal |
| F1-macro uji 0,8747; tertinggi per kelas clear sky, terendah stratocumulus | Komponen tertentu menyebabkan peningkatan akurasi |
| F1 validasi log tertinggi pada epoch 70 dari 80 epoch | Epoch terakhir selalu menghasilkan model terbaik |
| Jumlah durasi epoch latih+validasi 58.354,1 detik | Model lebih cepat atau lebih hemat GPU daripada baseline |
| Contoh Grad-CAM tersedia untuk cumulus | Peta membuktikan alasan keputusan untuk seluruh tujuh kelas |

Hasil yang dipertahankan hanya berasal dari folder run yang dipilih. Pengujian
tambahan tetap disebut sebagai pekerjaan lanjutan, bukan hasil yang sudah ada.
