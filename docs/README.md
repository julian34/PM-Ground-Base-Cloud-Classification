# Dokumentasi penelitian datasetB

Dokumentasi ini dibatasi pada hasil di
`artifacts/results/datasetB_laporan_19-11-2026_0251`.
Mulai dari [HASIL_DATASET_B.md](HASIL_DATASET_B.md) untuk angka, asal bukti,
koreksi perangkat oleh penulis, dan keterbatasan penelitian.

| Dokumen | Isi |
|---|---|
| [Hasil datasetB](HASIL_DATASET_B.md) | Metrik, 80 epoch, waktu, konfigurasi, perangkat dan batas kesimpulan |
| [Pembagian dataset](DATASET_SPLIT.md) | Repartisi lokal 13.299/3.800/1.901 dan support per kelas |
| [Data dan augmentasi](EDA_AUGMENTASI.md) | Distribusi kelas dan pipeline terdokumentasi |
| [Arsitektur](ARSITEKTUR_CLOUDDENSENET.md) | Model fusi DenseNet lokal + CloudViT yang dievaluasi |
| [Penjelasan model](PENJELASAN_MODEL_DETAIL.md) | Rumus, alur tensor, inisialisasi dan pelatihan |
| [Cabang CloudViT](CABANG_VIT_DETAIL.md) | Ekstraktor 256 fitur dan fusi 1.280 fitur |
| [Referensi komponen](REFERENSI_KOMPONEN_DETAIL.md) | Hubungan referensi, implementasi, dan batas klaim |
| [Instalasi](Panduan_Instalasi_Clone.md) | Panduan operasional untuk run baru, bukan rekaman eksperimen |
| [Google Colab](Panduan_Google_Colab.md) | Status panduan alternatif, bukan lingkungan penelitian ini |
| [Pembaruan repositori](Panduan_Pull.md) | Panduan kolaborasi, bukan protokol penelitian |

## Aturan penggunaan untuk laporan penelitian

1. Gunakan satu run ini untuk hasil. Jangan mengambil angka dari folder run lain.
2. Bedakan skor validasi selama pelatihan (terbaik 0,890024) dengan evaluasi ulang
   checkpoint (F1-macro validasi 0,8895). Penyebab selisih belum ditetapkan.
3. Hasil tersedia untuk model fusi saja. Model tunggal dan ablasi belum memiliki
   hasil pembanding pada protokol yang sama.
4. Rancangan, fungsi matematis, dan motivasi komponen tidak membuktikan kontribusi
   akurasi tiap komponen. Pernyataan kausal memerlukan eksperimen tambahan.
5. GPU pelatihan dan evaluasi dikoreksi penulis menjadi RTX 2060 SUPER, VRAM 8 GB,
   RAM sekitar 32 GB. Metadata perangkat pada laporan otomatis berbeda dan tetap
   disimpan sebagai sumber asli; lihat catatan koreksi pada ringkasan hasil.

Skema `ArsitekturDensenet121.drawio.png` adalah aset ilustrasi DenseNet yang belum
diverifikasi sebagai diagram model fusi run ini. Untuk arsitektur penelitian,
gunakan diagram teks pada dokumen arsitektur. Gambar hasil eksperimen mengacu
langsung ke PNG dalam folder hasil, bukan ilustrasi tersebut.

Salinan dokumen sebelum koreksi disimpan di
`artifacts/documentation_revision_2026-11-19/before/` sebagai riwayat, bukan sumber
hasil penelitian datasetB. Kode, bobot, dataset, dan artefak hasil asli tidak diubah.

Pemeriksaan pada `artifacts/documentation_revision_2026-11-19/validation.json`
mencakup 11 dokumen, 42 tautan lokal, tabel metrik, hitungan split, jadwal tujuh
tahap dan log 80 epoch. Hash seluruh 10 berkas pada folder hasil tetap sama.
Pemeriksaan ini memvalidasi pelaporan; tidak menjalankan eksperimen baru.
