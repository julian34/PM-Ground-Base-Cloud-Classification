# Status panduan Google Colab

Dokumen ini adalah catatan lingkungan alternatif, **bukan protokol eksperimen
datasetB_laporan_19-11-2026_0251**. Penulis mengonfirmasi pelatihan dan evaluasi
run tersebut memakai NVIDIA GeForce RTX 2060 SUPER dengan VRAM sekitar 8 GB
dan RAM sekitar 32 GB. Tidak ada bukti bahwa hasil itu berasal dari GPU Colab.

## Untuk menggunakan hasil penelitian ini

Baca [HASIL_DATASET_B.md](HASIL_DATASET_B.md), CSV dan PNG asli. Tidak perlu
memasang ulang dependensi, membagi ulang dataset, atau menjalankan training
di Colab untuk menyusun laporan hasil yang sudah tersedia.

## Jika kelak menjalankan eksperimen Colab

Eksperimen tersebut harus mempunyai nama run, direktori output, manifest data,
versi kode/konfigurasi dan catatan perangkat tersendiri. Jangan menganggap GPU,
versi paket, atau metriknya sama dengan penelitian ini. Konfigurasi aktif saat
revisi memakai masukan 512, sedangkan laporan datasetB mencatat 224×224.

Panduan lama yang membagi validasi 15% dari train asal tidak digunakan untuk
menjelaskan datasetB 70:20:10. Contoh persiapan split terpisah tersedia di
[Panduan_Instalasi_Clone.md](Panduan_Instalasi_Clone.md). Langkah khusus layanan
Colab perlu diperiksa kembali saat eksperimen baru benar-benar dilakukan;
revisi ini tidak memvalidasi lingkungan Colab atau melakukan eksperimen di sana.
