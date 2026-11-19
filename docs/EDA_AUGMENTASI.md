# Data dan augmentasi pada penelitian datasetB

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Distribusi yang diperiksa

Jumlah citra latih/validasi/uji adalah **13.299/3.800/1.901**; rincian kelas ada
di [DATASET_SPLIT.md](DATASET_SPLIT.md). Kelas latih terbesar adalah cumulonimbus
(4.035 citra), terkecil mixed (668 citra), dengan rasio sekitar **6,04:1**.
Ini merupakan ringkasan distribusi kelas, bukan analisis visual lengkap atau
audit duplikasi citra.

## 2. Pipeline transform yang terdokumentasi

Pipeline berikut berasal dari `data.build_transforms`, dengan ukuran 224 yang
dicatat laporan run. Snapshot lengkap transform saat pelatihan tidak tersimpan
di checkpoint, sehingga pipeline tidak dianggap bukti eksperimen ablasi.

| Urutan | Transform latih | Parameter pada masukan 224 |
|---|---|---|
| 1 | Resize | 224×224 |
| 2 | Reflect padding | 56 piksel per sisi; ukuran sementara 336×336 |
| 3 | RandomAffine | Rotasi ±15°; translasi maksimum 10% |
| 4 | CenterCrop | 224×224 |
| 5 | RandomHorizontalFlip | Peluang 0,5 |
| 6 | ColorJitter | Brightness 0,8–1,3; contrast 0,8–1,4; saturation 0,9–1,1; hue 0 |
| 7 | ToTensor | RGB dalam rentang [0,1] |

Augmentasi dilakukan saat sampel dimuat. Jumlah berkas unik tidak bertambah.
Reflect padding bertujuan menyediakan konteks tepi saat transform; ini tidak
menjamin semua artefak batas hilang. Hue 0 meniadakan jitter hue, tetapi perubahan
brightness, contrast dan saturation tetap dapat mengubah nilai kanal dan peta B/R.
Belum ada pengujian terpisah untuk menyatakan peningkatan akurasi tiap augmentasi.

Validasi dan uji memakai **Resize → ToTensor**, tanpa augmentasi acak. Pada
`siapkan_batch`, peta warna dihitung dari RGB [0,1] lebih dahulu, lalu citra
dinormalisasi memakai mean `(0.485,0.456,0.406)` dan std `(0.229,0.224,0.225)`.

## 3. Sampling dan loss

Laporan mencatat penyeimbangan dengan sampler. Implementasi terdokumentasi memakai
`WeightedRandomSampler`, bobot `1/n_k` dan pengambilan dengan pengembalian.
Konfigurasi mencatat **21.000 penarikan per epoch**, bukan 21.000 citra unik atau
tepat 3.000 contoh setiap kelas. Distribusi kelas seimbang dalam harapan, bukan
jaminan setiap batch seimbang.

Dengan batch 32 dan `drop_last=True` pada kode, 21.000 indeks sampler menghasilkan
656 batch penuh atau 20.992 contoh yang diproses bila konfigurasi ini diterapkan.
Angka tersebut adalah konsekuensi kode, bukan hitungan contoh yang tercatat pada CSV.
Validasi/uji tidak memakai sampler penyeimbang dan tetap mengikuti support asli.

Focal Loss gamma 2 mengurangi kontribusi contoh yang mudah. Identitas kelas
minoritas tidak otomatis sama dengan contoh sulit; efek sampler dan loss belum
diisolasi lewat ablasi.

## 4. Opsi yang tidak menjadi hasil penelitian tambahan

Konfigurasi yang diperiksa menyatakan `STRONG_AUG=False`, `MIXUP_P=0`,
`LABEL_SMOOTHING=0`, dan `GROUPED_SPLIT=False`. Opsi kode tersebut tidak boleh
dinarasikan sebagai eksperimen tambahan yang telah berhasil atau sebagai
konfigurasi historis yang diverifikasi dari bobot.

Kurva menunjukkan akurasi latih di bawah validasi. Augmentasi, sampler, dropout
dan perbedaan mode model dapat berkontribusi, tetapi penyebab dan besarnya
kontribusi belum diuji terpisah. Selisih itu tidak membuktikan bebas overfitting
atau bebas kebocoran. Hasil yang dapat dilaporkan ada pada
[HASIL_DATASET_B.md](HASIL_DATASET_B.md).
