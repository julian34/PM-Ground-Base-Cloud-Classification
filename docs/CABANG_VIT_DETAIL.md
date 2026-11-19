# Cabang CloudViT pada model fusi datasetB

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Identitas cabang yang dipakai

Laporan mencatat `USE_VIT=True` dan `VIT_ARCH="cloudvit"`. Struktur checkpoint
yang diperiksa pada revisi jurnal mendukung cabang CloudViT dengan keluaran
**256 fitur** dan head bersama **1.280 fitur**. Uraian di bawah mengikuti
`cloudvit_branch.py` dan pemanggilannya pada `model.py` untuk input 224×224.

Cabang ini membaca tensor citra yang sama dengan cabang DenseNet: RGB setelah
normalisasi ImageNet. Ia mempunyai stem sendiri dan tidak mengambil peta stem
64×56×56 dari DenseNet. Dua cabang bertemu pada konkatenasi fitur sebelum head.

## 2. Alur tensor

| Tahap | Modul | Bentuk keluaran |
|---|---|---|
| Input | Citra ternormalisasi | B×3×224×224 |
| I | Conv 3×3 stride 2 → BN → SiLU | B×16×112×112 |
| L1 | DModule → EModule | B×32×56×56 |
| L2 | DModule → EModule | B×64×28×28 |
| G1 | DModule → GModule | B×128×14×14 |
| G2 | DModule → GModule | B×192×7×7 |
| G3 | DModule → GModule | B×256×4×4 |
| Pooling | Adaptive average pool → flatten | B×256 |
| Fusi | Concat dengan vektor DenseNet B×1024 | B×1280 |
| Kepala bersama | 1280→512→128→32→16→7 | B×7 |

## 3. Modul lokal dan global

**DModule** mengecilkan resolusi memakai dua jalur stride 2. Jalur pertama
memakai depthwise-separable convolution, BN dan SiLU. Jalur kedua menambahkan
konvolusi pointwise serta efficient channel attention (ECA). Kedua keluaran
digabung dan dikenai channel shuffle.

**EModule** membagi kanal menjadi dua. Separuh melewati jalur identitas dan
separuh diproses dengan pointwise, depthwise-separable convolution, BN, SiLU
dan ECA. Keluaran digabung kembali. Operasi tersebut berupa konkatenasi,
bukan penjumlahan residual pada seluruh tensor.

**GModule** memakai depthwise-separable convolution, mengubah setiap posisi
spasial menjadi satu token, menerapkan separable self-attention dan MLP dengan
residual/pre-layer normalization, lalu membentuk kembali peta fitur dan memakai
konvolusi pointwise. Untuk tiga resolusi, jumlah token adalah **196, 49, dan 16**.

Konfigurasi yang diperiksa memasok `CLOUDVIT_GDEPTH=1` dan dropout 0,1.
Default argumen konstruktor yang bernilai 2 bukan kedalaman yang dipasok oleh
`model.py` melalui konfigurasi ini. Kedalaman/dropout dijelaskan sebagai pilihan
implementasi lokal; tidak dianggap hasil optimasi hyperparameter pada run ini.

## 4. Separable self-attention

Untuk matriks token X, implementasi membentuk skor konteks scalar per token
`s = softmax(X W_I)` sepanjang sumbu token, vektor konteks
`c = sum_token(s * X W_K)`, dan keluaran `Y = (ReLU(X W_V) * c) W_O`.
Pada kode, lapisan linear juga memiliki bias. Biaya agregasi terhadap jumlah
token bersifat linear untuk dimensi tetap; ini tidak membuktikan model lengkap
lebih cepat daripada baseline yang belum diukur.

Cabang tidak memakai token CLS atau embedding posisi eksplisit. Ringkasannya
berasal dari global average pooling, bukan pengambilan token pertama.

## 5. Bagian yang tidak boleh tercampur

`ViTBranch` sederhana pada `vit_branch.py` adalah alternatif kode, dengan
embedding 192, token CLS dan TransformerEncoder. Uraian patch 8 pada stem,
50 token, atau head `1024+192=1216` hanya berlaku pada konfigurasi alternatif
tersebut, **bukan model CloudViT yang dilaporkan**. Tidak ada hasil uji varian
sederhana atau Transformer tunggal dalam folder acuan.

CloudViT lokal merupakan adaptasi: head publikasi diganti head fusi tujuh kelas,
dan resolusi tahap global lokal pada input 224 adalah 14/7/4. Publikasi yang
dibaca pada revisi jurnal mencantumkan 14/8/4. Tidak diklaim reproduksi persis
arsitektur maupun akurasi publikasi. Lihat [referensi](REFERENSI_KOMPONEN_DETAIL.md).

## 6. Pembekuan dan interpretasi

Pada mode frozen, bobot yang dipetakan dari ImageNet dibekukan, sedangkan
parameter baru termasuk cabang CloudViT dapat dilatih menurut kode. Jalur ND
`pq` mempunyai pengecualian pembekuan. Mode ini tidak berarti hanya head yang
dilatih. Perubahan ke partial/full mengikuti jadwal pada ringkasan hasil.

Tidak ada ablasi yang memisahkan kontribusi CloudViT terhadap F1 atau akurasi.
Grad-CAM yang tersedia menargetkan cabang DenseNet dan tidak boleh disebut
visualisasi keseluruhan cabang CloudViT.
