# Penjelasan matematis model fusi datasetB

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

Rumus di sini menjelaskan operasi implementasi. Contoh perhitungan adalah
ilustrasi matematis, bukan nilai aktivasi yang diukur saat pelatihan. Hasil
penelitian tetap mengacu pada [HASIL_DATASET_B.md](HASIL_DATASET_B.md).

## 1. Koneksi dense dan ukuran fitur

Layer ke-l menerima konkatenasi fitur sebelumnya:

$$x_l = H_l([x_0,x_1,\ldots,x_{l-1}]),\qquad C_{out}=C_{in}+Lk.$$

Dengan growth rate k=32, blok 6/12/24/16 layer menghasilkan kanal
256/512/1024/1024. Ukuran konvolusi dihitung sebagai

$$W_{out}=\left\lfloor\frac{W_{in}+2p-k}{s}\right\rfloor+1.$$

Untuk input 224, stem menghasilkan 56×56; tiga transisi menghasilkan 28×28,
14×14 dan 7×7. Uraian tersebut khusus ukuran pada laporan run, bukan pengaturan
512 yang kini tertulis di konfigurasi aktif.

## 2. DenseLayer dan fitur ND

DenseLayer lokal menjalankan BN → ReLU → conv 1×1 ke 128 kanal → BN → ReLU → SE.
Fitur kemudian diproses oleh SCConv 24 kanal serta jalur `pq` 16 kanal. Setelah
softplus, keluaran `pq` dibagi menjadi P dan Q, masing-masing delapan kanal:

$$F_{ND}=\frac{P-Q}{P+Q+10^{-3}}.$$

Karena P dan Q positif, nilai rasio berada di antara −1 dan 1. Contoh P=3, Q=1
memberi 2/4,001, mendekati 0,5. P dan Q adalah fitur terlatih, bukan kanal fisik
sensor. Hasil concat 24+8 menambah tepat 32 kanal pada koneksi dense.

## 3. Squeeze-and-excitation

$$z_c=\frac{1}{HW}\sum_{i,j}X_{c,i,j},\qquad
g=\sigma(W_2\,\mathrm{ReLU}(W_1z)),\qquad X'_c=g_cX_c.$$

Implementasi menggunakan conv 1×1 dengan bias, reduksi kanal 16, dan sigmoid.
Bobot lapisan akhir SE diinisialisasi nol dengan bias 3 sehingga gerbang awal
sekitar sigmoid(3)=0,953. Nilai ini mendekati, tetapi tidak sama dengan, identitas.
Tidak ada hasil ablasi yang membuktikan besarnya manfaat SE pada datasetB.

## 4. SCConv lokal

$$d=W*x,\quad u=\mathrm{clip}\left(
\frac{d}{(\lVert x_{patch}\rVert+q)(\lVert W\rVert+\epsilon)},-1,1\right),
\quad y=\mathrm{sign}(d)(|u|+\epsilon)^\tau.$$

Pada kode, norma patch memasukkan epsilon di dalam akar, q=exp(log_q)>0,
dan tau=exp(log_tau)>0 per kanal. Operasi memakai padding reflect dan float32
di dalam blok autocast yang dinonaktifkan. Istilah SCConv pada proyek ini
merujuk konvolusi cosine lokal tersebut.

Cosine ideal tanpa penstabil bersifat invarian terhadap perkalian skala positif.
Namun implementasi memiliki q dan epsilon: saat input dikali a, penyebut menjadi
`(a ||x|| + q)(||W|| + epsilon)` jika mengabaikan epsilon norma, sehingga skala
tidak dapat dibatalkan sepenuhnya. Karena itu tidak tepat menyatakan invariansi
pencahayaan mutlak atau ketahanan lintas kamera berdasarkan rumus ini saja.

## 5. Peta warna dan atensi spasial

Peta dihitung dari RGB [0,1] sebelum normalisasi ImageNet:

$$r=\frac{B-R}{B+R+10^{-6}},\qquad
\lambda=0.5\tanh\left(\frac{r-\operatorname{median}_{M}(r)}{0.5}\right)M.$$

Mask M bernilai satu bila maksimum kanal >0,10 dan minimum kanal <0,97.
Median dihitung atas piksel valid; bila tidak ada, kode memakai nol. Rentang
lambda dibatasi tanh. Tanda lambda menyatakan posisi relatif terhadap median
warna citra, bukan label awan/langit yang sudah dipastikan.

Peta diubah ukurannya ke resolusi fitur dan diskalakan 10 menurut kode:

$$A=\sigma(\operatorname{Conv}_{7\times7}
([\operatorname{mean}_cX,\operatorname{max}_cX,10\lambda])),\qquad Y=X\odot A.$$

Ini merupakan modifikasi atensi spasial dengan petunjuk warna. Bila lambda
tidak diberikan, kanal tambahannya nol; modul tiga kanal tersebut tidak berubah
menjadi modul CBAM publikasi yang identik. Belum tersedia ablasi peta warna.

## 6. Fusi dengan CloudViT

$$z_D=\operatorname{GAP}(F_D)\in\mathbb{R}^{1024},\quad
z_V=\operatorname{GAP}(F_V)\in\mathbb{R}^{256},\quad
z=[z_D;z_V]\in\mathbb{R}^{1280}.$$

Kepala bersama mempunyai dimensi **1280→512→128→32→16→7**. Setiap lapisan
tersembunyi diikuti BN1d dan ReLU; dropout 0,8 menurut laporan berada sebelum
linear terakhir. Rincian CloudViT ada pada [CABANG_VIT_DETAIL.md](CABANG_VIT_DETAIL.md).
Model bukan hanya DenseNet dengan head 1.024 fitur.

## 7. Fungsi objektif dan sampling

$$L=-\alpha(1-p_t)^\gamma\log p_t.$$

Kode menggunakan alpha=1, gamma=2, kemudian merata-ratakan loss sampel.
Faktor `(1-p_t)^2` adalah 0,01 untuk p_t=0,9 dan 0,49 untuk p_t=0,3.
Ini menunjukkan peredaman contoh mudah, bukan bukti peningkatan akurasi run.
Sampler memakai bobot kebalikan frekuensi kelas dan pengambilan dengan
pengembalian. Kelas minoritas tidak otomatis mempunyai loss lebih tinggi.

## 8. Inisialisasi dan transfer learning

LeCun uniform pada lapisan yang ditunjuk memakai

$$W\sim U(-a,a),\qquad a=\sqrt{3/\mathrm{fan\_in}},\qquad
\operatorname{Var}(W)=1/\mathrm{fan\_in}.$$

`_init_lapisan_baru` menerapkannya pada **TopBlock dan konvolusi pq**.
SE dan SpatialAttnBR mempertahankan inisialisasi gerbangnya; cabang CloudViT
tidak otomatis seluruhnya memakai fungsi LeCun ini. Backbone menerima pemetaan
bobot DenseNet-121 ImageNet, termasuk penyesuaian filter untuk SCConv/ND.
Penolaan kontribusi awal ND lewat bobot pembaca nol tidak membuat seluruh model
identik dengan DenseNet ImageNet karena operasi lain dan head sudah berbeda.

## 9. Tujuh tahap pelatihan

Jadwal epoch/LR terdapat pada ringkasan hasil dan terkonfirmasi log 80 epoch.
Perilaku mode dijelaskan dari `finetune.py`:

- **Frozen (epoch 1–20):** parameter yang ada pada `imagenet_keys` dibekukan;
  head dan parameter baru, termasuk CloudViT, dapat dilatih. Jalur `.pq.`
  dikecualikan dari pelatihan pada mode ini.
- **Partial (epoch 21–45):** membuka block 3–4 beserta atensi terkait,
  transisi 3 dan norm5, di samping parameter baru yang sudah aktif.
- **Full (epoch 46–80):** seluruh parameter dibuka.

Dengan `USE_SCCONV=True`, kode menetapkan `bn_beku=False`, sehingga helper
`set_bn_beku_eval` tidak membekukan running statistics BN pada jalur ini.
Pembekuan parameter bobot tidak sama dengan pembekuan statistik BatchNorm.

Kode yang diperiksa tidak membuktikan semua detail konfigurasi historis run.
CSV tidak menyimpan daftar trainable, status optimizer, ataupun snapshot kode.

## 10. Hubungan dengan hasil

Hasil uji adalah akurasi **86,74%** dan F1-macro **0,8747** pada 1.901 citra.
Rumus serta alasan rancangan di atas tidak mengisolasi penyebab hasil tersebut.
Keunggulan terhadap model tunggal, kontribusi tiap komponen dan efisiensi
komparatif masih memerlukan baseline/ablasi pada protokol yang sama.
