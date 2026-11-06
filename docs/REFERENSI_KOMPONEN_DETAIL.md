# Referensi Komponen CloudDenseNet — Bedah Detail per Makalah

Dokumen ini membedah **satu per satu** ide di balik tiap komponen CloudDenseNet, dikaitkan
dengan **makalah aslinya**. Untuk setiap komponen dijelaskan empat hal dengan bahasa
sederhana:

- 🧩 **Masalah** — kenapa ide ini sampai diciptakan.
- 💡 **Ide inti** — kontribusi utama makalahnya.
- 🧮 **Matematika** — rumus pokoknya.
- 🔌 **Di CloudDenseNet** — dipakai di file & fungsi mana, dan apa yang dimodifikasi.

> **Hubungan dokumen:**
> [`ARSITEKTUR_CLOUDDENSENET.md`](ARSITEKTUR_CLOUDDENSENET.md) = ringkasan arsitektur ·
> [`PENJELASAN_MODEL_DETAIL.md`](PENJELASAN_MODEL_DETAIL.md) = alur neuron + bukti hitung ·
> **dokumen ini** = bedah tiap makalah referensi.
> Sumber kebenaran tetap kode di `src/clouddensenet/`. **Tidak ada kode yang diubah.**

---

## Daftar isi

| # | Komponen (di kode) | Makalah | Venue |
|---|--------------------|---------|-------|
| 1 | `DenseBlock*`, `Transition*` | DenseNet | CVPR 2017 |
| 2 | `SEBlock` | Squeeze-and-Excitation | CVPR 2018 / TPAMI 2019 |
| 3 | `SpatialAttn(BR)` | CBAM (atensi spasial) | ECCV 2018 |
| 4 | `br_ratio_lambda` | Rasio Biru/Merah (fisika awan) | 2006 / 2010 |
| 5 | `SCConv` | Cosine Normalization | arXiv 2017 / ICANN 2018 |
| 6 | `FocalLoss` | Focal Loss | ICCV 2017 |
| 7 | `BatchNorm2d/1d` | Batch Normalization | ICML 2015 |
| 8 | `lecun_uniform_` | Inisialisasi skala-varians | 1998 |
| 9 | `muat_imagenet` + `finetune.py` | Transfer learning | CVPR 2009 / NeurIPS 2014 |

---

## 1. DenseNet — tulang punggung

> **Huang, Liu, van der Maaten & Weinberger.** *Densely Connected Convolutional Networks.*
> **CVPR 2017** (Best Paper). arXiv:1608.06993.
> Di kode: `blocks.DenseLayer`, `DenseBlock1-4`, `TransitionLayer1-3`.

🧩 **Masalah.** Jaringan yang sangat dalam menderita *vanishing gradient* — sinyal gradien
mengecil saat merambat mundur ke layer awal, sehingga layer awal "sulit belajar". ResNet
(2015) menambal ini dengan *skip connection* penjumlahan. DenseNet mengambil ide itu lebih
jauh.

💡 **Ide inti.** Di dalam satu *dense block*, **setiap layer menerima masukan dari SEMUA
layer sebelumnya** (lewat *concatenation*, bukan penjumlahan), dan mengirim fiturnya ke
semua layer sesudahnya. Untuk $L$ layer ada $\frac{L(L+1)}{2}$ koneksi langsung. Akibatnya:
jalur gradien selalu pendek, dan fitur dipakai ulang (*feature reuse*) sehingga **parameter
lebih hemat**.

🧮 **Matematika.** Layer ke-$\ell$ menghitung:

$$
x_\ell = H_\ell\big(\,[\,x_0, x_1, \dots, x_{\ell-1}\,]\,\big)
$$

dengan $[\cdot]$ = concat channel, dan $H_\ell$ = fungsi gabungan **BN → ReLU → Conv**.
Tiap $H_\ell$ menambah tepat $k$ channel baru (**growth rate**, di sini $k=32$). Dua trik
penting DenseNet-BC yang juga dipakai di repo:

- **Bottleneck:** Conv 1×1 menghasilkan $4k = 128$ channel sebelum conv 3×3 → biaya
  komputasi tetap, tak peduli masukan membesar.
- **Compression:** *transition layer* menyusutkan channel (faktor $\theta = 0.5$) dan
  spasial (AvgPool) → `256→128`, `512→256`, `1024→512`.

🔌 **Di CloudDenseNet.** Susunan block **`[6, 12, 24, 16]`** = **DenseNet-121 persis**,
itulah yang membuat bobot ImageNet bisa dipindah langsung (bagian 9). Yang dimodifikasi:
di dalam `DenseLayer`, conv 3×3 standar diganti **SCConv** (bagian 5) dan ditambah **cabang
ND** (lihat [`PENJELASAN_MODEL_DETAIL.md` §5.6](PENJELASAN_MODEL_DETAIL.md#56-channel-nd-pqpq)),
sehingga 32 channel per layer = 24 (SCConv) + 8 (ND).

---

## 2. SEBlock — Squeeze-and-Excitation (atensi channel)

> **Hu, Shen & Sun.** *Squeeze-and-Excitation Networks.* **CVPR 2018**; versi perluasan
> **IEEE TPAMI 2019**. arXiv:1709.01507. (Juara ILSVRC 2017 klasifikasi.)
> Di kode: `attention.SEBlock`.

🧩 **Masalah.** Konvolusi memperlakukan semua channel keluaran "setara" — padahal untuk
gambar tertentu, sebagian channel (mis. detektor tekstur awan) jauh lebih informatif
daripada yang lain. Tidak ada mekanisme eksplisit untuk **menimbang antar-channel**.

💡 **Ide inti.** Tambahkan modul ringan yang **mempelajari bobot per-channel** secara
adaptif terhadap isi gambar, lewat dua langkah: *squeeze* (ringkas konteks global) lalu
*excitation* (hasilkan gerbang 0–1 per channel).

🧮 **Matematika.**

1. **Squeeze** — ringkas tiap channel jadi satu angka via *global average pooling*:
   $$ z_c = \frac{1}{H W}\sum_{i,j} x_{c,i,j} $$
2. **Excitation** — dua lapisan dengan rasio reduksi $r$, lalu gerbang:
   $$ s = \sigma\!\big(W_2\,\delta(W_1 z)\big), \qquad \tilde{x}_c = s_c \cdot x_c $$
   ($\delta$=ReLU, $\sigma$=sigmoid, $r=16$). "Reduksi" $C \to C/r \to C$ menjaga modul tetap
   murah.

🔌 **Di CloudDenseNet.** Dipakai **dua tempat**: (a) di dalam tiap `DenseLayer` pada
bottleneck 128 channel, dan (b) sebagai `se1..se4` di akhir tiap block (256/512/1024/1024).
Modifikasi khas repo: **diinisialisasi ≈ identitas** — bobot konv terakhir di-nol-kan, bias
= `SE_INIT_BIAS = 3.0`, sehingga awalnya $s_c = \sigma(3) \approx 0.953$. Tujuannya agar
atensi **tidak merusak bobot ImageNet** di epoch awal (bukti hitung di
[`PENJELASAN_MODEL_DETAIL.md` §5.3](PENJELASAN_MODEL_DETAIL.md#53-seblock--atensi-channel)).
Di kode SE dibuat dengan **Conv1×1** (bukan `Linear`), tetapi secara matematis identik.

---

## 3. CBAM — modul atensi spasial

> **Woo, Park, Lee & Kweon.** *CBAM: Convolutional Block Attention Module.* **ECCV 2018.**
> arXiv:1807.06521.
> Di kode: `attention.SpatialAttn` (dan turunannya `SpatialAttnBR`).

🧩 **Masalah.** SE hanya menjawab "*channel* mana yang penting". Ia tidak tahu "*lokasi*
mana di gambar yang penting" (mis. langit di pojok vs. matahari silau di tengah).

💡 **Ide inti.** CBAM memakai **dua atensi berurutan**: *Channel Attention* (mirip SE, tapi
menggabungkan avg-pool **dan** max-pool) lalu *Spatial Attention*. Yang dipakai repo adalah
**modul spasialnya**: ringkas informasi lintas-channel di tiap piksel dengan rata-rata &
maksimum, lalu satu konvolusi menghasilkan peta perhatian.

🧮 **Matematika.** Modul spasial CBAM:

$$
M_s(F) = \sigma\!\Big(f^{7\times7}\big(\,[\,\text{AvgPool}_c(F)\,;\,\text{MaxPool}_c(F)\,]\,\big)\Big),
\qquad F' = F \odot M_s(F)
$$

Pooling dilakukan **sepanjang sumbu channel** → menghasilkan 2 peta (rata-rata & maks),
di-concat jadi 2 channel, lalu konv 7×7 → 1 peta perhatian $\in (0,1)$.

🔌 **Di CloudDenseNet.** `SpatialAttn` = modul spasial CBAM persis (konv 7×7, padding
`reflect`, init ≈ identitas bias 3.0). Yang **ditambahkan repo**: `SpatialAttnBR` menyisipkan
**channel ketiga** berisi peta fisika λ (B/R), sehingga konvolusinya menerima **3 channel**
`[mean, max, λ]`. Inilah jembatan ke komponen berikutnya. (Detail & contoh angka di
[`PENJELASAN_MODEL_DETAIL.md` §5.5](PENJELASAN_MODEL_DETAIL.md#55-spatialattnbr--peta-λ-br-sadar-fisika).)

---

## 4. Rasio Biru/Merah — pengetahuan fisika awan

> **(a) Long, Sabburg, Calbó & Pagès.** *Retrieving Cloud Characteristics from Ground-Based
> Daytime Color All-Sky Images.* **J. Atmos. Oceanic Technol., 2006.**
> **(b) Heinle, Macke & Srivastav.** *Automatic Cloud Classification of Whole Sky Images.*
> **Atmos. Meas. Tech., 2010.**
> Di kode: `spectral.br_ratio_lambda`, `lambda_relatif`.

🧩 **Masalah.** CNN belajar fitur dari nol lewat data. Tapi untuk awan, ada **pengetahuan
fisika gratis** yang sangat membedakan langit vs. awan — kenapa tidak dipakai sebagai
petunjuk?

💡 **Ide inti.** Karena *hamburan Rayleigh*, langit cerah menghamburkan cahaya biru lebih
kuat → **langit tampak biru** (Biru > Merah). Awan memantulkan semua panjang gelombang
hampir merata → **tampak putih/abu** (Biru ≈ Merah). Para peneliti kamera *all-sky*
(Long dkk. 2006; Heinle dkk. 2010) memakai **rasio Merah/Biru (atau Biru/Merah)** sebagai
fitur utama untuk memisahkan piksel awan dari langit. Heinle dkk. bahkan mengklasifikasikan
langit ke beberapa kelas memakai fitur spektral (termasuk selisih/rasio R–B) plus tekstur.

🧮 **Matematika.** Peta λ di repo:

$$
\lambda_{\text{raw}} = \frac{B - R}{B + R + \varepsilon} \in (-1, 1)
$$

$\lambda > 0$ → kebiruan (langit), $\lambda \approx 0$ → keputihan (awan). Lalu dibuat
**relatif** terhadap median per-gambar dan di-*soft-clamp*:

$$
\lambda = 0.5\,\tanh\!\Big(\tfrac{\lambda_{\text{raw}} - \text{median}}{0.5}\Big)\cdot \text{valid}
$$

`valid` membuang piksel gelap (<0.10) & silau (>0.97). Dihitung dari RGB **[0,1] sebelum**
normalisasi ImageNet (karena normalisasi akan merusak makna warna).

🔌 **Di CloudDenseNet.** λ dihitung di `data.siapkan_batch`, lalu dialirkan ke tiap
`SpatialAttnBR` sebagai channel ke-3. Diskalakan `LAM_SCALE = 10.0` agar sebanding dengan
aktivasi lain. Inilah yang disebut **atensi "sadar-fisika"**: model diberi tahu di mana
kemungkinan langit vs. awan, bukan menebak sepenuhnya dari nol. Jika λ tidak tersedia,
modul otomatis jatuh kembali menjadi CBAM biasa.

> **Catatan sitasi dataset.** Fisika B/R di atas bersumber jelas pada Long (2006) &
> Heinle (2010). Sitasi **dataset GCD** sebaiknya dikonfirmasi penulis dari sumber aslinya
> (ada beberapa dataset langit bersingkatan mirip) — lihat
> [`PENJELASAN_MODEL_DETAIL.md` §8](PENJELASAN_MODEL_DETAIL.md#8-catatan-teknis--kejujuran).

---

## 5. Cosine Normalization — inti SCConv

> **Luo, Zhan, Wang & Yang.** *Cosine Normalization: Using Cosine Similarity Instead of Dot
> Product in Neural Networks.* **arXiv:1702.05870 (2017); ICANN 2018.**
> Di kode: `conv.SCConv`.

🧩 **Masalah.** Konvolusi/linear menghitung *dot product* $w \cdot x$ yang **tak terbatas**
dan **ikut membesar bila intensitas input besar**. Akibatnya distribusi pra-aktivasi liar
(perlu BN), dan foto terang vs. redup bisa memberi respons jauh berbeda walau polanya sama.

💡 **Ide inti.** Ganti dot product dengan **cosine similarity** — ukuran **sudut** antara
vektor, bukan panjangnya:

$$
\cos(w, x) = \frac{w \cdot x}{\lVert w\rVert \, \lVert x\rVert} \in [-1, 1]
$$

Keluaran otomatis terbatas $[-1,1]$, variansnya kecil & stabil, dan **tidak peduli skala**
input/bobot.

🧮 **Matematika (versi SCConv di repo).** Repo memperkaya cosine normalization dengan
penstabil $q$, pangkat ketajaman $\tau$ (per-channel), dan mempertahankan tanda:

$$
\cos = \text{clip}\!\left(\frac{W * x}{(\lVert x_{\text{patch}}\rVert + q)(\lVert W\rVert + \varepsilon)}, -1, 1\right),
\qquad
y = \text{sign}(W*x)\cdot\big(|\cos| + \varepsilon\big)^{\tau}
$$

$q>0$ dan $\tau>0$ dipelajari (disimpan sebagai `log_q`, `log_tau` lalu di-`exp`). Batas
$[-1,1]$ dijamin oleh **Cauchy–Schwarz**. Operasi dipaksa **fp32** (`autocast enabled=False`)
agar stabil di AMP. (Bukti **invariansi skala** dengan contoh angka ada di
[`PENJELASAN_MODEL_DETAIL.md` §5.4](PENJELASAN_MODEL_DETAIL.md#54-scconv--konvolusi-cosine-ter-normalisasi).)

🔌 **Di CloudDenseNet.** SCConv **menggantikan conv 3×3** di tiap `DenseLayer`
(`USE_SCCONV=True`). Manfaatnya pas untuk citra langit: respons bergantung pada **pola awan**,
bukan tingkat pencahayaan — tahan terhadap foto cerah/mendung/berbeda kamera.

> ⚠️ **Perhatian nama:** "SCConv" di repo = **cosine-normalized conv** ini. Jangan tertukar
> dengan makalah lain berjudul *Spatial and Channel reconstruction Convolution* (CVPR 2023)
> yang kebetulan bersingkatan sama. Yang berlaku adalah definisi di `conv.py`.

---

## 6. Focal Loss — fokus ke contoh sulit

> **Lin, Goyal, Girshick, He & Dollár.** *Focal Loss for Dense Object Detection.*
> **ICCV 2017.** arXiv:1708.02002. (Dari RetinaNet.)
> Di kode: `losses.FocalLoss`.

🧩 **Masalah.** Saat kelas sangat tidak seimbang, ribuan contoh **mudah** (sudah pasti benar)
mendominasi total loss, sehingga gradien "tenggelam" dan contoh **sulit/minoritas**
(mis. kelas `7_mixed`) kurang terlatih.

💡 **Ide inti.** Tambahkan faktor peredam $(1-p_t)^\gamma$ pada Cross-Entropy, yang
**otomatis mengecilkan** kontribusi contoh yang sudah diprediksi dengan yakin ($p_t$ besar),
sambil membiarkan contoh sulit ($p_t$ kecil) tetap berbobot penuh.

🧮 **Matematika.** Dengan $p_t$ = probabilitas softmax untuk kelas benar:

$$
\text{CE} = -\ln(p_t), \qquad \text{FL} = -\alpha\,(1 - p_t)^{\gamma}\,\ln(p_t)
$$

$\gamma$ = *focusing parameter* ($\gamma=2$ terbaik di makalah; di repo `FOCAL_GAMMA=2.0`,
$\alpha=1.0$). Di kode $p_t$ dihitung rapi lewat `pt = exp(-ce_loss)`. Bukti kuantitatif
bahwa contoh sulit jadi ~50× lebih diperhatikan dibanding CE biasa ada di
[`PENJELASAN_MODEL_DETAIL.md` §5.7](PENJELASAN_MODEL_DETAIL.md#57-focal-loss).

🔌 **Di CloudDenseNet.** Dipakai sebagai fungsi loss utama. Bekerja **berdampingan** dengan
`WeightedRandomSampler` (`IMBALANCE="sampler"` di `data.py`) — oversampling meratakan jumlah
contoh per-batch, Focal meratakan *bobot kesulitan* di dalam batch.

---

## 7. Batch Normalization — penstabil training

> **Ioffe & Szegedy.** *Batch Normalization: Accelerating Deep Network Training by Reducing
> Internal Covariate Shift.* **ICML 2015.** arXiv:1502.03167.
> Di kode: `nn.BatchNorm2d` / `BatchNorm1d` (di stem, tiap `DenseLayer`, transisi, `norm5`,
> Top Block).

🧩 **Masalah.** Saat training, distribusi masukan tiap layer terus berubah karena bobot
layer sebelumnya berubah (*internal covariate shift*). Ini memaksa pakai *learning rate*
kecil dan membuat training lambat.

💡 **Ide inti.** Normalkan aktivasi tiap channel **per mini-batch** agar rata-rata ≈ 0 dan
varians ≈ 1, lalu izinkan jaringan menggeser/menskala kembali dengan parameter belajar
$\gamma, \beta$. Efek samping: training lebih cepat, boleh LR lebih besar, dan ada efek
regularisasi ringan.

🧮 **Matematika.**

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta
$$

Saat inferensi, $\mu_B, \sigma_B^2$ diganti **running mean/variance** yang diakumulasi
selama training.

🔌 **Di CloudDenseNet.** Hadir di hampir semua blok (pola BN-ReLU-Conv khas DenseNet).
Detail halus yang penting di repo: karena **SCConv mengubah distribusi aktivasi**, BN tidak
dibekukan total saat transfer learning. `finetune.set_bn_beku_eval` hanya membekukan
*running stats* BN bawaan ImageNet pada layer yang memang beku, selebihnya dibiarkan
beradaptasi. (Lihat
[`PENJELASAN_MODEL_DETAIL.md` §6](PENJELASAN_MODEL_DETAIL.md#6-mengapa-training-nya-bertahap).)

---

## 8. Inisialisasi LeCun-uniform

> **LeCun, Bottou, Orr & Müller.** *Efficient BackProp.* Dalam *Neural Networks: Tricks of
> the Trade*, 1998. (Cikal bakal init Xavier/Glorot 2010 & He 2015.)
> Di kode: `losses.lecun_uniform_`, `init_new_layers`.

🧩 **Masalah.** Jika bobot awal terlalu besar, aktivasi (dan gradien) **meledak**; jika
terlalu kecil, **menghilang**. Keduanya menggagalkan training. Nilai awal harus "pas".

💡 **Ide inti.** Pilih bobot awal agar **varians sinyal terjaga** saat melewati layer —
patokan klasik: varians bobot = $1/\text{fan\_in}$ (banyaknya masukan ke neuron).

🧮 **Matematika.** Repo memakai distribusi **uniform**:

$$
W \sim \mathcal{U}(-L, L), \quad L = \sqrt{\tfrac{3}{\text{fan\_in}}}
\;\Rightarrow\;
\text{Var}(W) = \frac{(2L)^2}{12} = \frac{L^2}{3} = \frac{1}{\text{fan\_in}}
$$

(Turunan lengkap di
[`PENJELASAN_MODEL_DETAIL.md` §5.8](PENJELASAN_MODEL_DETAIL.md#58-inisialisasi-lecun-uniform).)

🔌 **Di CloudDenseNet.** Diterapkan **hanya ke lapisan baru** (Top Block & cabang ND `pq`)
lewat `CloudDenseNet._init_lapisan_baru`. **Sengaja tidak** menyentuh backbone (agar bobot
ImageNet aman) maupun SE/SpatialAttnBR (yang sudah di-init ≈ identitas). Bias di-nol-kan;
BN di-init $\gamma=1, \beta=0$.

---

## 9. Transfer learning ImageNet — titik awal yang cerdas

> **(a) Deng, Dong, Socher, Li, Li & Fei-Fei.** *ImageNet: A Large-Scale Hierarchical Image
> Database.* **CVPR 2009.**
> **(b) Yosinski, Clune, Bengio & Lipson.** *How transferable are features in deep neural
> networks?* **NeurIPS 2014.** arXiv:1411.1792.
> Di kode: `pretrained.muat_imagenet`, `nolkan_input_nd`, seluruh `finetune.py`.

🧩 **Masalah.** Dataset awan relatif kecil. Melatih DenseNet-121 dari nol di data kecil
cenderung **overfit** dan butuh waktu lama.

💡 **Ide inti.** Mulai dari bobot yang sudah dilatih di **ImageNet** (±1,2 juta gambar,
1000 kelas). Yosinski dkk. (2014) menunjukkan: layer **awal** belajar fitur umum (tepi,
warna, tekstur) yang **transferable** ke tugas lain, sementara layer **akhir** makin
spesifik. Maka: pertahankan fitur umum, latih ulang bagian spesifik.

🧮 **Matematika (bukan rumus, tapi prosedur).** Strategi *staged fine-tuning* di repo
(`config.PHASES`): bekukan backbone → latih kepala → buka blok dalam secara bertahap →
full defrost, dengan **LR menurun** ($10^{-3} \to 10^{-6}$) agar perubahan makin halus dan
mencegah *catastrophic forgetting*.

🔌 **Di CloudDenseNet.**
- `muat_imagenet` memetakan nama bobot torchvision DenseNet121 ke penamaan repo. Karena
  8 dari 32 filter dialihkan ke jalur ND, filter dipotong `32 → 24` saat dimuat.
- `nolkan_input_nd` menol-kan bobot yang **membaca** channel ND di awal → perilaku awal =
  **ImageNet murni**, lalu kontribusi ND tumbuh saat dilatih.
- `finetune.py` (`bekukan_imagenet` → `buka_blok_3_4` → `buka_semua`) menjalankan jadwal
  6 fase. (Tabel fase lengkap di
  [`PENJELASAN_MODEL_DETAIL.md` §6](PENJELASAN_MODEL_DETAIL.md#6-mengapa-training-nya-bertahap).)

---

## Rangkuman: "siapa menyelesaikan apa"

| Komponen | Pertanyaan yang dijawabnya |
|----------|----------------------------|
| **DenseNet** | *Bagaimana* fitur mengalir tanpa hilang & hemat parameter? |
| **SEBlock** | *Channel* mana yang penting? |
| **CBAM / SpatialAttn** | *Lokasi* mana yang penting? |
| **Rasio B/R (λ)** | *Petunjuk fisika*: mana langit, mana awan? |
| **SCConv (cosine)** | Kenali *pola*, bukan *terang-gelap*. |
| **Focal Loss** | Fokus belajar ke contoh *sulit/jarang*. |
| **BatchNorm** | Jaga training tetap *stabil & cepat*. |
| **LeCun init** | Mulai dari nilai bobot yang *pas*. |
| **Transfer learning** | Mulai dari *titik awal cerdas* (ImageNet), bukan nol. |

---

*Dokumen penjelasan — tidak ada kode yang diubah. Perbarui bila modul di
`src/clouddensenet/` berubah.*
