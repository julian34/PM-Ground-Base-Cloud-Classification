# Penjelasan Detail CloudDenseNet — Matematika, Jalur Neuron, & Referensi

Dokumen ini adalah **penjelasan mendalam** dari model **CloudDenseNet** (DenseNet-121
yang dimodifikasi untuk klasifikasi 7 kelas awan). Tujuannya tiga:

1. **Menjelaskan "mengapa"** tiap komponen ada, dikaitkan dengan **referensi peneliti**
   (jurnal & konferensi) yang relevan.
2. **Membuktikan dengan perhitungan matematika** (lengkap dengan contoh angka) bagaimana
   tiap modifikasi bekerja.
3. **Menelusuri jalur sebuah "neuron"/aktivasi** langkah demi langkah — dari piksel
   mentah sampai menjadi 7 logit kelas.

> **Catatan hubungan dokumen:** ringkasan arsitektur singkat ada di
> [`ARSITEKTUR_CLOUDDENSENET.md`](ARSITEKTUR_CLOUDDENSENET.md). Dokumen *ini* adalah
> pendalamannya. **Sumber kebenaran tetap kode** di `src/clouddensenet/`
> (`model.py`, `blocks.py`, `attention.py`, `conv.py`, `spectral.py`, `losses.py`,
> `pretrained.py`, `finetune.py`, `config.py`). Hyperparameter yang dipakai:
> `growth_rate=32`, `ND_CH=8`, `USE_SE=True`, `USE_SCCONV=True`, `USE_BR=True`,
> `NUM_CLASSES=7`, `IMG_SIZE=224`.

---

## Daftar isi

1. [Analogi sederhana dulu](#1-analogi-sederhana-dulu)
2. [Dari mana ide ini? (peta referensi)](#2-dari-mana-ide-ini-peta-referensi)
3. [Jalur makro: dari 1 gambar ke 7 angka](#3-jalur-makro-dari-1-gambar-ke-7-angka)
4. [Jalur neuron yang spesifik (telusur langkah demi langkah)](#4-jalur-neuron-yang-spesifik-telusur-langkah-demi-langkah)
5. [Bukti matematika tiap modifikasi](#5-bukti-matematika-tiap-modifikasi)
   - [5.1 Konvolusi & Batch Normalization](#51-konvolusi--batch-normalization-dasar)
   - [5.2 Koneksi dense (DenseNet)](#52-koneksi-dense-mengapa-concat)
   - [5.3 SEBlock — atensi channel](#53-seblock--atensi-channel)
   - [5.4 SCConv — konvolusi cosine ter-normalisasi](#54-scconv--konvolusi-cosine-ter-normalisasi)
   - [5.5 SpatialAttnBR + peta λ B/R](#55-spatialattnbr--peta-λ-br-sadar-fisika)
   - [5.6 Channel ND (P−Q)/(P+Q)](#56-channel-nd-pqpq)
   - [5.7 Focal Loss](#57-focal-loss)
   - [5.8 Inisialisasi LeCun-uniform](#58-inisialisasi-lecun-uniform)
6. [Mengapa training-nya bertahap?](#6-mengapa-training-nya-bertahap)
7. [Tabel referensi lengkap](#7-tabel-referensi-lengkap)
8. [Catatan teknis & kejujuran](#8-catatan-teknis--kejujuran)

---

## 1. Analogi sederhana dulu

Bayangkan sekelompok juri menilai foto langit untuk menebak jenis awan.

- **DenseNet (tulang punggung)** = tim pemeriksa bertingkat. Setiap pemeriksa baru boleh
  melihat **semua catatan pemeriksa sebelumnya**, bukan hanya catatan terakhir. Tidak ada
  informasi yang hilang di tengah jalan.
- **SEBlock** = seorang "manajer channel" yang berkata: *"Fitur bertekstur awan ini
  penting, besarkan; fitur warna ini kurang relevan, kecilkan."* Dia mengatur **kanal mana**
  yang diperkuat.
- **SpatialAttnBR** = seorang "manajer lokasi" yang berkata: *"Perhatikan area langit di
  pojok ini, abaikan matahari yang silau di tengah."* Dia mengatur **wilayah mana** yang
  diperhatikan — dan diberi petunjuk fisika: *"daerah yang kebiruan biasanya langit,
  yang keputihan biasanya awan."*
- **SCConv** = pemeriksa yang menilai **pola/bentuk**, bukan sekadar terang-gelap. Walau
  foto satu diambil saat cerah dan satu saat mendung, pola awannya tetap dikenali.
- **Focal Loss** = pelatih yang menyuruh juri **berhenti membuang waktu pada soal yang
  sudah pasti benar**, dan fokus berlatih pada soal yang masih sering salah.

Sisa dokumen ini menjelaskan tiap "juri" itu secara matematis.

---

## 2. Dari mana ide ini? (peta referensi)

Setiap modifikasi bukan buatan acak — masing-masing punya akar di literatur:

| Komponen di kode | Ide aslinya dari | Publikasi (venue) |
|------------------|------------------|-------------------|
| Backbone `DenseBlock*`, `Transition*` | **DenseNet** (koneksi dense) | Huang, Liu, van der Maaten & Weinberger, *Densely Connected Convolutional Networks*, **CVPR 2017** |
| `SEBlock` | **Squeeze-and-Excitation** (atensi channel) | Hu, Shen & Sun, *Squeeze-and-Excitation Networks*, **CVPR 2018** (versi lengkap **IEEE TPAMI 2019**) |
| `SpatialAttn` (mean+max → konv → sigmoid) | **CBAM** (modul atensi spasial) | Woo, Park, Lee & Kweon, *CBAM: Convolutional Block Attention Module*, **ECCV 2018** |
| `br_ratio_lambda` = (B−R)/(B+R) | **Rasio Biru/Merah untuk deteksi awan** | Long et al. (**J. Atmos. Oceanic Technol. 2006**); Heinle, Macke & Srivastav (**Atmos. Meas. Tech. 2010**) |
| `SCConv` (normalisasi cosine) | **Cosine Normalization** | Luo, Zhan, Wang & Yang, *Cosine Normalization…*, arXiv 2017 / **ICANN 2018** |
| `FocalLoss` | **Focal Loss** | Lin, Goyal, Girshick, He & Dollár, *Focal Loss for Dense Object Detection*, **ICCV 2017** |
| `BatchNorm2d/1d` di mana-mana | **Batch Normalization** | Ioffe & Szegedy, *Batch Normalization…*, **ICML 2015** |
| `lecun_uniform_` | **Inisialisasi skala-varians** | LeCun, Bottou, Orr & Müller, *Efficient BackProp*, 1998 |
| Transfer learning ImageNet | **Pre-training & fine-tuning** | Deng et al. *ImageNet* (**CVPR 2009**); Yosinski et al. *How transferable are features…* (**NeurIPS 2014**) |

Sitasi lengkap ada di [bagian 7](#7-tabel-referensi-lengkap).

---

## 3. Jalur makro: dari 1 gambar ke 7 angka

Model menerima tensor `x` berukuran `[B, 3, 224, 224]` (RGB, sudah dinormalisasi
ImageNet) dan opsional peta `lam` `[B, 1, H, W]`.

### 3.1 Rumus ukuran konvolusi (yang dipakai berulang)

Untuk konvolusi/pooling dengan ukuran kernel `k`, stride `s`, padding `p`, lebar keluaran:

$$
W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2p - k}{s} \right\rfloor + 1
$$

**Bukti ukuran di Stem** (input 224):

- `Conv2d(k=7, s=2, p=3)`:
  $\left\lfloor \frac{224 + 6 - 7}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = 112$.
- `MaxPool2d(k=3, s=2, p=1)`:
  $\left\lfloor \frac{112 + 2 - 3}{2} \right\rfloor + 1 = 55 + 1 = 56$.

Jadi stem: `224×224 → 56×56`, channel `3 → 64`. ✅

### 3.2 Tabel dimensi penuh (sesuai `model.py:forward`)

| Tahap | Operasi | Channel | Spasial |
|-------|---------|--------:|--------:|
| input | — | 3 | 224×224 |
| `stem` | Conv7×7 s2 → BN → ReLU → MaxPool3×3 s2 | 64 | 56×56 |
| `block1` | 6 DenseLayer → SE1 → SpatialAttnBR | 256 | 56×56 |
| `trans1` | BN-ReLU-Conv1×1 → AvgPool2 | 128 | 28×28 |
| `block2` | 12 DenseLayer → SE2 → SpatialAttnBR | 512 | 28×28 |
| `trans2` | BN-ReLU-Conv1×1 → AvgPool2 | 256 | 14×14 |
| `block3` | 24 DenseLayer → SE3 → SpatialAttnBR | 1024 | 14×14 |
| `trans3` | BN-ReLU-Conv1×1 → AvgPool2 | 512 | 7×7 |
| `block4` | 16 DenseLayer → SE4 → SpatialAttnBR | 1024 | 7×7 |
| `norm5` | BN → ReLU | 1024 | 7×7 |
| `top` | GAP → FC1024→32 → BN → ReLU → Dropout → FC32→7 | 7 | 1×1 |

**Bukti pertumbuhan channel tiap block** (tiap DenseLayer menambah tepat `growth_rate=32`):

$$
C_{\text{out}} = C_{\text{in}} + L \cdot k
$$

- Block1: $64 + 6\cdot32 = 256$
- Block2: $128 + 12\cdot32 = 512$
- Block3: $256 + 24\cdot32 = 1024$
- Block4: $512 + 16\cdot32 = 1024$ ✅

Susunan `[6, 12, 24, 16]` ini **persis DenseNet-121**, itulah sebabnya bobot ImageNet
bisa langsung dipindahkan (lihat [bagian 6](#6-mengapa-training-nya-bertahap)).

> **Urutan baku tiap block:** `DenseBlock → SEBlock (channel) → SpatialAttnBR (spasial+λ) → Transition`.

---

## 4. Jalur neuron yang spesifik (telusur langkah demi langkah)

Di sini kita "ikuti" aliran angka sampai ke level neuron. Fokusnya **satu DenseLayer**,
karena di situlah seluruh inti modifikasi berkumpul.

### 4.1 Apa yang diterima satu DenseLayer

Di dalam `DenseBlock`, fitur disimpan dalam daftar `feats = [x0, h1, h2, …]` dan setiap
layer menerima **gabungan (concat)** semua fitur sebelumnya:

```python
feats = [x]
for layer in self.layers:
    feats.append(layer(feats))   # layer melihat SEMUA fitur sebelumnya
return torch.cat(feats, 1)
```

Jadi untuk layer ke-$l$ (0-indeks) di sebuah block dengan masukan awal $C_{\text{in}}$:

$$
\text{channel masuk ke layer } l = C_{\text{in}} + l \cdot k
$$

Contoh di Block1: layer ke-0 melihat 64 channel, layer ke-1 melihat 96, …, layer ke-5
melihat $64 + 5\cdot32 = 224$ channel. Inilah inti DenseNet — **tidak ada informasi awal
yang hilang**, karena selalu ikut di-concat (Huang et al., CVPR 2017).

### 4.2 Lima stasiun di dalam satu DenseLayer

Kode `DenseLayer.forward` (disederhanakan):

```python
x = torch.cat(x, 1)          # (1) gabung semua fitur sebelumnya
h = self.se(self.h(x))       # (2) bottleneck H_l, lalu (3) SE pada bottleneck
out = self.conv(h)           # (4) SCConv → 24 channel
if self.pq is not None:      # (5) cabang ND
    P, Q = F.softplus(self.pq(h)).chunk(2, 1)
    out = torch.cat([out, (P - Q) / (P + Q + self.eps)], 1)  # 24 + 8 = 32
return out
```

Mari telusuri tiap stasiun pada satu "neuron keluaran":

**Stasiun 1 — Bottleneck `H_l`** (`self.h`): `BN → ReLU → Conv1×1(C→128) → BN → ReLU`.
Konvolusi 1×1 di sini berfungsi sebagai **kombinasi linear antar-channel** di tiap piksel.
Untuk satu piksel dengan vektor masukan $\mathbf{x} \in \mathbb{R}^{C}$, neuron bottleneck
ke-$j$ menghitung:

$$
h_j = \text{ReLU}\!\Big( \sum_{c=1}^{C} W^{(1)}_{jc}\, \tilde{x}_c \Big), \quad j = 1,\dots,128
$$

dengan $\tilde{x}$ = hasil BN-ReLU. Jumlah channel dipadatkan jadi tetap **128** ($=4k$)
tak peduli berapa besar $C$ — ini menjaga biaya komputasi konstan (ide bottleneck
"DenseNet-BC", Huang et al. 2017).

**Stasiun 2 — SE pada bottleneck** (`self.se`, pada 128 channel): lihat
[bagian 5.3](#53-seblock--atensi-channel). Dia menghasilkan 128 bilangan antara 0–1 dan
**mengalikan** tiap channel bottleneck dengan bobotnya. Neuron bottleneck yang "tidak
berguna untuk gambar ini" diredam mendekati 0, yang penting dibiarkan ≈1.

**Stasiun 3 — SCConv** (`self.conv`, 128 → 24): konvolusi 3×3 **ter-normalisasi cosine**
([bagian 5.4](#54-scconv--konvolusi-cosine-ter-normalisasi)). Satu neuron keluaran di
posisi $(i,j)$ channel $o$ menilai **seberapa mirip pola** patch 3×3 di sekitarnya dengan
kernel $o$ — bukan seberapa kuat sinyalnya. Keluaran 24 channel.

**Stasiun 4 — Cabang ND** (`self.pq`, 128 → 16 → 8): konvolusi 3×3 lain menghasilkan 16
channel, dibelah jadi $P$ dan $Q$ (8 + 8). Dilewatkan `softplus` agar positif, lalu
dihitung rasio ter-normalisasi $(P-Q)/(P+Q+\varepsilon)$ → **8 channel**. Ini "versi
belajar" dari ide λ B/R ([bagian 5.6](#56-channel-nd-pqpq)).

**Stasiun 5 — Gabung keluaran:** `concat[SCConv(24), ND(8)] = 32 = growth_rate`. Inilah 32
channel baru yang akan di-concat ke `feats` dan dilihat semua layer berikutnya.

> **Bukti akuntansi channel:** `growth_rate − nd = 32 − 8 = 24` dari SCConv, ditambah
> `nd = 8` dari cabang ND = **32**. Konsisten dengan `config.ND_CH = 8`.

### 4.3 Setelah block: dua manajer atensi

Keluaran block (mis. 256 channel di Block1) melewati:

1. **SE block-level** (`se1`, pada 256 channel) — mengatur **channel mana** yang penting.
2. **SpatialAttnBR** (`sa`) — mengatur **lokasi mana** yang penting, dibantu peta fisika λ.

Baru kemudian masuk Transition (kecilkan channel & spasial).

### 4.4 Akhir: dari 1024 peta fitur ke 7 angka

Di `TopBlock`:

```
GAP: [B,1024,7,7] → [B,1024,1,1]   (rata-rata tiap peta 7×7 jadi 1 angka)
Flatten → [B,1024]
Linear(1024→32) → BN → ReLU → Dropout(0.5)
Linear(32→7) → logits [B,7]
```

- **GAP (Global Average Pooling)** mengubah tiap peta fitur 7×7 menjadi **satu angka**
  "seberapa kuat fitur ini muncul di seluruh gambar": $z_c = \frac{1}{49}\sum_{i,j} a_{c,i,j}$.
- **Bottleneck 32** sengaja sempit → memaksa model meringkas, berperan sebagai
  **regularisasi** (mengurangi overfit).
- Keluaran adalah **logit mentah** (belum softmax). Softmax baru dihitung di dalam loss.

Logit ke-7 kelas: `1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus,
6_cumulonimbus, 7_mixed`.

---

## 5. Bukti matematika tiap modifikasi

### 5.1 Konvolusi & Batch Normalization (dasar)

**Konvolusi 2D.** Untuk kernel $W$ berukuran $k\times k$, keluaran di posisi $(i,j)$
channel $o$:

$$
y_{o,i,j} = b_o + \sum_{c=1}^{C_{\text{in}}} \sum_{u=1}^{k}\sum_{v=1}^{k}
W_{o,c,u,v}\, x_{c,\,i+u,\,j+v}
$$

Ini sekadar **jumlah berbobot** patch lokal — "neuron" di CNN hanya melihat jendela kecil
(receptive field), bukan seluruh gambar.

**Batch Normalization** (Ioffe & Szegedy, ICML 2015). Untuk aktivasi $x$ pada satu channel,
dengan rata-rata batch $\mu_B$ dan varians $\sigma_B^2$:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta
$$

$\gamma, \beta$ dipelajari. BN membuat distribusi aktivasi stabil antar-batch → training
lebih cepat & stabil. **Detail penting di proyek ini:** karena SCConv mengubah skala
aktivasi, BN tidak dibekukan seluruhnya saat transfer learning — lihat
[bagian 6](#6-mengapa-training-nya-bertahap).

### 5.2 Koneksi dense (mengapa concat?)

Pada ResNet, blok menambah: $x_{l} = x_{l-1} + H(x_{l-1})$.
Pada **DenseNet** (Huang et al., 2017), blok **meng-concat**:

$$
x_{l} = H_l\big(\,[\,x_0, x_1, \dots, x_{l-1}\,]\,\big)
$$

Keuntungan matematisnya: setiap layer punya **akses langsung ke gradien loss** melalui
jalur pendek ke semua layer sebelumnya (mengurangi *vanishing gradient*), dan **fitur
dipakai ulang** (feature reuse) sehingga jumlah parameter lebih hemat untuk akurasi yang
sama. Itu sebabnya DenseNet-121 (±8 juta parameter) bisa menyaingi model yang jauh lebih
besar.

### 5.3 SEBlock — atensi channel

**Ide (Hu et al., CVPR 2018):** tidak semua channel sama penting; pelajari bobot per-channel.

Dua langkah:

1. **Squeeze** (ringkas tiap channel jadi 1 angka via GAP):
   $$ z_c = \frac{1}{H\cdot W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_{c,i,j} $$
2. **Excitation** (pelajari bobot 0–1 tiap channel):
   $$ s = \sigma\big(W_2\,\delta(W_1 z)\big), \qquad \tilde{x}_c = s_c \cdot x_c $$
   dengan $\delta$ = ReLU, $\sigma$ = sigmoid, $W_1 \in \mathbb{R}^{(C/r)\times C}$,
   $W_2 \in \mathbb{R}^{C\times(C/r)}$, rasio reduksi $r = 16$ (`SE_REDUCTION`).

**Bukti "inisialisasi ≈ identitas".** Di kode, bobot konv kedua di-nol-kan dan bias diisi
`SE_INIT_BIAS = 3.0`:

```python
nn.init.zeros_(self.excitation[2].weight)
nn.init.constant_(self.excitation[2].bias, init_bias)   # = 3.0
```

Maka di awal, sebelum belajar, apa pun masukannya:

$$
s_c = \sigma(0 \cdot \delta(W_1 z) + 3.0) = \sigma(3.0) = \frac{1}{1+e^{-3}} = \frac{1}{1.0498} \approx 0.9526
$$

Jadi tiap channel dikalikan ≈ 0.95 → **SE nyaris tidak mengubah apa pun di awal**. Ini
penting agar bobot ImageNet tidak langsung "dirusak" di epoch pertama transfer learning;
SE belajar menyimpang dari identitas secara bertahap.

### 5.4 SCConv — konvolusi cosine ter-normalisasi

**Masalah yang diselesaikan:** konvolusi biasa menghasilkan *dot product* yang **ikut
membesar kalau intensitas piksel besar**. Foto langit yang terang vs. redup bisa memberi
respons sangat berbeda walau polanya sama. **Cosine Normalization** (Luo et al., 2017/ICANN
2018) mengganti dot product dengan **cosine similarity** sehingga respons bergantung pada
**sudut/pola**, bukan besar sinyal.

Rumus di `conv.SCConv.forward`:

$$
\text{dot} = W * x, \qquad
\cos = \text{clip}\!\left(\frac{\text{dot}}{(\lVert x_{\text{patch}}\rVert + q)\,(\lVert W\rVert + \varepsilon)},\,-1,\,1\right)
$$

$$
y = \text{sign}(\text{dot}) \cdot \big(|\cos| + \varepsilon\big)^{\tau}
$$

dengan:
- $\lVert x_{\text{patch}}\rVert$ = norma L2 patch input (dihitung via `avg_pool2d` dari
  jumlah kuadrat lintas-channel),
- $\lVert W\rVert$ = norma kernel per filter keluaran,
- $q > 0$ (skalar, dari `log_q`) = penstabil pembagi,
- $\tau > 0$ (per-channel, dari `log_tau`) = "ketajaman" respons — dipelajari.
- Keduanya dijamin positif karena disimpan sebagai log lalu di-`exp`.

**Bukti nilai cosine terbatas $[-1,1]$.** Menurut ketaksamaan Cauchy–Schwarz,
$|\langle x, W\rangle| \le \lVert x\rVert\,\lVert W\rVert$, sehingga rasio
$\text{dot}/(\lVert x\rVert \lVert W\rVert) \in [-1,1]$. Penambahan $q,\varepsilon$ hanya
mengecilkan sedikit (regularisasi) dan `clamp(-1,1)` menjaga batas.

**Contoh angka (membuktikan invariansi skala).** Ambil patch disederhanakan jadi vektor
$x = [3, 4]$ (maka $\lVert x\rVert = 5$) dan kernel $W = [0.6, 0.8]$ (maka $\lVert W\rVert = 1$),
dengan $q = 0.01$, $\tau = 1$:

$$
\text{dot} = 3(0.6) + 4(0.8) = 5.0,\qquad
\cos = \frac{5.0}{(5 + 0.01)(1)} = 0.998,\qquad y = (0.998)^1 \approx 0.998
$$

Sekarang **gandakan intensitas** input: $x' = [6, 8]$ ($\lVert x'\rVert = 10$):

$$
\text{dot}' = 10,\qquad \cos' = \frac{10}{(10+0.01)(1)} = 0.999,\qquad y' \approx 0.999
$$

Walau sinyal **dua kali lebih kuat**, keluaran nyaris sama (0.998 vs 0.999) — inilah
**invariansi terhadap skala intensitas** yang berguna untuk foto langit dengan pencahayaan
berbeda-beda. (Konvolusi biasa akan memberi $y=5$ vs $y'=10$ — dua kali lipat.)

**Peran $\tau$.** Karena $|\cos|\le 1$, memangkatkan dengan $\tau$ mengatur ketajaman:
$\tau$ besar menekan respons yang "setengah mirip" (mendekati 0), $\tau$ kecil melembutkan.
Dipelajari per-channel agar tiap detektor pola punya ketajaman sendiri.

**Mengapa dipaksa fp32?** `with torch.autocast(..., enabled=False)`. Operasi norma &
pembagian rawan *overflow/underflow* di presisi setengah (fp16) saat AMP aktif; fp32
menjaganya stabil.

> **Catatan nama:** "SCConv" di proyek ini merujuk pada **konvolusi ter-normalisasi
> cosine seperti yang diimplementasikan di `conv.py`** (akar idenya Cosine Normalization,
> Luo et al.). Ini **berbeda** dari makalah lain yang kebetulan juga bernama "SCConv"
> (mis. *Spatial and Channel reconstruction Convolution*, CVPR 2023). Yang berlaku adalah
> definisi di kode.

### 5.5 SpatialAttnBR + peta λ B/R (sadar-fisika)

**Bagian spasial (akar: CBAM, Woo et al., ECCV 2018).** Ringkas tiap posisi piksel lintas
channel dengan **rata-rata** dan **maksimum**, lalu satu konvolusi menghasilkan satu peta
perhatian:

$$
m = \big[\,\text{mean}_c(x),\ \max_c(x)\,\big], \qquad a = \sigma\!\big(\text{Conv}_{7\times7}(m)\big), \qquad \tilde{x} = x \odot a
$$

$a \in (0,1)$ per posisi → "seberapa penting lokasi ini". Diinisialisasi ≈ identitas lagi
(bias 3.0 → $a \approx 0.953$), sama logikanya dengan SE.

**Tambahan sadar-fisika (λ B/R).** `SpatialAttnBR` menambah **channel ketiga**: peta λ rasio
Biru/Merah, sehingga konvolusinya menerima **3 channel**:

$$
m = \big[\,\text{mean}_c(x),\ \max_c(x),\ \lambda\,\big], \qquad a = \sigma\!\big(\text{Conv}_{7\times7}(m)\big)
$$

Peta λ dihitung dari citra RGB $[0,1]$ **sebelum** normalisasi (di `spectral.lambda_relatif`):

$$
\lambda_{\text{raw}} = \frac{B - R}{B + R + \varepsilon}
$$

**Mengapa fitur ini masuk akal secara fisika?** Hamburan Rayleigh membuat langit cerah
tampak **biru** (B > R), sedangkan awan memantulkan cahaya secara merata sehingga tampak
**putih/abu** (B ≈ R). Rasio B/R (atau R/B) adalah fitur klasik untuk deteksi awan dari
kamera *all-sky* (Long et al., 2006; Heinle et al., 2010).

**Contoh angka:**

| Piksel | R | B | $\lambda_{\text{raw}} = (B-R)/(B+R)$ | Tafsiran |
|--------|---|---|---|---|
| Langit biru | 0.30 | 0.60 | $0.30/0.90 = +0.33$ | λ > 0 → langit |
| Awan putih | 0.50 | 0.50 | $0.00/1.00 = 0.00$ | λ ≈ 0 → awan |
| Permukaan kemerahan | 0.60 | 0.30 | $-0.30/0.90 = -0.33$ | λ < 0 |

Lalu λ dibuat **relatif** (kurangi median per-gambar) dan di-*soft-clamp*:

$$
\lambda = 0.5\,\tanh\!\Big(\frac{\lambda_{\text{raw}} - \text{median}}{0.5}\Big)\cdot \text{valid}
$$

- `valid` membuang piksel **gelap** (<0.10, noise) dan **silau** (>0.97, matahari).
- `tanh` menjamin $|\lambda| < 0.5$ (lihat: $|\tanh(\cdot)|<1$, dikali 0.5).
- Kurangi median → λ menyatakan "lebih biru / lebih putih **dibanding rata-rata langit di
  foto ini**", bukan nilai absolut → tahan terhadap perbedaan kamera/cuaca.

Saat masuk model, λ di-`interpolate` ke ukuran peta fitur dan dikali `LAM_SCALE = 10.0`
agar sinyalnya sebanding dengan aktivasi lain sebelum masuk sigmoid.

**Fallback penting:** jika `lam=None`, λ diisi nol → `SpatialAttnBR` berperilaku **persis
seperti CBAM biasa**. Jadi fitur fisika bersifat *tambahan*, bukan syarat.

### 5.6 Channel ND (P−Q)/(P+Q)

Ini **generalisasi λ B/R yang dipelajari**. Alih-alih memakai channel Merah/Biru mentah,
model mempelajari dua peta fitur $P, Q$ (lewat `self.pq`), dijadikan positif dengan
`softplus` $\big(\text{softplus}(z)=\ln(1+e^{z})>0\big)$, lalu:

$$
\text{ND} = \frac{P - Q}{P + Q + \varepsilon}
$$

**Bukti terbatas $(-1, 1)$.** Karena $P, Q \ge 0$, maka $|P - Q| \le P + Q$, sehingga

$$
\left|\frac{P-Q}{P+Q+\varepsilon}\right| < 1
$$

**Contoh:** $P=3, Q=1 \Rightarrow \tfrac{2}{4}=0.5$; $P=1, Q=3 \Rightarrow -0.5$;
$P=Q \Rightarrow 0$. Bentuk rasio ini juga **sebagian tahan skala**: mengali $P,Q$ dengan
konstanta yang sama (mengabaikan $\varepsilon$) tidak mengubah nilainya — persis sifat yang
diinginkan dari "indeks" seperti NDVI di penginderaan jauh. 8 channel ND ini di-concat ke
24 channel SCConv → total 32.

**Mulai dari "nol" yang aman.** `pretrained.nolkan_input_nd` menol-kan bobot yang *membaca*
channel ND di layer berikutnya (dan di transition). Jadi di awal training, channel ND **tidak
berpengaruh apa-apa** → perilaku awal = DenseNet ImageNet murni, lalu kontribusinya tumbuh
saat dilatih.

### 5.7 Focal Loss

**Masalah:** dataset awan tidak seimbang (mis. `7_mixed` jarang). Dengan Cross-Entropy biasa,
ribuan contoh "mudah" mendominasi gradien dan contoh sulit terabaikan.

**Cross-Entropy** untuk kelas benar dengan probabilitas $p_t$ (hasil softmax):

$$
\text{CE} = -\ln(p_t)
$$

**Focal Loss** (Lin et al., ICCV 2017) menambah faktor peredam $(1-p_t)^\gamma$:

$$
\text{FL} = -\alpha\,(1 - p_t)^{\gamma}\,\ln(p_t), \qquad \gamma = 2.0,\ \alpha = 1.0
$$

Di kode, $p_t$ diperoleh elegan sebagai `pt = exp(-ce_loss)` (karena $e^{-(-\ln p_t)} = p_t$):

```python
ce_loss = self.ce_loss(logits, target)   # = -ln(pt)
pt = torch.exp(-ce_loss)                  # = pt
focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
```

**Bukti kuantitatif bahwa Focal "memindahkan fokus".** Bandingkan contoh **mudah**
($p_t = 0.9$) dan **sulit** ($p_t = 0.3$), dengan $\gamma = 2$:

| | $p_t$ | CE $=-\ln p_t$ | Faktor $(1-p_t)^2$ | FL |
|---|---|---|---|---|
| Mudah | 0.90 | 0.105 | $0.1^2 = 0.010$ | **0.00105** |
| Sulit | 0.30 | 1.204 | $0.7^2 = 0.490$ | **0.590** |

- Dengan **CE biasa**, rasio perhatian sulit:mudah = $1.204 / 0.105 \approx 11\times$.
- Dengan **Focal**, rasio menjadi $0.590 / 0.00105 \approx 562\times$.

Artinya, Focal memperbesar kepedulian pada contoh sulit **sekitar 50×** lebih besar
daripada CE ($562/11 \approx 51$, cocok dengan faktor $(0.49/0.01)=49$). Contoh yang sudah
hampir pasti benar praktis **diabaikan**, sehingga gradien dialokasikan ke kelas yang masih
sering salah.

> Di proyek ini ketidakseimbangan juga ditangani lewat `WeightedRandomSampler`
> (`IMBALANCE="sampler"`) di `data.py`, jadi Focal Loss dan oversampling bekerja bersama.

### 5.8 Inisialisasi LeCun-uniform

Hanya untuk **lapisan baru** (Top Block & cabang ND). Rumus di `losses.lecun_uniform_`:

$$
W \sim \mathcal{U}(-L, L), \qquad L = \sqrt{\frac{3}{\text{fan\_in}}}
$$

**Bukti bahwa ini menjaga varians $\approx 1/\text{fan\_in}$.** Varians distribusi uniform
$\mathcal{U}(-L, L)$ adalah:

$$
\text{Var} = \frac{(2L)^2}{12} = \frac{L^2}{3} = \frac{1}{3}\cdot\frac{3}{\text{fan\_in}} = \frac{1}{\text{fan\_in}}
$$

Target "varians = $1/\text{fan\_in}$" (LeCun et al., 1998) menjaga **varians sinyal
tetap stabil** saat melewati layer (tidak meledak/menghilang), sehingga training lapisan
baru lancar sejak awal. Bias di-nol-kan; BatchNorm di-init $\gamma=1, \beta=0$ (identitas).

**Mengapa HANYA lapisan baru?** Jika `init_new_layers` di-`apply` ke seluruh model, ia akan
**menimpa bobot ImageNet** backbone dan merusak transfer learning. SE & SpatialAttnBR juga
sengaja **tidak** di-init ulang karena sudah di-set ≈ identitas (bobot 0 + bias 3.0). Lihat
komentar di `model.CloudDenseNet._init_lapisan_baru`.

---

## 6. Mengapa training-nya bertahap?

Backbone dimulai dari bobot **DenseNet-121 ImageNet** (`pretrained.muat_imagenet`). Karena
susunan block `[6,12,24,16]` identik, bobot bisa dipetakan langsung (ada penyesuaian kecil:
filter 32→24 dipotong karena 8 channel dialihkan ke jalur ND).

Jika semua layer langsung dilatih dengan LR besar, bobot ImageNet yang bagus bisa rusak
("catastrophic forgetting"). Maka dipakai **6 fase** (`config.PHASES`, dieksekusi
`finetune.py`):

| Fase (≤epoch) | LR | Status | Yang dilatih |
|--------------:|----|--------|--------------|
| 10 | 1e-3 | `frozen` | Top Block + lapisan baru (SE/SA/`log_q`/`log_tau`) |
| 20 | 1e-4 | `frozen` | idem |
| 30 | 1e-5 | `partial` | + Block3-4, trans3, norm5 |
| 40 | 5e-5 | `partial` | idem |
| 60 | 1e-5 | `full` | seluruh backbone |
| 80 | 1e-6 | `full` | full defrost end-to-end |

Logika: **latih kepala dulu** (fitur ImageNet dibekukan), lalu **buka bertahap** dari layer
dalam (yang lebih spesifik tugas) ke seluruh jaringan, dengan **LR makin kecil** agar
perubahan makin halus. Ini pola *fine-tuning* standar (Yosinski et al., NeurIPS 2014).

Dua detail yang mudah terlewat:
- **BatchNorm tidak dibekukan penuh** (`model.bn_beku` terkait `USE_SCCONV`): karena SCConv
  mengubah distribusi aktivasi, statistik BN perlu beradaptasi. `set_bn_beku_eval` hanya
  membekukan *running stats* BN ImageNet pada layer yang memang beku.
- **Jalur `.pq.` dilewati saat beku**: pembacanya (`h.2`) sudah bernilai 0, jadi melatihnya
  di fase awal sia-sia.

Optimasi: **AdamW** (`WEIGHT_DECAY=1e-2`), **gradient clipping** norma 5.0 (`GRAD_CLIP`),
dan ada pengaman NaN di `engine.py`.

---

## 7. Tabel referensi lengkap

> Semua di bawah ini adalah publikasi nyata & relevan dengan komponen yang dipakai. Untuk
> **sitasi dataset GCD**, lihat [catatan di bagian 8](#8-catatan-teknis--kejujuran).

1. **DenseNet** — G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger. *Densely Connected
   Convolutional Networks.* CVPR 2017. arXiv:1608.06993.
2. **Squeeze-and-Excitation** — J. Hu, L. Shen, G. Sun. *Squeeze-and-Excitation Networks.*
   CVPR 2018; versi perluasan di IEEE TPAMI 2019. arXiv:1709.01507.
3. **CBAM** — S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. *CBAM: Convolutional Block Attention
   Module.* ECCV 2018. arXiv:1807.06521.
4. **Focal Loss** — T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for
   Dense Object Detection.* ICCV 2017. arXiv:1708.02002.
5. **Batch Normalization** — S. Ioffe, C. Szegedy. *Batch Normalization: Accelerating Deep
   Network Training by Reducing Internal Covariate Shift.* ICML 2015. arXiv:1502.03167.
6. **Cosine Normalization** — C. Luo, J. Zhan, X. Wang, Q. Yang (dkk.). *Cosine
   Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks.*
   arXiv:1702.05870, 2017 (dipresentasikan di ICANN 2018).
7. **Inisialisasi skala-varians** — Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller.
   *Efficient BackProp.* Dalam *Neural Networks: Tricks of the Trade*, 1998.
8. **B/R ratio untuk deteksi awan** —
   (a) C. N. Long, J. M. Sabburg, J. Calbó, D. Pagès. *Retrieving Cloud Characteristics from
   Ground-Based Daytime Color All-Sky Images.* J. Atmos. Oceanic Technol., 2006.
   (b) A. Heinle, A. Macke, A. Srivastav. *Automatic Cloud Classification of Whole Sky
   Images.* Atmos. Meas. Tech., 2010.
9. **ImageNet** — J. Deng dkk. *ImageNet: A Large-Scale Hierarchical Image Database.*
   CVPR 2009.
10. **Transferabilitas fitur** — J. Yosinski, J. Clune, Y. Bengio, H. Lipson. *How
    transferable are features in deep neural networks?* NeurIPS 2014. arXiv:1411.1792.

---

## 8. Catatan teknis & kejujuran

- **Verifikasi angka.** Ukuran spasial (56→28→14→7) dan channel (256/512/1024) di dokumen
  ini dihitung dari rumus konvolusi dan dicocokkan dengan komentar di `model.py:forward`.
  Catatan: beberapa docstring di `blocks.py` menyebut spasial "64/32/16" — itu **sisa dari
  ukuran input lain** dan tidak sesuai dengan input 224 yang dipakai sekarang. Yang benar
  untuk input 224 adalah **56/28/14/7** (sesuai `model.py`). Ini perbedaan komentar saja,
  bukan bug kode; sesuai permintaan, **kode tidak diubah**.
- **Dataset GCD.** CLAUDE.md menyebut "dataset GCD" dengan 7 kelas awan. Saya **tidak**
  menautkan satu sitasi akademik spesifik untuk GCD di sini karena ada beberapa dataset
  langit berbeda dengan singkatan mirip; sebaiknya sitasi dataset dikonfirmasi langsung
  oleh penulis dari sumber aslinya agar tidak salah atribusi.
- **"SCConv".** Seperti ditegaskan di [5.4](#54-scconv--konvolusi-cosine-ter-normalisasi),
  nama ini di proyek merujuk implementasi cosine-normalized conv di `conv.py`, bukan makalah
  lain yang kebetulan bernama sama.
- **Angka contoh** di bagian 5 adalah ilustrasi untuk membuktikan *sifat* (invariansi skala,
  peredaman focal, batas rasio), bukan nilai sebenarnya dari training.

---

*Dokumen ini murni penjelasan — tidak ada kode yang diubah. Perbarui bila modul di
`src/clouddensenet/` berubah. Lihat juga: [`ARSITEKTUR_CLOUDDENSENET.md`](ARSITEKTUR_CLOUDDENSENET.md),
[`EDA_AUGMENTASI.md`](EDA_AUGMENTASI.md).*
