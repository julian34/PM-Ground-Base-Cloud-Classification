# Arsitektur CloudDenseNet

Dokumentasi arsitektur model **CloudDenseNet** — DenseNet-121 yang dimodifikasi untuk
klasifikasi citra langit ke **7 kelas awan** (dataset GCD). Dokumen ini menjelaskan
setiap komponen, alur forward, dimensi tensor, dan modifikasi terhadap DenseNet-121 standar.

> Sumber kebenaran: `src/clouddensenet/{model,blocks,attention,conv,spectral,losses,
> pretrained,finetune}.py` dan `src/clouddensenet/config.py`.
> Hyperparameter default yang dipakai dokumen ini: `growth_rate=32`, `ND_CH=8`,
> `USE_SE=True`, `USE_SCCONV=True`, `USE_BR=True`, `NUM_CLASSES=7`.

---

## 1. Gambaran umum

CloudDenseNet = **backbone bergaya DenseNet-121** (stem + 4 dense block + 3 transisi)
yang diperkaya dengan beberapa modifikasi:

| Modifikasi | Modul | Peran |
|------------|-------|-------|
| **SEBlock** | `attention.SEBlock` | Atensi channel (squeeze-and-excitation) — di tiap bottleneck DenseLayer **dan** di akhir tiap block. |
| **SCConv** | `conv.SCConv` | Konvolusi ter-normalisasi cosine, dipaksa fp32 agar stabil di AMP, menggantikan conv 3×3 di DenseLayer. |
| **SpatialAttnBR** | `attention.SpatialAttnBR` | Atensi spasial + 1 channel λ rasio Biru/Merah (sadar-fisika). |
| **Channel ND** | jalur `pq` di `DenseLayer` | `ND_CH=8` channel ekstra per layer dari rasio ter-normalisasi `(P−Q)/(P+Q)`. |
| **Top Block ringkas** | `blocks.TopBlock` | GAP → FC(1024→32) → BN → ReLU → Dropout → FC(32→7). |
| **Focal Loss** | `losses.FocalLoss` | γ=2.0, menangani ketidakseimbangan kelas. |
| **Fine-tuning bertahap** | `finetune.py` | Transfer learning ImageNet 6 fase (beku → buka 3-4 → full). |

---

## 2. Alur forward (dimensi tensor)

Input: `x` = `[B, 3, 224, 224]` (RGB ternormalisasi ImageNet) dan opsional `lam`
= `[B, 1, H, W]` (peta λ B/R dari `siapkan_batch`).

```
x [B,3,224,224]
 │
 ▼  Stem: Conv7×7(s2) → BN → ReLU → MaxPool3×3(s2)
s  [B, 64, 56, 56]
 │
 ▼  Block1 (6 DenseLayer)  → SE1 → SpatialAttnBR(·,λ)
b1 [B, 256, 56, 56]
 │  Trans1: BN-ReLU-Conv1×1(256→128) → AvgPool2×2
t1 [B, 128, 28, 28]
 │
 ▼  Block2 (12 DenseLayer) → SE2 → SpatialAttnBR(·,λ)
b2 [B, 512, 28, 28]
 │  Trans2: BN-ReLU-Conv1×1(512→256) → AvgPool2×2
t2 [B, 256, 14, 14]
 │
 ▼  Block3 (24 DenseLayer) → SE3 → SpatialAttnBR(·,λ)
b3 [B, 1024, 14, 14]
 │  Trans3: BN-ReLU-Conv1×1(1024→512) → AvgPool2×2
t3 [B, 512, 7, 7]
 │
 ▼  Block4 (16 DenseLayer) → SE4 → SpatialAttnBR(·,λ)
b4 [B, 1024, 7, 7]
 │  norm5 (BN) → ReLU
out [B, 1024, 7, 7]
 │
 ▼  Top Block: GAP → Flatten → FC(1024→32) → BN → ReLU → Dropout → FC(32→7)
logits [B, 7]
```

Urutan per-block konsisten: **DenseBlock → SEBlock (channel) → SpatialAttnBR (spasial+λ)**,
baru masuk Transition. `forward(..., return_all=True)` mengembalikan dict berisi semua
tensor antara (`stem, block1, trans1, …, out, logits`) untuk visualisasi/debug.

---

## 3. Komponen rinci

### 3.1 Stem

```
Conv2d(3→64, k=7, s=2, p=3, bias=False) → BatchNorm2d(64) → ReLU → MaxPool2d(k=3, s=2, p=1)
```
Mereduksi 224×224 → 56×56, menaikkan channel 3 → 64. Identik DenseNet-121.

### 3.2 DenseLayer (unit penyusun tiap block)

Setiap DenseLayer menerima **concat semua fitur sebelumnya** (koneksi dense), lalu:

```
x (concat)
 → BN → ReLU → Conv1×1 (in → 4·k = 128)   # bottleneck H_l
 → BN → ReLU
 → SE(128)                                 # atensi channel di bottleneck
 → SCConv 3×3 (128 → k−nd = 24)            # jalur utama
 ├─ (jika nd>0) Conv3×3 (128 → 2·nd=16) → softplus → P,Q
 │               → (P−Q)/(P+Q+eps)  (nd=8 channel)
 → concat[ SCConv(24), ND(8) ] = 32 = growth_rate
```

- **Output tiap layer = `growth_rate` (32) channel**, sebagian dari SCConv (24) dan
  sebagian dari jalur ND rasio ter-normalisasi (8).
- Channel ND membawa informasi rasio dua peta fitur `(P−Q)/(P+Q)` — mirip semangat λ B/R
  tapi dipelajari. Bobot pembacanya **dinolkan saat init** (`nolkan_input_nd`) agar awal
  training setara ImageNet murni.

**Pertumbuhan channel per block** (mulai dari `in_channels`, +32 per layer):

| Block | #Layer | Channel masuk → keluar | Spasial |
|-------|-------:|------------------------|---------|
| Block1 | 6  | 64 → 64 + 6·32 = **256**   | 56×56 |
| Block2 | 12 | 128 → 128 + 12·32 = **512** | 28×28 |
| Block3 | 24 | 256 → 256 + 24·32 = **1024** | 14×14 |
| Block4 | 16 | 512 → 512 + 16·32 = **1024** | 7×7 |

> Konfigurasi layer (6/12/24/16) = **DenseNet-121** persis.

### 3.3 Transition Layer (1–3)

```
BN → ReLU → Conv1×1 (reduksi channel 50%) → AvgPool2×2 (spasial ÷2)
```
| Transisi | Channel | Spasial |
|----------|---------|---------|
| Trans1 | 256 → 128 | 56 → 28 |
| Trans2 | 512 → 256 | 28 → 14 |
| Trans3 | 1024 → 512 | 14 → 7 |

### 3.4 SEBlock — atensi channel

```
GAP → Conv1×1(C→C/16) → ReLU → Conv1×1(C/16→C) → Sigmoid → skalakan x per-channel
```
- Rasio reduksi `r=16` (Hu et al., 2018).
- **Inisialisasi ≈ identitas:** bobot layer akhir di-nol-kan, bias = `SE_INIT_BIAS=3.0`
  → `sigmoid(3) ≈ 0.953`, jadi di awal SE hampir tidak mengubah fitur (stabil untuk
  transfer learning).
- Dipakai dua tempat: di dalam tiap DenseLayer (pada bottleneck 128) dan sebagai
  `se1..se4` di akhir tiap block (256/512/1024/1024).

### 3.5 SpatialAttnBR — atensi spasial sadar-fisika

```
m = concat[ mean(x) , max(x) , λ ]    # 3 channel
a = sigmoid( Conv_k×k(m) )            # k=7, padding reflect
out = x * a
```
- Turunan dari `SpatialAttn` biasa (yang hanya pakai mean+max → 2 channel); versi BR
  menambah **1 channel λ** sehingga conv punya 3 channel input.
- `λ` di-`interpolate` ke ukuran fitur dan diskalakan `LAM_SCALE=10.0`.
- **Fallback:** jika `lam=None`, λ diisi nol → berperilaku seperti SpatialAttn biasa.
- Init bias = 3.0 (sigmoid≈0.95) → awal ≈ identitas.

### 3.6 SCConv — konvolusi cosine ter-normalisasi

Menggantikan conv 3×3 di DenseLayer (`USE_SCCONV=True`). Menghitung konvolusi lalu
menormalkannya jadi cosine similarity antara patch input & kernel:

```
cos = dot / ((‖x_patch‖ + q) · (‖w‖ + eps)),  di-clamp [−1,1]
out = sign(dot) · (|cos| + eps)^τ
```
- `q` (skalar, `log_q`) dan `τ` (per-channel, `log_tau`) dipelajari (dijamin > 0 via `exp`).
- **Dipaksa fp32** (`autocast enabled=False`) agar stabil di bawah AMP.
- Padding `reflect`.

### 3.7 Peta λ B/R (`spectral.lambda_relatif`)

Dihitung di `siapkan_batch` pada citra **[0,1] yang belum dinormalisasi**:

```
λ_raw = (B − R) / (B + R + eps)              # > 0 biru/langit, < 0 putih/awan
λ = 0.5 · tanh( (λ_raw − median) / 0.5 ) · valid_mask    # soft-clamp |λ| < 0.5
```
- `valid_mask` membuang piksel gelap (<0.10) dan silau (>0.97).
- Median per-gambar dikurangi agar λ **relatif** terhadap kondisi tiap citra.
- Inilah fitur fisik yang dikonsumsi SpatialAttnBR.

### 3.8 Top Block (kepala klasifikasi)

```
GAP(1×1) → Flatten → Linear(1024→32) → BN1d(32) → ReLU → Dropout(0.5) → Linear(32→7)
```
Menghasilkan **logit** mentah `[B, 7]` (softmax dilakukan di loss). Bottleneck 32 yang
sempit berperan sebagai regularisasi.

---

## 4. Loss & inisialisasi

- **FocalLoss** (`α=1.0`, `γ=2.0`): `FL = α·(1−pₜ)^γ · CE`. Menekan kontribusi contoh
  mudah → fokus ke kelas sulit/minoritas (mis. `7_mixed`).
- **LeCun Uniform** (`init_new_layers`): hanya untuk **lapisan baru** (SE, cabang,
  Top Block) — jangan di-`apply` ke seluruh model agar bobot ImageNet tidak tertimpa.

---

## 5. Transfer learning & fine-tuning bertahap

Backbone di-inisialisasi dari **DenseNet-121 ImageNet** (`muat_imagenet`), lalu dilatih
lewat jadwal 6 fase (`config.PHASES`) dengan strategi pembekuan di `finetune.py`:

| Fase (epoch) | LR | Status | Yang dilatih |
|-------------:|----|--------|--------------|
| ≤10 | 1e-3 | `frozen` (`bekukan_imagenet`) | Top Block + lapisan baru (SE/SA/log_q/log_tau) |
| ≤20 | 1e-4 | `frozen` | idem (tahap 2) |
| ≤30 | 1e-5 | `partial` (`buka_blok_3_4`) | + Block3-4, trans3, norm5 |
| ≤40 | 5e-5 | `partial` | idem (unfreeze Dense Block 3-4) |
| ≤60 | 1e-5 | `full` (`buka_semua`) | seluruh backbone |
| ≤80 | 1e-6 | `full` | full defrost end-to-end |

Catatan penting:
- `bekukan_imagenet` **melewati jalur `.pq.`** saat beku (pembacanya `h.2` beku = 0).
- `model.bn_beku`: karena SCConv mengubah distribusi aktivasi, BatchNorm perlu adaptasi —
  `set_bn_beku_eval` menjaga BN ImageNet yang beku agar tidak meng-update running stats.
- `nolkan_input_nd`: bobot pembaca channel ND dinolkan → awal training = ImageNet murni.

Optimasi: **AdamW** (`WEIGHT_DECAY=1e-2`), **grad clip** norma 5.0.

---

## 6. Ringkasan modifikasi vs DenseNet-121 standar

| Aspek | DenseNet-121 | CloudDenseNet |
|-------|--------------|---------------|
| Conv 3×3 di DenseLayer | Conv standar | **SCConv** (cosine, fp32) |
| Atensi channel | — | **SEBlock** (bottleneck + akhir block) |
| Atensi spasial | — | **SpatialAttnBR** (+λ B/R sadar-fisika) |
| Channel ekstra | — | **ND** `(P−Q)/(P+Q)`, 8/layer |
| Kepala | FC(1024→1000) | **Top Block** ringkas (1024→32→7) |
| Loss | CrossEntropy | **Focal Loss** (γ=2.0) |
| Training | — | **fine-tuning ImageNet 6 fase** |

Konfigurasi kedalaman block (6/12/24/16), stem, dan transisi **tetap sama** dengan
DenseNet-121, sehingga bobot ImageNet bisa dipindahkan langsung.

---

## 7. Verifikasi cepat

Smoke test membangun model & menjalankan satu forward pass (tanpa dataset/GPU),
memverifikasi shape output `(N, 7)` dan key `return_all=True`, serta mencetak jumlah parameter:

```bash
pytest                         # tests/test_forward.py
python tests/test_forward.py   # cetak "forward pass sukses" + jumlah parameter
```

---

*Dibuat sebagai dokumentasi arsitektur. Perbarui bila modul di `src/clouddensenet/` berubah.*
