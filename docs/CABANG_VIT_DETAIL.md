# Penjelasan Detail Cabang ViT — Arsitektur Dua-Cabang CloudDenseNet

Dokumen ini menjelaskan **cabang Vision Transformer (ViT)** yang ditambahkan ke
CloudDenseNet pada branch `ViT-branch` (commit `bd03ce1`). Cabang ini mengubah model
dari **satu aliran (DenseNet murni)** menjadi **dua cabang paralel (DenseNet + ViT)**
yang difusikan tepat sebelum Top Block.

> **Sumber kebenaran tetap kode** di `src/clouddensenet/`:
> `vit_branch.py` (arsitektur ViT), `model.py` (integrasi & fusi), `blocks.py`
> (`TopBlock` dengan `extra_dim`), `config.py` (flag & hyperparameter), dan
> `tests/test_forward.py` (smoke test). Hyperparameter default dibahas di §4.
>
> **Catatan penting:** flag `USE_VIT = False` secara **default**. Artinya, tanpa
> mengubah apa pun, arsitektur, `state_dict`, dan checkpoint lama **tidak berubah
> sama sekali**. Cabang ViT bersifat *opt-in*.

---

## Daftar isi

1. [Ide besar: mengapa dua cabang?](#1-ide-besar-mengapa-dua-cabang)
2. [Di mana percabangan terjadi (titik fork)](#2-di-mana-percabangan-terjadi-titik-fork)
3. [Anatomi `ViTBranch` langkah demi langkah](#3-anatomi-vitbranch-langkah-demi-langkah)
4. [Hyperparameter cabang ViT (config.py)](#4-hyperparameter-cabang-vit-configpy)
5. [Fusi di Top Block (`extra_dim`)](#5-fusi-di-top-block-extra_dim)
6. [Aliran tensor & shape lengkap](#6-aliran-tensor--shape-lengkap)
7. [Training: bagaimana cabang ViT dilatih](#7-training-bagaimana-cabang-vit-dilatih)
8. [Kompatibilitas mundur & checkpoint](#8-kompatibilitas-mundur--checkpoint)
9. [Biaya parameter & cara mengaktifkan](#9-biaya-parameter--cara-mengaktifkan)
10. [Catatan teknis & kejujuran](#10-catatan-teknis--kejujuran)

---

## 1. Ide besar: mengapa dua cabang?

**CNN (DenseNet)** unggul menangkap pola **lokal** — tekstur, tepi, gradasi — lewat
konvolusi berjendela kecil yang bertumpuk. Tapi untuk menangkap **relasi jarak-jauh**
(misal: sebaran awan di seluruh bingkai, hubungan antar-wilayah langit yang
berjauhan), CNN harus menumpuk banyak lapis dulu sampai receptive field-nya cukup luas.

**Vision Transformer (ViT)** justru sebaliknya: lewat **self-attention**, setiap patch
bisa "melihat" semua patch lain **sejak lapis pertama**, sehingga relasi global
tertangkap langsung.

Gagasannya: **jalankan keduanya paralel**, lalu gabungkan. DenseNet memberi vektor
ringkasan kaya-fitur-lokal (1024-dim), ViT memberi vektor ringkasan kaya-konteks-global
(`VIT_DIM`-dim). Keduanya **di-concat** sebelum MLP klasifikasi, sehingga Top Block
memutuskan kelas berbekal **dua sudut pandang sekaligus**.

```
                         ┌─────────────── Cabang A: DenseNet ───────────────┐
  citra ──► stem (conv)  │  block1..block4 + SE + SpatialAttnBR + norm5     │ ─► out (B,1024,7,7)
            (B,64,56,56) │                                                   │
                  │      └───────────────────────────────────────────────────┘
                  │                                                              ╲
                  │      ┌─────────────── Cabang B: ViT ───────────────┐          ╲ (GAP → 1024)
                  └────► │  patch-embed + CLS + pos + TransformerEncoder │ ─► vec_vit (B,VIT_DIM)
                         └───────────────────────────────────────────────┘          │
                                                                                     ▼
                                                      Top Block: concat([1024, VIT_DIM]) ─► MLP ─► 7 logit
```

---

## 2. Di mana percabangan terjadi (titik fork)

Percabangan terjadi **setelah stem**, bukan dari citra mentah. "Stem" adalah
konvolusi pertama DenseNet: `conv7x7 stride2 → BN → ReLU → maxpool stride2`.

Di `model.py`:

```python
def forward(self, x, lam=None, return_all=False):
    s = self.maxpool(self.relu(self.bn(self.conv(x))))   # (B, 64, 56, 56)  ← STEM

    # --- Cabang A: backbone DenseNet (seperti semula) ---
    b1 = self.sa(self.se1(self.block1(s)), lam)          # dst. ... → out (B,1024,7,7)
    ...
    out = F.relu(self.norm5(b4))

    # --- Cabang B: ViT (opsional) dari peta stem yang sama ---
    vec_vit = self.vit(s) if self.vit is not None else None   # (B, VIT_DIM) | None

    # --- Fusi ---
    logits = self.top(out, extra=vec_vit)                # (B, n_cls)
```

**Mengapa dari peta stem (B,64,56,56), bukan dari citra mentah (B,3,224,224)?**

- **Hemat komputasi.** 56×56 jauh lebih kecil dari 224×224; jumlah token ViT jadi
  kecil dan terkendali (lihat §3).
- **Berbagi fitur awal.** Stem sudah mengekstrak fitur low-level (tepi, warna) yang
  berguna bagi kedua cabang — tidak perlu dihitung dua kali.
- **Input yang sama, dua pemroses berbeda.** Peta stem `s` menjadi titik fork
  tunggal; DenseNet dan ViT sama-sama membacanya.

---

## 3. Anatomi `ViTBranch` langkah demi langkah

File: `src/clouddensenet/vit_branch.py`. Kelas `ViTBranch` mengubah peta stem
`(B, 64, 56, 56)` menjadi **satu vektor ringkasan** `(B, VIT_DIM)`.

### Langkah 1 — Patch embedding (memecah peta jadi token)

```python
self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
```

Satu konvolusi ber-stride `patch` membelah peta 56×56 menjadi grid tak-saling-tindih.
Dengan `patch=8`:

```
grid        = 56 // 8 = 7
num_patches = 7 × 7   = 49 token
```

Tiap sel grid 8×8 → satu vektor sepanjang `dim` (VIT_DIM). Hasilnya
`(B, dim, 7, 7)`, lalu di-flatten jadi **deretan 49 token**:

```python
t = self.patch_embed(x)              # (B, dim, 7, 7)
t = t.flatten(2).transpose(1, 2)     # (B, 49, dim)
```

### Langkah 2 — Token CLS (peringkas global)

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
...
cls = self.cls_token.expand(B, -1, -1)   # (B, 1, dim)
t   = torch.cat([cls, t], dim=1)         # (B, 50, dim)
```

Satu token "CLS" yang **dapat dilatih** ditempel di depan. Setelah melewati encoder,
token inilah yang dipakai sebagai ringkasan seluruh gambar (gaya BERT/ViT standar).
Token total menjadi `1 + 49 = 50`.

### Langkah 3 — Embedding posisi

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))  # (1, 50, dim)
...
t = self.drop(t + self.pos_embed)
```

Self-attention bersifat *permutation-invariant* (tidak tahu urutan). Embedding posisi
yang dapat dilatih **menyuntikkan informasi "token ini berasal dari petak mana"**.
Ditambahkan (bukan di-concat) ke setiap token, lalu di-dropout.

> Inisialisasi `cls_token` dan `pos_embed` memakai `trunc_normal_(std=0.02)` —
> konvensi umum ViT agar training stabil.

### Langkah 4 — Transformer Encoder (bawaan PyTorch)

```python
layer = nn.TransformerEncoderLayer(
    d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio),
    dropout=drop, activation="gelu", batch_first=True)
self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
```

Dipakai `nn.TransformerEncoder` **standar PyTorch** (tanpa dependensi tambahan) agar
ringkas dan mudah dibaca. Tiap lapis = **Multi-Head Self-Attention + MLP (GELU)** dengan
residual & LayerNorm. `batch_first=True` → bentuk tensor `(B, token, dim)`.

### Langkah 5 — Ambil token CLS

```python
t = self.encoder(t)     # (B, 50, dim)
t = self.norm(t)        # LayerNorm akhir
return t[:, 0]          # token CLS → (B, dim)
```

Setelah encoder + LayerNorm akhir, hanya **token ke-0 (CLS)** yang diambil. Itulah
vektor ringkasan `(B, VIT_DIM)` yang dikirim ke fusi.

---

## 4. Hyperparameter cabang ViT (config.py)

Semua di `src/clouddensenet/config.py` (sumber kebenaran — ubah di sini, bukan
hard-code):

| Flag | Default | Arti |
|------|--------:|------|
| `USE_VIT` | `False` | `True` = aktifkan cabang ViT paralel |
| `VIT_DIM` | `192` | Dimensi embedding/token ViT. Kecil agar tak mudah overfit |
| `VIT_DEPTH` | `4` | Jumlah lapis `TransformerEncoder` |
| `VIT_HEADS` | `4` | Jumlah attention head (`VIT_DIM` **harus habis dibagi** angka ini: 192/4=48) |
| `VIT_PATCH` | `8` | Ukuran patch pada peta stem 56×56 → grid 7×7 = **49 token** |
| `VIT_MLP_RATIO` | `2.0` | Lebar hidden MLP di encoder = `VIT_DIM × rasio` (192×2=384) |
| `VIT_DROP` | `0.1` | Dropout di dalam cabang ViT |

> **Perubahan lain di commit yang sama:** `BATCH_SIZE` non-debug dinaikkan dari `8`
> menjadi `32`. Ini berdiri sendiri (bukan bagian arsitektur ViT) tapi ikut masuk di
> commit `bd03ce1` — perhatikan saat membandingkan run lama vs baru.

**Hubungan ukuran yang wajib konsisten:**

```
stem_size   = IMG_SIZE // 4   = 224 // 4 = 56     (conv stride2 + maxpool stride2)
grid        = stem_size // VIT_PATCH = 56 // 8 = 7
num_patches = grid × grid = 49
token total = num_patches + 1 (CLS) = 50
```

Jika `IMG_SIZE` atau `VIT_PATCH` diubah, `num_patches` ikut berubah dan `pos_embed`
menyesuaikan otomatis (dihitung di `__init__`).

---

## 5. Fusi di Top Block (`extra_dim`)

File: `src/clouddensenet/blocks.py`, kelas `TopBlock`. Fusi dilakukan dengan
**concatenation** — menggabung dua vektor menjadi satu yang lebih lebar.

```python
def __init__(self, in_channels=1024, ..., extra_dim=0):
    ...
    self.extra_dim = extra_dim
    lapisan, d_in = [], in_channels + extra_dim   # +extra_dim untuk fusi concat
    ...

def forward(self, x, extra=None):
    x = self.global_pool(x)      # (B,1024,1,1)  Global Average Pooling
    x = self.flatten(x)          # (B,1024)
    if extra is not None:        # fusi: gabung vektor DenseNet + vektor cabang
        x = torch.cat([x, extra], dim=1)   # (B, 1024 + extra_dim)
    x = self.bottleneck(x)       # MLP
    return self.fc(x)            # (B, n_cls) logits
```

Poin kunci:

- **`extra_dim`** memperlebar `Linear` pertama agar muat vektor gabungan. Saat ViT
  aktif, `extra_dim = VIT_DIM = 192`, jadi input MLP = `1024 + 192 = 1216`.
- **`extra=None`** (ViT nonaktif) → cabang `if` dilewati → `TopBlock` berperilaku
  **persis seperti versi lama** (input MLP tetap 1024). Inilah yang menjaga
  kompatibilitas mundur.
- DenseNet di-**Global Average Pool** dulu `(B,1024,7,7) → (B,1024)` sebelum
  di-concat dengan vektor ViT `(B,192)`.

---

## 6. Aliran tensor & shape lengkap

Dengan `IMG_SIZE=224`, `USE_VIT=True`, `VIT_DIM=192`, `VIT_PATCH=8`:

| Tahap | Operasi | Shape keluar |
|-------|---------|--------------|
| Input | citra | `(B, 3, 224, 224)` |
| Stem | conv7x7/s2 → BN → ReLU → maxpool/s2 | `(B, 64, 56, 56)` ← **titik fork** |
| **A** block1→trans1 | DenseBlock1 + SE + SA | `(B, 256, 56, 56)` → `(B, 128, 28, 28)` |
| **A** block2→trans2 | DenseBlock2 + SE + SA | `(B, 512, 28, 28)` → `(B, 256, 14, 14)` |
| **A** block3→trans3 | DenseBlock3 + SE + SA | `(B, 1024, 14, 14)` → `(B, 512, 7, 7)` |
| **A** block4 → norm5 | DenseBlock4 + SE + SA + BN + ReLU | `out = (B, 1024, 7, 7)` |
| **B** patch_embed | Conv2d k8/s8 pada stem | `(B, 192, 7, 7)` |
| **B** flatten+CLS+pos | token-isasi + CLS + posisi | `(B, 50, 192)` |
| **B** encoder → CLS | TransformerEncoder ×4 → ambil `[:,0]` | `vec_vit = (B, 192)` |
| Fusi | GAP(out)=`(B,1024)` ⊕ `vec_vit` | `(B, 1216)` |
| MLP + fc | bottleneck → Linear | `(B, 7)` logits |

Saat `return_all=True`, dictionary keluaran kini juga memuat key **`"vit"`**
(berisi `vec_vit`, atau `None` bila ViT nonaktif), di samping key lama.

---

## 7. Training: bagaimana cabang ViT dilatih

File: `src/clouddensenet/finetune.py`. Pembekuan bertahap mengklasifikasikan tiap
parameter berdasar namanya:

```python
def bekukan_imagenet(model, latih_baru=True, verbose=True):
    for n, p in model.named_parameters():
        if n.startswith("top."):
            p.requires_grad = True
        elif n in model.imagenet_keys:
            p.requires_grad = False
        else:   # SE, SA, log_q, log_tau, ...
            p.requires_grad = latih_baru and ".pq." not in n
```

Parameter cabang ViT bernama prefix **`vit.`** (mis. `vit.cls_token`,
`vit.encoder...`). Mereka:

- **bukan** `top.*`, dan
- **bukan** bagian `imagenet_keys` (bobot pra-latih ImageNet),

sehingga jatuh ke cabang `else` → **ikut dilatih sejak Tahap 1** (bersama SE, SA,
dll.) dan tetap dilatih di seluruh tahap berikutnya. Ini masuk akal: ViT **tidak punya
bobot ImageNet**, jadi harus dilatih dari awal (from scratch), tidak pernah dibekukan.

Fungsi `muat_imagenet(strict=False)` dan `nolkan_input_nd` **tidak terpengaruh**:
bobot ViT tak ada di checkpoint ImageNet, dan `strict=False` membiarkan key `vit.*`
yang "hilang" begitu saja (diinisialisasi acak oleh konstruktor).

---

## 8. Kompatibilitas mundur & checkpoint

Desain ini **aditif dan opt-in**:

- **`USE_VIT = False` (default):** `self.vit = None`, `extra_dim = 0`. Model tidak
  membangun modul ViT sama sekali; `state_dict` identik dengan versi pra-ViT.
  Checkpoint lama tetap **dimuat dan dijalankan tanpa perubahan**.
- **`USE_VIT = True`:** muncul parameter baru berprefix `vit.*` **dan** `TopBlock`
  memakai `Linear` pertama yang lebih lebar (`1024+192`). Artinya **bentuk `top.*`
  berubah**, sehingga checkpoint ViT **tidak** bisa dimuat ke model non-ViT dan
  sebaliknya. Simpan/berinama checkpoint ViT secara terpisah.
- Parameter konstruktor `use_vit` meng-override config per-instance:
  `CloudDenseNet(use_vit=True)` — dipakai di `test_forward.py` agar tak perlu
  mengubah config global.

---

## 9. Biaya parameter & cara mengaktifkan

Menurut catatan commit, cabang ViT menambah **± 2,08 juta parameter**
(dengan `VIT_DIM=192`, `VIT_DEPTH=4`). `pytest` tetap **6 passed** (termasuk test baru
`test_forward_vit_branch` yang memverifikasi output tetap `(N, 7)`).

**Mengaktifkan lewat config** (`src/clouddensenet/config.py`):

```python
USE_VIT = True
```

**Atau per-instance tanpa menyentuh config:**

```python
from clouddensenet.model import CloudDenseNet
model = CloudDenseNet(use_vit=True)
```

**Cek cepat (tanpa dataset/GPU):**

```bash
pytest                          # harus 6 passed
python tests/test_forward.py    # smoke test manual
```

---

## 10. Catatan teknis & kejujuran

- **ViT ini sengaja kecil & sederhana.** Memakai `nn.TransformerEncoder` bawaan
  PyTorch, bukan implementasi ViT penuh (mis. timm). Tujuannya keterbacaan dan nol
  dependensi tambahan, bukan performa SOTA.
- **Belum ada bobot pra-latih untuk ViT.** Cabang B dilatih dari nol. Pada dataset
  kecil (GCD) ini berisiko overfit; karena itu `VIT_DIM`, `VIT_DEPTH` dibuat kecil dan
  `VIT_DROP=0.1`. Pantau kurva val saat mengaktifkannya.
- **Efektivitas belum divalidasi di dokumen ini.** Penambahan cabang *bisa* menolong
  atau *bisa* tidak, tergantung data. Bandingkan run `USE_VIT=False` vs `True` secara
  adil — **ingat `BATCH_SIZE` juga berubah (8→32)** di commit yang sama, jadi
  kontrol variabel itu saat eksperimen.
- **AMP/fp32:** modul ViT tidak dipaksa fp32 seperti `SCConv`. Jika muncul
  instabilitas numerik di AMP, itu kandidat pertama untuk diperiksa.

---

**Dokumen terkait:**
[`ARSITEKTUR_CLOUDDENSENET.md`](ARSITEKTUR_CLOUDDENSENET.md) ·
[`PENJELASAN_MODEL_DETAIL.md`](PENJELASAN_MODEL_DETAIL.md) ·
[`REFERENSI_KOMPONEN_DETAIL.md`](REFERENSI_KOMPONEN_DETAIL.md) ·
[`../TRAINING.md`](../TRAINING.md)
