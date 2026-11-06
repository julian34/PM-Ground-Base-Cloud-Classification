# Pembagian Dataset & Augmentasi — CloudDenseNet

Dokumentasi **pembagian dataset GCD** (7 kelas awan) menjadi `train/val/test` dan
ringkasan **augmentasi yang dilakukan** saat training.

> **Sumber kebenaran:** `scripts/split_dataset.py` (split offline), `src/clouddensenet/data.py`
> (`build_loaders`, `build_transforms`, carve val runtime), dan `src/clouddensenet/config.py`.
> Angka pada dokumen ini dihitung **langsung dari `data/processed/` pada mesin ini**.

---

## 1. Sumber data

`data/raw/GCD.zip` → diekstrak menjadi `data/raw/GCD/{train,test}/<kelas>/*.jpg`.

- Dataset GCD asli **hanya** punya `train/` dan `test/` — **tidak ada `val/`**.
- Validasi (`val/`) dibuat sendiri dari `train/` (lihat §3), sehingga `test/` tetap
  benar-benar *held-out* dan tidak pernah dilihat model saat training.

7 kelas (urutan indeks ImageFolder 0–6):
`1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.

---

## 2. Hasil split (angka nyata di disk)

| Kelas             |  train |   val |  test | train+val | Porsi (tr+val) |
|-------------------|-------:|------:|------:|----------:|---------------:|
| 1_cumulus         |    659 |   116 |   750 |       775 |         7,8 %  |
| 2_altocumulus     |    616 |   109 |   750 |       725 |         7,2 %  |
| 3_cirrus          |    980 |   173 |   753 |     1.153 |        11,5 %  |
| 4_clearsky        |  1.828 |   322 | 1.589 |     2.150 |        21,5 %  |
| 5_stratocumulus   |  1.569 |   277 | 1.790 |     1.846 |        18,5 %  |
| 6_cumulonimbus    |  2.553 |   450 | 2.761 |     3.003 |        30,0 %  |
| 7_mixed           |    296 |    52 |   607 |       348 |         3,5 %  |
| **TOTAL**         | **8.501** | **1.499** | **9.000** | **10.000** | **100 %** |

- **train + val = 10.000** (seluruh `train/` GCD asli), dibagi **85 % / 15 %** per kelas.
- **test = 9.000** (salinan utuh `test/` GCD asli, tidak disentuh split).
- **Rasio ketidakseimbangan ≈ 8,6 : 1** — `6_cumulonimbus` (3.003) vs `7_mixed` (348).
  Ini ditangani di level *sampling* + loss, bukan dengan mengubah split (lihat §6).

> ⚠️ **Catatan:** komentar `N_SPLIT = (8340, 1471, 9000)` di `config.py` **tidak sinkron**
> dengan isi `data/processed/` saat ini (yang benar **8.501 / 1.499 / 9.000**). `N_SPLIT`
> hanya catatan dan tidak dipakai saat training — tapi sebaiknya diperbarui agar tidak
> membingungkan.

### 2.1 Distribusi kelas (train) — tidak seimbang

```text
6_cumulonimbus  ██████████████████████████  2.553  (30,0 %)
4_clearsky      ███████████████████         1.828  (21,5 %)
5_stratocumulus ████████████████            1.569  (18,5 %)
3_cirrus        ██████████                    980  (11,5 %)
1_cumulus       ███████                       659  ( 7,8 %)
2_altocumulus   ██████                        616  ( 7,2 %)
7_mixed         ███                           296  ( 3,5 %)
```

---

## 3. Metode pembagian

Ada **dua jalur** menghasilkan `val`. Keduanya *stratified per-kelas* dan *seeded*
(`SEED = 42`), jadi proporsi tiap kelas terjaga dan hasilnya reproducible.

### 3.1 Jalur utama — split offline (`scripts/split_dataset.py`)

Menyalin/memindahkan isi `data/raw/GCD` → `data/processed/{train,val,test}`:

```bash
python scripts/split_dataset.py                 # ekstrak zip + split (copy; raw tetap utuh)
python scripts/split_dataset.py --move          # pindahkan train (hemat disk), test tetap copy
python scripts/split_dataset.py --val-frac 0.15 --seed 42 --force
```

Logika per kelas:

1. Kumpulkan semua citra di `train/<kelas>/`.
2. Acak indeks dengan RNG ber-seed (`numpy.random.default_rng(42)`).
3. Ambil `n_val = max(1, round(jumlah × val_frac))` citra pertama → `val/`, sisanya → `train/`.
4. `test/<kelas>/` **selalu di-copy utuh** (menjaga `data/raw/test` tetap cadangan).

| Argumen | Default | Keterangan |
|---|---|---|
| `--zip` | `data/raw/GCD.zip` | Arsip sumber (diekstrak bila `raw-dir` belum ada). |
| `--raw-dir` | `data/raw/GCD` | Folder hasil ekstrak (`train/` + `test/`). |
| `--out` | `data/processed` | Tujuan split. |
| `--val-frac` | `0.15` | Proporsi tiap kelas yang jadi validasi. |
| `--seed` | `42` | Seed RNG (sinkron dengan `config.py`). |
| `--move` | *(off)* | Pindahkan file train alih-alih menyalin. |
| `--force` | *(off)* | Timpa folder `--out` bila sudah ada. |

### 3.2 Jalur cadangan — carve val saat runtime (`data.py`)

Bila `<data_root>/val/` **tidak ada**, `build_loaders()` meng-carve val dari `train/`
**saat memuat data** lewat `_stratified_val_indices()` (juga 15 %, stratified, seed 42).
Train & val tetap dibaca dari folder `train/` yang sama, hanya dengan *transform* berbeda
(train diaugmentasi, val tidak).

> Pada repo ini `data/processed/val/` **sudah ada** (hasil §3.1), jadi jalur cadangan ini
> tidak aktif — val dibaca langsung dari folder.

### 3.3 Subset test seimbang (opsional, untuk evaluasi)

`build_test_loader(..., per_class=N)` dapat menyaring test jadi **N citra per kelas**
(stratified, seed 42) untuk evaluasi yang seimbang antar-kelas. Default `None` memakai
seluruh 9.000 citra test apa adanya.

---

## 4. Reproducibility

- `SEED = 42` diterapkan via `set_seed()` (random, numpy, torch, cuda) dan RNG split.
- Carve/subset memakai `numpy.random.default_rng(42)` → urutan pemilihan selalu sama.
- DataLoader memakai `seed_worker` + `Generator(manual_seed=42)` agar urutan sampler
  dan worker konsisten antar-run.

---

## 5. Augmentasi yang dilakukan (ringkas)

Augmentasi bersifat **on-the-fly per-sampel** (bukan disimpan ke disk) — tiap epoch model
melihat variasi acak berbeda. Pipeline aktif dari `build_transforms()`:

```text
Resize(224×224)
  → Pad(56, mode="reflect")                 # 224 → 336, cegah border hitam
  → RandomAffine(degrees=±15, translate=±0.10)
  → CenterCrop(224)                          # kembali ke 224×224
  → RandomHorizontalFlip(p=0.5)
  → ColorJitter(brightness=0.8–1.3, contrast=0.8–1.4,
                saturation=0.9–1.1, hue=0)   # hue=0: jaga rasio Biru/Merah
  → ToTensor()                               # [0,1], belum dinormalisasi
```

| Transform | Parameter | Tujuan singkat |
|---|---|---|
| `Resize` | 224×224 | Seragamkan ukuran input DenseNet-121. |
| `Pad` (reflect) | 56 px (`IMG_SIZE//4`) | Bantalan pantulan agar affine tak munculkan border hitam. |
| `RandomAffine` | rotasi ±15°, translasi ±10 % | Invarian orientasi & posisi awan (rigid ringan, tanpa zoom/shear). |
| `CenterCrop` | 224×224 | Potong kembali ke ukuran target. |
| `RandomHorizontalFlip` | p = 0.5 | Awan simetris horizontal → variasi gratis. |
| `ColorJitter` | bright 0.8–1.3, contrast 0.8–1.4, sat 0.9–1.1, **hue 0** | Simulasi pencahayaan langit; **hue=0** menjaga fitur sadar-fisika λ B/R. |
| `ToTensor` | — | PIL → tensor `[3,224,224]` di `[0,1]`. |

- **Val & test:** tanpa augmentasi — hanya `Resize(224×224) → ToTensor()` (deterministik).
- **Normalisasi** ImageNet (`mean/std`) dilakukan belakangan di `siapkan_batch()`, di device.
- **`STRONG_AUG = False`** → augmentasi kuat (rotasi 90°, vertical flip, hue jitter,
  grayscale, equalize) **nonaktif**. **MixUp** dan **label smoothing** juga **mati**
  (`MIXUP_P = 0`, `LABEL_SMOOTHING = 0`).

> 📖 **Detail lengkap** tiap augmentasi (definisi, alasan domain, interaksi dengan sinyal
> λ Biru/Merah, dan dampak mode `STRONG_AUG`) ada di
> [`EDA_AUGMENTASI.md`](EDA_AUGMENTASI.md).

---

## 6. Penanganan ketidakseimbangan (bukan bagian split)

Split **tidak** menyeimbangkan kelas — distribusi asli dipertahankan. Ketidakseimbangan
8,6 : 1 dilawan saat training:

- **`IMBALANCE = "sampler"`** → `WeightedRandomSampler` (bobot `1/jumlah_kelas`,
  `replacement=True`), `SAMPLES_PER_EPOCH = 7 × 3000 = 21.000` → tiap kelas ditarik
  ~3.000/epoch. Sampel minoritas yang berulang **tidak identik** karena augmentasi acak.
- **Focal Loss** (γ = 2.0) menambah fokus ke sampel sulit/minoritas.

---

*Dibuat sebagai dokumentasi pembagian dataset. Perbarui bila `config.py`, `data.py`, atau
`scripts/split_dataset.py` berubah. Untuk langkah menjalankan split, lihat
[`../TRAINING.md`](../TRAINING.md) §5.*
