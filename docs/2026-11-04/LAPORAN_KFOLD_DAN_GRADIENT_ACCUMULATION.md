# Laporan Penerapan: Stratified K-Fold CV & Gradient Accumulation

**Proyek:** CloudDenseNet — klasifikasi 7 kelas awan (dataset GCD)
**Tanggal:** 2026-11-04
**Branch:** `Stratified-K-Fold-Cross-Validation`
**Status:** Kode terpasang & terverifikasi (smoke test lulus). Angka F1 akhir menyusul
setelah menjalankan training penuh (lihat §7).

> Dokumen ini ditulis bertingkat: bagian **🟢 Untuk Pemula** menjelaskan intuisi,
> **🔵 Untuk Mahir** menambah detail teknis. Baca yang Anda butuhkan.

---

## 1. Ringkasan eksekutif

Dua kemampuan baru ditambahkan ke pipeline training CloudDenseNet:

1. **Stratified K-Fold Cross-Validation** — evaluasi yang lebih kredibel daripada
   satu kali holdout. Melatih model K kali pada lipatan (fold) data yang berbeda,
   menjaga proporsi 7 kelas tetap sama di setiap fold, lalu melaporkan
   **F1 = rata-rata ± simpangan baku**.
2. **Gradient Accumulation** — teknik melatih dengan *batch efektif* besar meski VRAM
   GPU kecil, dengan menumpuk gradien beberapa mikro-batch sebelum satu langkah
   optimizer. Sangat berguna di fase **full defrost** (seluruh lapisan di-unfreeze)
   yang paling boros memori.

Keduanya **tidak mengubah perilaku lama**: training biasa (`scripts/train.py`) tetap
jalan persis seperti sebelumnya selama `ACCUM_STEPS = 1`.

---

## 2. Apa saja yang dikerjakan (file nyata)

| File | Jenis | Isi |
|------|-------|-----|
| `src/clouddensenet/config.py` | diubah | tambah knob `ACCUM_STEPS = 1` (Gradient Accumulation) |
| `src/clouddensenet/engine.py` | diubah | `train_one_epoch(..., accum_steps=1)` + helper `_langkah_optimizer` |
| `src/clouddensenet/data_kfold.py` | **baru** | `base_targets()`, `build_fold_loaders()` untuk K-fold |
| `scripts/train.py` | diubah | teruskan `C.ACCUM_STEPS` ke loop training + info batch efektif |
| `scripts/train_kfold.py` | **baru** | entrypoint CV: `StratifiedKFold` + jadwal `PHASES_CV` + OOF |
| `docs/STRATIFIED_KFOLD_CV.md` | sudah ada | dokumen konsep & desain (pendamping laporan ini) |
| `docs/2026-11-04/…` | **baru** | laporan ini |

Semua selaras konvensi repo: hyperparameter di `config.py`, Bahasa Indonesia untuk
komentar/nama fungsi (`siapkan_batch`, `bekukan_imagenet`, dsb.), `SEED = 42`.

---

## 3. Bagian A — Stratified K-Fold Cross-Validation

### 3.1 🟢 Untuk Pemula: kenapa butuh ini?

Bayangkan menilai kepintaran siswa hanya dari **satu** kali ulangan. Kalau soalnya
kebetulan mudah/sulit, nilai itu bisa menipu. **Holdout** (cara lama di repo: iris 15%
data jadi validasi, satu kali) persis seperti itu — skornya bergantung pada *data mana
yang kebetulan* masuk validasi.

**K-Fold** = beri **K kali ulangan** dengan soal berbeda, lalu ambil rata-ratanya.
Data dibagi K bagian; model dilatih K kali, tiap kali satu bagian jadi penguji dan
sisanya jadi bahan belajar. Hasil akhir: **rata-rata ± seberapa goyang** (std).

**Stratified** = tiap bagian dijaga punya **komposisi kelas yang sama**. Penting di
sini karena dataset awan kita timpang: kelas `7_mixed` cuma **3,5%**. Tanpa
stratifikasi, sebuah fold bisa kebetulan nyaris tak punya `7_mixed` → nilainya ngawur.

```
Holdout (lama):   [ train 85% ][ val 15% ]      -> 1 angka, rapuh
Stratified 3-Fold: data dibagi 3, proporsi kelas sama di tiap bagian
   Putaran 1: [ VAL ][train][train]
   Putaran 2: [train][ VAL ][train]   -> 3 angka F1 -> mean ± std
   Putaran 3: [train][train][ VAL ]
```

### 3.2 🔵 Untuk Mahir: desain penerapan

- CV berjalan **hanya di pool `train/`** (8.501 citra). Folder `test/` (9.000 citra)
  **tidak pernah disentuh** selama CV → tetap holdout final yang jujur.
- `scripts/train_kfold.py` memakai `sklearn.model_selection.StratifiedKFold(
  n_splits=K, shuffle=True, random_state=42)` → fold **deterministik & reproducible**,
  konsisten dengan `SEED = 42` repo.
- Indeks fold dipakai membangun `Subset` di atas **dua** `ImageFolder` pada folder
  `train/` yang sama: satu ber-augmentasi (`train_tf`) untuk subset latih, satu polos
  (`val_tf`) untuk subset validasi. Keduanya urut `sorted()` → indeks sejajar. Ini
  pola yang sama persis dengan `data.build_loaders` yang sudah ada.
- **`WeightedRandomSampler` dihitung ulang per fold** dari distribusi kelas fold itu
  (bukan bobot global), karena tiap fold komposisinya sedikit berbeda.
- **Model & optimizer fresh tiap fold** (`set_seed` → `CloudDenseNet()` +
  `muat_imagenet()` + optimizer baru). Tidak ada bobot yang dibawa antar-fold — itu
  kebocoran. Checkpoint diberi nama per-fold (`cloud_fold0_Versi01.pth`, …) agar tak
  saling menimpa.
- **Out-of-Fold (OOF):** karena tiap sampel menjadi validasi **tepat sekali**,
  prediksinya dikumpulkan → satu `classification_report` + macro-F1 atas seluruh
  8.501 citra. Disimpan ke `artifacts/results/kfold_<run>_oof.npz`.

### 3.3 Jadwal fase terkompresi (hemat waktu CV)

Jadwal training penuh bertahap sampai **80 epoch** (`config.PHASES`). Menjalankan itu
K kali sangat mahal. Untuk CV dipakai **`PHASES_CV`** — keenam fase **dikompresi
proporsional** menjadi **30 epoch** (bukan dipotong, agar fase `full` tetap diuji):

| Fase | Epoch asli | Epoch CV | LR | Mode | Target |
|-----:|-----------:|---------:|----|------|--------|
| 1 | 1–10  | **1–4**   | 1e-3 | frozen  | Top Block (ImageNet beku) |
| 2 | 11–20 | **5–8**   | 1e-4 | frozen  | Tahap 2 (ImageNet beku) |
| 3 | 21–30 | **9–12**  | 1e-5 | partial | Adaptasi Top Block |
| 4 | 31–40 | **13–16** | 5e-5 | partial | Unfreeze Dense Block 3–4 |
| 5 | 41–60 | **17–23** | 1e-5 | full    | Partial Fine-Tuning |
| 6 | 61–80 | **24–30** | 1e-6 | full    | Full Defrost (end-to-end) |

Panjang fase `4·4·4·4·7·7 = 30` — **terverifikasi** (lihat §5). LR dipertahankan;
hanya jumlah epoch dipadatkan.

> 🔵 **Kenapa kompresi ini sah?** Mode `sampler` menarik `SAMPLES_PER_EPOCH = 21.000`
> sampel/epoch → 1 epoch = 21.000 / `BATCH_SIZE`(8) = **2.625 langkah optimizer**.
> Jadi fase 5 (7 epoch) tetap ~18.375 langkah — banyak. Model tetap sempat belajar
> meski epoch dipadatkan.

### 3.4 Cara menjalankan

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_kfold.py                    # default: 3 fold, 30 epoch/fold
python scripts/train_kfold.py --folds 5          # 5-fold
python scripts/train_kfold.py --accum-steps 4    # + gradient accumulation (lihat §4)
```

### 3.5 Cara membaca hasil

Di akhir, skrip mencetak dan menyimpan:

```
Hasil 3-Fold Stratified CV (macro-F1), jadwal 30 epoch:
  fold 0: 0.xxxx
  fold 1: 0.xxxx
  fold 2: 0.xxxx
  ----------------------------------
  F1 = 0.xxxx ± 0.0xxx         <- INI yang dilaporkan di skripsi/paper
  OOF macro-F1 (gabungan) = 0.xxxx
```

- `F1 = mean ± std` → estimasi performa + ketidakpastiannya.
- File `artifacts/results/kfold_cloud.json` → ringkasan machine-readable.
- `..._oof.npz` → `y_true`, `y_pred` untuk membuat confusion matrix gabungan.

---

## 4. Bagian B — Gradient Accumulation

### 4.1 🟢 Untuk Pemula: masalah & solusi

Model belajar lebih stabil kalau melihat **banyak gambar sekaligus** (batch besar).
Tapi batch besar butuh **VRAM besar** — dan GPU sering kehabisan memori (error "CUDA
out of memory"), terutama saat **semua lapisan dibuka** (full defrost), karena gradien
untuk setiap lapisan harus disimpan.

**Gradient Accumulation** = trik "mencicil". Alih-alih memproses 32 gambar sekaligus
(berat), proses **4 × 8 gambar** satu per satu, **jumlahkan gradiennya**, baru
perbarui bobot sekali. Hasil matematisnya **setara batch 32**, tapi memori puncaknya
cuma sebesar batch 8.

```
Tanpa accumulation (butuh VRAM batch 32):
  [32 gambar] -> hitung gradien -> update bobot

Dengan accumulation ACCUM_STEPS=4 (VRAM cukup batch 8):
  [8] -> grad   \
  [8] -> grad    > jumlahkan  -> update bobot sekali  (= batch efektif 32)
  [8] -> grad   /
  [8] -> grad  /
```

### 4.2 🔵 Untuk Mahir: implementasi di `engine.py`

`train_one_epoch(..., accum_steps=1)` kini mendukung akumulasi. Inti perubahan:

1. `optimizer.zero_grad()` dipindah ke **luar** loop (sekali di awal) + setelah tiap
   langkah optimizer.
2. Loss dibagi `accum_steps` sebelum `backward()`:
   `scaler.scale(loss / accum_steps).backward()`. Ini membuat gradien terakumulasi =
   **rata-rata** batch efektif (konsisten dengan reduksi `mean` pada FocalLoss/CE),
   bukan jumlah — jadi skala LR tidak perlu diubah.
3. Langkah optimizer (unscale → clip → step → update → zero_grad) hanya dijalankan
   **setiap `accum_steps` mikro-batch**, dan sekali lagi untuk sisa di akhir epoch.

Interaksi dengan komponen lain — semua terjaga benar:

- **AMP (`GradScaler`):** `scaler.scale().backward()` dipanggil tiap mikro-batch;
  `unscale_` + `step` + `update` sekali per langkah akumulasi. Urutan ini sesuai
  rekomendasi PyTorch untuk akumulasi + AMP.
- **Gradient clipping:** `clip_grad_norm_(GRAD_CLIP=5.0)` dihitung atas gradien
  **yang sudah terakumulasi**, tepat sebelum `step` — benar.
- **Pengaman NaN:** batch NaN tetap dilewati (`continue`) tanpa menambah hitungan
  akumulasi; statistik BatchNorm yang tercemar dipulihkan dari backup. Logika lama
  utuh.
- **BatchNorm beku:** `set_bn_beku_eval` tetap dipanggil. Catatan: akumulasi membuat
  statistik BN diperbarui per **mikro-batch** (ukuran 8), bukan batch efektif 32 —
  ini perilaku standar akumulasi dan dapat diterima.

### 4.3 Cara memakai

```powershell
# Opsi 1: global, lewat config (berlaku untuk train.py & train_kfold.py)
#   edit src/clouddensenet/config.py -> ACCUM_STEPS = 4

# Opsi 2: khusus CV, lewat argumen
python scripts/train_kfold.py --accum-steps 4     # batch efektif 8 x 4 = 32
```

Panduan memilih nilai: naikkan `ACCUM_STEPS` saat muncul *CUDA out of memory*, mulai
dari 2 lalu 4/8. Batch efektif = `BATCH_SIZE × ACCUM_STEPS`. Karena fase `full`
(epoch 24–30 di CV / 41–80 di training penuh) paling boros VRAM, di situlah
akumulasi paling terasa manfaatnya.

> 🔵 **Trade-off:** akumulasi menukar **waktu** dengan **memori** — epoch jadi sedikit
> lebih lambat (lebih banyak forward/backward per langkah optimizer) tapi muat di GPU
> kecil. Ia **tidak** mempercepat training; ia **memungkinkan** batch besar.

---

## 5. Verifikasi yang sudah dilakukan

| Cek | Perintah | Hasil |
|-----|----------|-------|
| Byte-compile semua file | `python -m py_compile …` | ✅ OK |
| Smoke test forward pass | `pytest` | ✅ **2 passed** (83 dtk) |
| Jadwal fase CV benar | impor `train_kfold` | ✅ panjang `[4,4,4,4,7,7]=30` |
| Stratifikasi nyata (3-fold) | `StratifiedKFold` di distribusi GCD asli | ✅ lihat bawah |

Bukti stratifikasi pada distribusi train GCD sebenarnya (8.501 citra):

```
fold 0: val=2834  dist=[220,205,327,609,523,851, 99]  porsi_mixed=3.49%
fold 1: val=2834  dist=[220,205,327,609,523,851, 99]  porsi_mixed=3.49%
fold 2: val=2833  dist=[219,206,326,610,523,851, 98]  porsi_mixed=3.46%
porsi mixed populasi = 3.48%
```

Setiap fold validasi memegang porsi `7_mixed` ≈ 3,48% — **identik** dengan populasi.
Inilah yang dijamin stratifikasi: fold yang representatif untuk semua kelas, termasuk
minoritas.

---

## 6. Biaya komputasi (ringkas)

Satuan = *epoch-run*. Training penuh = 80 epoch. CV melatih K model dari nol.

| Strategi | K | Epoch/fold | Epoch CV | + Model final (80) | Relatif holdout |
|----------|--:|-----------:|---------:|-------------------:|----------------:|
| Holdout (lama) | 1 | 80 | 80 | — | 1,0× |
| 5-Fold penuh | 5 | 80 | 400 | 480 | 6,0× |
| **3-Fold terkompresi (default)** | 3 | 30 | 90 | 170 | **2,1×** |
| 5-Fold terkompresi | 5 | 30 | 150 | 230 | 2,9× |

Contoh konkret: jika 1 epoch ≈ 18 menit di GPU Anda, maka **3-Fold terkompresi ≈
27 jam** CV, + ~24 jam untuk model final = **± 51 jam** total proyek. Estimasi
rasio-epoch ini valid karena kompresi mempertahankan proporsi fase (porsi `full`
~47%), sehingga biaya rata-rata per-epoch hampir sama dengan run penuh.

---

## 7. Catatan kejujuran & langkah berikutnya

- **Belum ada angka F1 empiris** di laporan ini: CV penuh butuh ~puluhan jam GPU dan
  belum dijalankan. Yang sudah pasti adalah **kode terpasang + lolos verifikasi**
  (§5). Jangan melaporkan angka yang belum dihasilkan.
- **Alur yang benar untuk model final:** CV dipakai untuk *memilih konfigurasi* dan
  *mengestimasi* performa (`mean ± std`). Model produksi dilatih **sekali** di
  **seluruh** `train/` dengan jadwal penuh `PHASES` (80 epoch), lalu dievaluasi
  **sekali** di `test/` (9.000 citra). Jangan tukar angka CV dengan angka test final.
- **Saran lanjutan (opsional):** jika ada citra langit nyaris-duplikat (burst dari
  langit yang sama), pertimbangkan `StratifiedGroupKFold` agar satu grup tak terpecah
  antara train & val — mencegah kebocoran kemiripan. Config sudah punya penanda
  `GROUPED_SPLIT`/`GROUP_THR = 0.95` untuk kekhawatiran ini.

---

## 8. Referensi silang

- `docs/STRATIFIED_KFOLD_CV.md` — dokumen konsep & desain (lebih dalam soal teori).
- `docs/EDA_AUGMENTASI.md` — sumber angka distribusi kelas train.
- `src/clouddensenet/config.py` — sumber kebenaran hyperparameter (`ACCUM_STEPS`,
  `PHASES`, `SEED`, `VAL_SPLIT`, `IMBALANCE`).
- `scripts/train_kfold.py` — entrypoint CV (`PHASES_CV`, `phase_of_cv`, OOF).
- `src/clouddensenet/engine.py` — `train_one_epoch(accum_steps=...)`.
