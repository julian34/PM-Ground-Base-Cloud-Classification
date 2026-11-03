# EDA — Augmentasi Dataset CloudDenseNet

Analisis eksploratif (EDA) dataset **GCD** (7 kelas awan) dengan fokus pada
**augmentasi on-the-fly** yang diterapkan saat training. Dokumen ini menjelaskan
*apa* yang diaugmentasi, *kenapa*, dan *bagaimana* efeknya terhadap distribusi data.

> Sumber kebenaran: `src/clouddensenet/data.py` (`build_transforms`, `build_loaders`,
> `siapkan_batch`) dan `src/clouddensenet/config.py`. Angka dataset diambil langsung
> dari `data/processed/` pada mesin ini.

---

## 1. Ringkasan dataset

| Split  | Jumlah citra | Sumber                                   |
|--------|-------------:|------------------------------------------|
| train  | **8.501**    | `data/processed/train/<kelas>/`          |
| val    | **1.499**    | di-carve stratified dari train (seed 42) |
| test   | **9.000**    | `data/processed/test/<kelas>/`           |

- **Resolusi target:** 224 × 224 piksel, RGB.
- **7 kelas** (urutan indeks ImageFolder 0–6):
  `1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed`.

### 1.1 Distribusi kelas (train) — tidak seimbang

| Idx | Kelas             | Jumlah | Porsi   | Bar                              |
|----:|-------------------|-------:|--------:|----------------------------------|
| 0   | 1_cumulus         |    659 |  7,8 %  | `██████▌`                        |
| 1   | 2_altocumulus     |    616 |  7,2 %  | `██████`                         |
| 2   | 3_cirrus          |    980 | 11,5 %  | `█████████▊`                     |
| 3   | 4_clearsky        |  1.828 | 21,5 %  | `██████████████████▎`            |
| 4   | 5_stratocumulus   |  1.569 | 18,5 %  | `███████████████▋`               |
| 5   | 6_cumulonimbus    |  2.553 | 30,0 %  | `█████████████████████████▌`     |
| 6   | 7_mixed           |    296 |  3,5 %  | `███`                            |

- **Rasio ketidakseimbangan ≈ 8,6 : 1** (kelas terbesar `6_cumulonimbus` = 2.553
  vs terkecil `7_mixed` = 296).
- Implikasi: tanpa penanganan, model akan bias ke kelas mayoritas. Dua mekanisme
  dipakai untuk melawan ini → **WeightedRandomSampler** (lihat §4) dan **Focal Loss**
  (γ = 2.0, di luar cakupan augmentasi).

---

## 2. Pipeline augmentasi train (on-the-fly)

Augmentasi dilakukan **per-sampel saat DataLoader membaca batch** (bukan disimpan ke
disk), jadi setiap epoch model melihat variasi acak yang berbeda. Pipeline persis dari
`build_transforms()`:

```
Resize(224×224)
  → Pad(56, mode="reflect")                 # 224 → 336, border pantulan
  → RandomAffine(degrees=15, translate=0.10)
  → CenterCrop(224)                          # kembali ke 224×224
  → RandomHorizontalFlip(p=0.5)
  → ColorJitter(brightness=[0.8,1.3],
                contrast=[0.8,1.4],
                saturation=[0.9,1.1],
                hue=0)
  → ToTensor()                               # [0,1], belum dinormalisasi
```

Normalisasi ImageNet (mean/std) **tidak** dilakukan di sini — dilakukan belakangan di
`siapkan_batch()` (lihat §3).

### 2.1 Penjelasan tiap tahap

| Tahap | Transform | Parameter | Tujuan |
|------:|-----------|-----------|--------|
| 1 | `Resize` | 224×224 | Samakan ukuran semua citra. |
| 2 | `Pad` (reflect) | 56 px (= `IMG_SIZE//4`), `padding_mode="reflect"` | Beri "bantalan" pantulan di tepi supaya rotasi & translasi tahap berikutnya **tidak memunculkan border hitam**. Ukuran sementara jadi 336×336. |
| 3 | `RandomAffine` | rotasi ±15°, translasi ±10 % | Invarian terhadap orientasi & posisi awan dalam frame. |
| 4 | `CenterCrop` | 224×224 | Potong kembali ke ukuran target setelah affine, membuang tepi berlebih. |
| 5 | `RandomHorizontalFlip` | p = 0.5 | Awan simetris horizontal → gandakan variasi gratis. |
| 6 | `ColorJitter` | brightness 0.8–1.3, contrast 0.8–1.4, saturation 0.9–1.1, hue 0 | Simulasi kondisi pencahayaan langit berbeda (siang/mendung/kontras) tanpa mengubah warna dominan — **hue sengaja 0** agar rasio Biru/Merah (fitur sadar-fisika, §5) tetap bermakna. |
| 7 | `ToTensor` | — | PIL → tensor `[3,224,224]` di rentang `[0,1]`. |

### 2.2 Augmentasi "kuat" (opsional, saat ini **NONAKTIF**)

Flag `STRONG_AUG` di `config.py` saat ini **`False`**. Bila diaktifkan, pipeline
menambah transform berikut (disisipkan via `*([...] if STRONG_AUG else [])`):

| Transform tambahan | Efek |
|--------------------|------|
| `RandomChoice([RandomRotation(0/90/180/270)])` | Rotasi tegak-lurus (awan tak punya arah "atas" baku). |
| `RandomVerticalFlip(0.5)` | Balik vertikal. |
| `ColorJitter(hue=0.08)` | Jitter rona ringan (meng-override hue=0 di atas). |
| `RandomGrayscale(0.2)` | 20 % batch jadi grayscale → paksa model belajar tekstur, bukan hanya warna. |
| `RandomEqualize(0.2)` | Pemerataan histogram → tingkatkan kontras acak. |

> **Catatan EDA:** mengaktifkan `STRONG_AUG` memperkuat regularisasi tetapi `hue=0.08`
> dan `RandomGrayscale` akan **mengganggu asumsi fisik λ B/R** (§5). Jika mengaktifkannya,
> evaluasi ulang kontribusi cabang spektral.

### 2.3 Transform validasi/test (tanpa augmentasi)

Val & test **tidak** diaugmentasi — hanya deterministik, agar evaluasi konsisten:

```
Resize(224×224) → ToTensor()
```

---

## 2.4 Penjelasan detail tiap augmentasi (deep dive)

Bagian ini menguraikan **setiap** transform di `build_transforms()` secara rinci:
*definisi operasinya*, *parameter persis di proyek ini*, *bagaimana ia mengubah citra*,
*alasan domain* (kenapa cocok untuk citra langit/awan), dan *interaksinya dengan
sinyal λ B/R* (§5). Urutan mengikuti pipeline aktif.

### A. `Resize((224, 224))` — penyeragaman ukuran

- **Operasi:** interpolasi bilinear (default torchvision) ke kanvas tepat 224 × 224.
  Ini **deterministik** (bukan augmentasi acak) — langkah pra-proses agar seluruh
  batch punya dimensi sama untuk DenseNet-121 (yang mengharapkan input 224 × 224).
- **Catatan rasio aspek:** memberi tuple `(224, 224)` memaksa bentuk persegi tanpa
  menjaga rasio aspek asli; citra lanskap/potret akan sedikit teregang. Untuk awan
  (objek bertekstur, tanpa geometri kaku) distorsi ringan ini **tidak merusak label**.
- **Dampak distribusi:** tidak mengubah statistik warna, hanya resolusi spasial.

### B. `Pad(56, padding_mode="reflect")` — bantalan tepi anti-border hitam

- **Parameter:** `56 = IMG_SIZE // 4`. Citra 224 × 224 → **336 × 336** (tambah 56 px di
  keempat sisi).
- **`reflect`:** tepi diisi dengan **pantulan cermin** piksel di dalam citra (bukan nol
  / hitam, bukan replikasi piksel tepi). Untuk langit yang gradasinya halus, pantulan
  menghasilkan sambungan tekstur yang **mulus dan natural**.
- **Kenapa ada:** ini "asuransi" untuk tahap **RandomAffine** berikutnya. Saat citra
  dirotasi/ditranslasi, sudut-sudut frame bisa kosong; tanpa bantalan ini `CenterCrop`
  akan menampakkan **border hitam** yang bisa dipelajari model sebagai fitur palsu.
  Dengan pad 56 px, area yang nanti dipakai (`CenterCrop(224)`) **selalu terisi konten**.
- **λ B/R:** netral — pantulan menyalin nilai kanal apa adanya, rasio B/R tepi tetap
  konsisten dengan area aslinya.

### C. `RandomAffine(degrees=15, translate=(0.10, 0.10))` — invarian orientasi & posisi

- **Rotasi:** sudut diambil **uniform di [−15°, +15°]** setiap pembacaan sampel.
- **Translasi:** geser horizontal & vertikal **masing-masing uniform di ±10 %** dari
  lebar/tinggi (pada kanvas 336 px ≈ **±33,6 px**). Tidak ada `scale` maupun `shear`
  (keduanya `None`), jadi **tidak ada zoom/geser-miring** — hanya rigid-body ringan.
- **Kenapa ±15° (bukan 360°):** awan bukan objek tanpa arah sepenuhnya — struktur
  vertikal didorong gravitasi (mis. menara **cumulonimbus**, dasar datar). Rotasi
  **moderat** menambah variasi orientasi kamera tanpa menciptakan langit yang
  "terbalik" secara tidak fisik. (Rotasi kelipatan 90° baru muncul di `STRONG_AUG`, §2.2.)
- **Beroperasi di kanvas 336 px** (hasil pad B), sehingga konten keluar-frame diisi oleh
  bantalan reflect, bukan hitam.
- **λ B/R:** netral — transformasi geometris murni memindahkan piksel, tidak mengubah
  nilai per-kanal, jadi rasio B/R tiap piksel tetap.

### D. `CenterCrop(224)` — kembalikan ke ukuran target

- **Operasi:** potong 224 × 224 di **pusat** kanvas 336 px pasca-affine.
- **Efek gabungan B→C→D:** pad-reflect → affine → center-crop berperilaku seperti
  **random-translate + random-rotate dengan tepi selalu terisi**. Inilah resep klasik
  untuk augmentasi geometris tanpa artefak border.
- **λ B/R:** netral (pemotongan spasial).

### E. `RandomHorizontalFlip(p=0.5)` — cermin kiri-kanan gratis

- **Operasi:** dengan peluang 0,5, balik citra secara horizontal.
- **Kenapa aman:** awan **tidak punya kiralitas kiri/kanan** — citra yang dicerminkan
  tetap contoh valid dari kelas yang sama. Ini melipatgandakan variasi efektif tanpa
  biaya dan tanpa risiko label salah.
- **λ B/R:** netral — flip hanya menata-ulang posisi piksel; nilai B dan R tiap piksel
  tidak berubah.

### F. `ColorJitter(brightness=(0.8,1.3), contrast=(0.8,1.4), saturation=(0.9,1.1), hue=0)`

Augmentasi **fotometrik** — mensimulasikan kondisi cahaya & kamera berbeda. Keempat
komponen diterapkan dalam urutan acak internal torchvision.

- **`brightness=(0.8, 1.3)`:** faktor diambil uniform di [0,8 – 1,3] lalu **mengalikan**
  intensitas piksel. Mensimulasikan matahari terang vs redup / over-under exposure.
  Karena **multiplikatif pada semua kanal secara sama**, rasio B/R ≈ terjaga.
- **`contrast=(0.8, 1.4)`:** interpolasi piksel ke arah rata-rata abu-abu citra.
  < 1 = lebih "berkabut"/flat; > 1 = lebih tajam. Meniru haze atmosfer vs langit jernih.
- **`saturation=(0.9, 1.1)`:** sengaja **sempit (±10 %)**. Saturasi mengubah jarak warna
  dari abu-abu, sehingga bisa menggeser keseimbangan B vs R. Rentang sempit menjaga
  perbedaan **langit biru vs awan abu-abu** tetap informatif.
- **`hue=0` (KUNCI):** **tidak ada rotasi rona sama sekali** di mode default. Rotasi hue
  memutar roda warna dan akan **merusak rasio Biru/Merah** yang jadi fitur sadar-fisika
  (§5). Inilah batasan fotometrik paling penting di proyek ini.
- **λ B/R:** komponen brightness/contrast relatif aman (skala ~seragam antar kanal);
  saturation dibatasi; hue dimatikan — **desain sadar-fisika yang disengaja**.

### G. `ToTensor()` — konversi akhir

- **Operasi:** PIL `HWC` uint8 [0, 255] → tensor `CHW` `float32` di **[0, 1]** (÷255).
- **Tidak** menormalisasi mean/std ImageNet di sini; itu ditunda ke `siapkan_batch()`
  (§3) agar bisa di-device dan konsisten dengan val/test.

---

### Augmentasi mode `STRONG_AUG` (detail) — saat ini **NONAKTIF** (`STRONG_AUG=False`)

Disisipkan via `*([...] if STRONG_AUG else [])`. Diuraikan di sini untuk kelengkapan dan
sebagai peringatan dampaknya terhadap λ B/R.

- **`RandomChoice([RandomRotation((a, a)) for a in (0, 90, 180, 270)])`:**
  `RandomRotation((a, a))` dengan batas bawah = atas berarti rotasi **tepat a°**
  (deterministik), dan `RandomChoice` memilih **satu** dari {0°, 90°, 180°, 270°} secara
  seragam. Digabung dengan affine ±15° (C) menghasilkan orientasi hampir sembarang.
  *Caveat:* 90°/180°/270° membuat menara **cumulonimbus** bisa "menyamping/terbalik" —
  tidak fisik; uji ulang apakah menolong atau menyakiti kelas berstruktur-vertikal. λ B/R netral.
- **`RandomVerticalFlip(p=0.5)`:** balik atas-bawah. *Caveat:* awan punya asimetri
  gravitasi (dasar datar di bawah, puncak membumbung) — flip vertikal dapat menciptakan
  sampel tak realistis. λ B/R netral (spasial).
- **`ColorJitter(..., hue=0.08)`:** meng-**override** `hue=0` default dengan rotasi rona
  **±0,08 × 360° ≈ ±28,8°**. **Merusak rasio B/R** — inilah konflik utama dengan §5.
- **`RandomGrayscale(p=0.2)`:** 20 % sampel dijadikan grayscale (R=G=B via luminansi
  berbobot). Memaksa model belajar **tekstur**, bukan warna — tetapi **menghapus total**
  sinyal B/R (rasio → 1). Caveat terberat untuk cabang spektral.
- **`RandomEqualize(p=0.2)`:** pemerataan histogram **per-kanal** pada 20 % sampel —
  pemetaan **non-linear** yang berbeda untuk tiap kanal, sehingga menggeser B/R secara
  tak-seragam. λ B/R terganggu.

> **Kesimpulan deep-dive:** keempat transform geometris (B–E) **netral** terhadap λ B/R,
> sedangkan transform fotometrik (F) **dirancang** menjaga rasio B/R (hue=0, saturasi
> sempit). Semua transform mode `STRONG_AUG` yang mengubah warna (hue, grayscale,
> equalize) **berkonflik** dengan fitur sadar-fisika dan hanya boleh diaktifkan dengan
> evaluasi ulang kontribusi cabang spektral.

### Augmentasi level-batch yang tersedia tapi **tidak** dijalankan

- **MixUp:** `MIXUP_P = 0.0` → **mati**. Bila diaktifkan (`MIXUP_ALPHA = 0.2`) akan
  mencampur pasangan citra+label pada level batch (di luar `build_transforms`).
- **Label smoothing:** `LABEL_SMOOTHING = 0.0` → **mati** (regularisasi label, bukan citra).

Keduanya dicatat agar inventaris "augmentasi yang dilakukan" lengkap: **keduanya saat ini
tidak berkontribusi** pada pipeline.

---

## 3. Normalisasi (pasca-augmentasi, di `siapkan_batch`)

Augmentasi hanya menghasilkan tensor `[0,1]`. Normalisasi dipisah ke `siapkan_batch()`
agar bisa dilakukan di device (GPU) dan konsisten train/val:

- `NORM_MODE = "imagenet"` → `x = (x − mean) / std`
  dengan `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
- Alternatif `NORM_MODE = "01"` → hanya `÷255` (tanpa mean/std).

---

## 4. Penyeimbangan kelas saat sampling (bagian dari strategi data)

`IMBALANCE = "sampler"` → train memakai **`WeightedRandomSampler`** dengan bobot
`1 / jumlah_kelas` per sampel dan `replacement=True`:

- `SAMPLES_PER_EPOCH = 7 × 3000 = 21.000` sampel per epoch.
- Efek: tiap epoch, **ketujuh kelas ditarik kira-kira setara (~3.000/kelas)**, sehingga
  kelas minoritas (`7_mixed`, hanya 296 citra asli) **di-oversample dengan pengulangan**.
- Sinergi dengan augmentasi on-the-fly: sampel minoritas yang berulang **tidak identik**
  karena setiap pembacaan menghasilkan crop/flip/jitter acak yang berbeda — mengurangi
  risiko overfitting akibat oversampling.

```
Distribusi asli (tak seimbang)        →  Distribusi efektif per epoch (sampler)
6_cumulonimbus  ██████████ 2553           setiap kelas  ███████ ~3000
7_mixed         █ 296                      setiap kelas  ███████ ~3000
```

Mode alternatif: `"weights"` (bobot di loss) atau `"none"` (shuffle biasa).

---

## 5. Fitur sadar-fisika λ B/R (bukan augmentasi, tapi memengaruhi pilihan augmentasi)

`USE_BR = True` → `siapkan_batch` menghitung peta **λ = rasio kanal Biru/Merah**
(`spectral.lambda_relatif`) sebagai channel tambahan untuk cabang atensi spasial.
Rasio B/R adalah penanda fisik tebal-tipisnya awan vs langit biru.

> **Konsekuensi untuk augmentasi:** karena model mengandalkan rasio B/R, augmentasi
> warna dibatasi — `hue=0` pada ColorJitter default dan saturasi hanya 0.9–1.1. Augmentasi
> yang mengubah keseimbangan warna (hue besar, grayscale, equalize) hanya muncul di mode
> `STRONG_AUG` dan perlu diuji dampaknya terhadap sinyal spektral ini.

---

## 6. Reproducibility

- `SEED = 42` lewat `set_seed()` (random, numpy, torch, cuda).
- `seed_worker` + `Generator(manual_seed=SEED)` menjaga urutan sampler/worker konsisten.
- Carve val dari train bersifat **stratified & seeded** (`_stratified_val_indices`),
  jadi split train/val tetap sama di setiap run.

---

## 7. Ringkasan temuan EDA

1. **Dataset tidak seimbang** (8,6 : 1); `7_mixed` sangat minoritas (3,5 %).
2. **Augmentasi geometris** (affine + flip + reflect-pad) menangani variasi
   orientasi/posisi awan tanpa artefak tepi.
3. **Augmentasi fotometrik dibatasi** (hue=0, saturasi sempit) demi menjaga sinyal
   rasio Biru/Merah yang sadar-fisika.
4. **Ketidakseimbangan ditangani di level sampling** (WeightedRandomSampler, ~3.000/kelas)
   dan diperkuat oleh augmentasi acak agar oversampling tidak jadi duplikasi mentah.
5. **Augmentasi kuat tersedia tapi nonaktif** — aktifkan dengan hati-hati karena dapat
   mengganggu fitur spektral.

---

*Dibuat otomatis sebagai dokumentasi EDA. Perbarui bila `config.py` / `data.py` berubah.*
