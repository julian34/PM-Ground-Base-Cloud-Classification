# Rencana Penelitian Tesis (S2 / KKNI Level 8)

**Topik:** Klasifikasi genus awan berbasis darat yang sadar-fisika dan sadar-ketidakseimbangan
menggunakan fusi CNN–Transformer (CloudDenseNet + CloudViT), dengan fokus pada reduksi
kebingungan antar-genus mirip (stratocumulus ↔ cumulonimbus).

> **Status dokumen:** rencana (belum eksperimen). Disusun dari sesi *grilling* keputusan
> penelitian. Sumber kebenaran hyperparameter tetap `src/clouddensenet/config.py`.
> Dokumen ini adalah kontrak metodologis — bukan laporan hasil.
>
> **Catatan kritis:** draf awal (`laporan/.../draf-klasifikasi-awan.pdf`) adalah *pilot/null
> report* — satu model, satu run, satu split non-standar (`datasetB`), tanpa baseline terlatih,
> tanpa ablasi, tanpa uji signifikansi. Rencana ini adalah program untuk menaikkannya ke
> standar tesis S2.

## ⚠️ Revisi lingkup — 24 November 2026 (keputusan peneliti)

Lingkup penelitian **dibatasi pada rekonstruksi dan implementasi** agar DenseNet-121 dan
(Cloud)ViT dapat dijalankan, **serta data yang digunakan**. Tujuan yang berlaku:

| Kode | Tujuan |
|------|--------|
| **T1** | Rekonstruksi CloudDenseNet dari notebook Colab menjadi paket `clouddensenet` yang berjalan lokal (GPU/CPU) |
| **T2** | Implementasi DenseNet-121 termodifikasi dan cabang CloudViT: dapat dijalankan sendiri-sendiri dan sebagai fusi |
| **T3** | Penyiapan data GCD/datasetB dan pipeline masukan |
| **T4** | Pelatihan dan evaluasi model fusi sampai selesai (run datasetB, 80 epoch) |

**Di luar lingkup → pekerjaan lanjutan:** RQ1 (penurunan konfusi Sc↔Cb), RQ2 (ablasi), RQ3 (biaya),
baseline multi-seed, audit kebocoran, split resmi, uji lintas lokasi/kamera. Bagian §4–§8 di bawah
dipertahankan sebagai **arsip rencana lanjutan**, bukan kontrak penelitian saat ini.
Laporan yang mengikuti lingkup ini: `laporan/ai-writing-space/papers/laporan-desain-eksperimen/`.

---

## 1. Sub-bidang & Pernyataan Masalah (Problem Statement)

### 1.1 Sub-bidang komputasional spesifik
*Fine-Grained Visual Classification* (FGVC) untuk citra awan berbasis darat **di bawah
distribusi kelas long-tailed**, sub-area *image classification / representation learning* dalam
Computer Vision, dengan penekanan pada **fusi fitur CNN–Transformer yang sadar-fisika
(physics-informed)**.

Bukan "klasifikasi citra langit" generik: akurasi dasar pengenalan awan sudah tinggi
(~90%+), sehingga masalah bernilai bukan "mengenali awan" melainkan **membedakan genus yang
mirip secara visual-fisis**.

### 1.2 Masalah utama (tunggal & terukur)
> Bagaimana mengurangi **kebingungan antar-genus pada kelas awan yang mirip secara visual-fisis**
> (khususnya kelompok konvektif rendah: stratocumulus–cumulonimbus–altocumulus) pada kondisi
> **distribusi kelas long-tailed**, **tanpa mengorbankan** akurasi kelas mayoritas?

**Target case terukur:** pasangan **stratocumulus ↔ cumulonimbus**. Pada pilot, pasangan ini
menyumbang **61,11%** dari seluruh kesalahan uji; recall stratocumulus hanya 0,65–0,69, di
tengah ketidakseimbangan ekstrem (cumulonimbus 4.035 vs mixed 668 pada data latih `datasetB`).
Masalah ini **diakui literatur** sebagai konfusi sulit yang bersumber dari *visual overlap*,
bukan sekadar kurang label.

### 1.3 Jenis & unit kontribusi
**Kontribusi metode** (bukan studi empiris murni, bukan implementasi sistem). Inti kebaruan =
**mekanisme physics-informed + sadar-imbalance yang dirancang spesifik untuk memisahkan pasangan
genus mirip**, dibuktikan pada **sel confusion matrix** dan **recall kelas sulit** — bukan hanya
akurasi agregat.

---

## 2. Research Gap vs State-of-the-Art (2021–2026)

### 2.1 Posisi SOTA pada GCD (split resmi 10.000/9.000)
| Metode | Venue / Tahun | Akurasi (klaim) | Keluarga |
|--------|---------------|-----------------|----------|
| Liu et al. (baseline CGAT) | IEEE TGRS 2022 | 84,30% | CNN/graph |
| CSSL (self-supervised) | Remote Sensing 2022 | 84,62% | CNN |
| CloudDenseNet *(induk arsitektur)* | Sensors 2023 | 93,43% | CNN (DenseNet rekonstruksi) |
| Transformer-on-GCD | Remote Sensing 2022 | 93,57% | ViT |
| ALGA-DenseNet | PLOS One 2025 | 97,94% | CNN multi-skala |

**Temuan kunci:** kluster angka **93–98% belum tereplikasi independen** dan memakai protokol
berbeda-beda. Tren bidang: CNN → ViT → hybrid CNN–ViT.
Backbone DenseNet+SE+atensi+ViT tergolong **mainstream** — tidak novel dengan sendirinya.

### 2.2 Gap (tiga bagian)
1. **Fokus-masalah** — metode GCD yang ada mengoptimalkan **akurasi agregat**; tidak ada yang
   merancang mekanisme yang **secara eksplisit menargetkan & mengukur** reduksi kebingungan
   antar-genus mirip (Sc↔Cb) di bawah long-tailed.
2. **Representasi fisika** — isyarat biru–merah (B/R, B−R) **mapan untuk deteksi/segmentasi**
   awan (Long 2006; Heinle 2010; Dev dkk.), tetapi **belum dimanfaatkan sebagai atensi
   sadar-fisika untuk klasifikasi genus** yang terbukti menyasar pasangan sulit.
3. **Rigor evaluasi** — klaim keunggulan pada GCD umumnya **single-run, tanpa ablasi terisolasi,
   tanpa uji signifikansi**, dan angka tinggi belum tereplikasi independen.

### 2.3 Sikap terhadap klaim
Klaim utama **bukan** "SOTA akurasi agregat". Klaim utama =
**(a)** reduksi terukur & **signifikan** pada sel konfusi Sc↔Cb dan recall minoritas,
**(b)** pada **split resmi GCD** yang dapat dibandingkan,
**(c)** dengan **kontribusi tiap komponen terisolasi**.
Akurasi agregat dilaporkan apa adanya sebagai konteks, bukan sebagai kemenangan. Frasa jujur
"setara/menyamai" dipakai saat uji tidak signifikan.

---

## 3. Bentuk Novelty (tepat 2 mekanisme)

| Sumbu | Mekanisme | Status klaim |
|-------|-----------|--------------|
| **Representasi** | **Atensi terpandu-fisika** (pengembangan SpatialAttnBR): bukan sekadar kanal rasio λ biru–merah, melainkan rancangan atensi yang menyorot wilayah pembeda genus mirip | **Novelty primer** |
| **Optimisasi** | **Loss sadar-konfusi/imbalance**: *pair-margin* / *logit-adjustment* long-tailed di atas focal, menghukum eksplisit kesalahan pasangan sulit & kelas minoritas | **Novelty sekunder** |
| Representasi | SEBlock, SCConv, cabang CloudViT | **Diadopsi** dari literatur — diuji lewat ablasi, **tidak diklaim** sebagai kebaruan |

Novelty **berhenti di dua mekanisme** untuk menghindari kritik "assembly of known parts" dan
menjaga lingkup tesis terkendali. Keduanya **isolabel** di ablasi dan terikat langsung ke
masalah Q2.

**Risiko yang diantisipasi:** fitur B/R adalah cue deteksi berumur puluhan tahun dengan
*confound aerosol/haze* (RBR juga naik oleh haze). Maka klaim fisika hanya bertahan bila ablasi
membuktikan reduksi konfusi Sc↔Cb yang signifikan; confound haze wajib diakui & diuji. Jika
atensi-fisika terbukti tak signifikan, inti klaim bergeser ke loss (narasi cadangan sah).

---

## 4. Pertanyaan Penelitian (RQ)

> **Revisi 24 November 2026 (keputusan peneliti):** RQ1 difokuskan pada penurunan konfusi
> Sc↔Cb; RQ3 (biaya) dikeluarkan dari lingkup.

- **RQ1 (Efektivitas):** Apakah model usulan **menurunkan konfusi stratocumulus–cumulonimbus**
  dibanding baseline pada **split resmi GCD**? Ukuran utama = laju konfusi berarah Sc→Cb dan
  Cb→Sc, dengan **signifikansi statistik** (multi-seed + uji berpasangan). Macro-F1 & recall
  kelas lain = metrik pengaman (penurunan konfusi tak boleh mengorbankan kelas lain).
- **RQ2 (Atribusi / Ablasi):** Berapa **kontribusi terisolasi** tiap komponen (atensi-fisika,
  loss; serta SE/SCConv/ViT yang diadopsi) lewat ablasi terkontrol?
- ~~**RQ3 (Biaya)**~~ — **di luar lingkup** (revisi 24 Nov 2026). Jumlah parameter hanya
  dilaporkan sebagai deskripsi arsitektur dan sebagai kontrol ablasi CloudViT, tanpa klaim efisiensi.

---

## 5. Dataset, Split & Integritas

### 5.1 Protokol split
- **Protokol pelaporan utama = split resmi GCD 10.000/9.000.** Dalam repo ini diwujudkan
  sebagai `N_SPLIT = (8340, 1471, 9000)` ([config.py:31](src/clouddensenet/config.py#L31)):
  **test 9.000 utuh** (sebanding dengan semua publikasi), validasi 1.471 di-carve stratified &
  seeded **dari train** (tanpa kebocoran test).
- **`datasetB` (13.299/3.800/1.901) didemosikan** — re-split 70/20/10 dari gabungan train+test
  resmi, sehingga **tidak sebanding** dengan angka publikasi mana pun. Dipertahankan hanya untuk
  transparansi riwayat / analisis sekunder berlabel jelas; **bukan** basis klaim.

### 5.2 Rekonsiliasi wajib
8.340 + 1.471 = **9.811 ≠ 10.000** → ada **189 citra** yang harus dijelaskan di
`DATASET_SPLIT.md` (corrupt / dibuang / duplikat). Test harus terbukti tetap 9.000 utuh.

### 5.3 Audit kebocoran (gerbang wajib pra-eksperimen)
1. **Duplikat persis** — hash berkas (SHA-256); hash sama lintas split = bocor.
2. **Near-duplicate perseptual** — `imagehash` (pHash/dHash), jarak Hamming lintas split
   (ambang mis. ≤5 bit), tinjau manual.
3. **Near-duplicate semantik** — embedding encoder pralatih (mis. DINO), cosine-similarity,
   tetangga terdekat lintas split.
4. **Isolasi test** — assert nama berkas test tak pernah di train/val; **seleksi checkpoint
   hanya pada val**.
5. **Reprodusibilitas** — simpan **manifest split** (daftar berkas per split + seed) sebagai
   berkas yang ikut tesis.
6. **Lapor** — tabel #duplikat-persis, #near-dupe, ambang, #dibuang (bukti bagian gap "rigor").

### 5.4 Catatan taksonomi
GCD sudah menggabung genus (Sc/St/As; Cb/Ns) ke dalam 7 kelas; `Mixed` sangat kecil & tersulit.
Oleh karena itu **macro-F1 & balanced accuracy** wajib dilaporkan, bukan hanya top-1.

---

## 6. Desain Eksperimen & Validasi

### 6.1 Matriks baseline (dilatih ulang, protokol identik, ≥5 seed)
| Kode | Baseline | Peran |
|------|----------|-------|
| A | DenseNet-121 polos (ImageNet) | *lower anchor* |
| B | Reproduksi CloudDenseNet | **pembanding kunci** (induk arsitektur) |
| C | ViT/hybrid (ViT-B atau CloudViT mandiri) | wakil tren Transformer |
| **M** | **Model usulan**: DenseNet-SE-SCConv + atensi-fisika + loss sadar-konfusi | metode tesis |

> Jika M jauh di atas publikasi, wajib ada penjelasan (bukan kebocoran). Jika reproduksi B jauh
> dari 93,43%, selisih didokumentasikan (memperkuat gap "rigor").

### 6.2 Rancangan ablasi (mengisolasi + kendalikan confound sampler↔loss)
- Base tetap: DenseNet + {SE, SCConv, ViT adopsi} + focal + sampler.
- Matriks **2×2**: `{atensi-fisika on/off} × {loss-konfusi on/off}`.
- **Isolasi imbalance terpisah**: `{sampler vs tanpa-sampler}` saat mengukur kontribusi loss —
  **jangan** sampler + loss aktif bersamaan tanpa kontrol (`IMBALANCE` di
  [config.py:32](src/clouddensenet/config.py#L32)).
- Ablasi komponen adopsi (SE/SCConv/ViT) untuk kelengkapan.
- Tiap sel **multi-seed**.

### 6.3 Metrik & uji signifikansi
- **Metrik primer (RQ1):** **laju konfusi per-pasangan** (Sc→Cb, Cb→Sc).
  **Metrik pengaman:** macro-F1 · balanced accuracy · recall tiap kelas. Akurasi agregat = konteks saja.
- **Signifikansi:** ≥5 seed (mean±std) · **McNemar** berpasangan (model vs baseline, test sama)
  · uji berpasangan atas macro-F1 level-seed · **koreksi Holm** lintas sel ablasi · **bootstrap
  CI** pada test.
- **Aturan klaim:** keunggulan ditulis hanya bila interval terpisah **dan** uji signifikan;
  jika tidak → "setara/menyamai".

---

## 7. Roadmap Bertahap (7 Fase)

| Fase | Fokus | Kriteria selesai |
|------|-------|------------------|
| **0** | Fondasi data & integritas | Split dapat diaudit; kebocoran lintas-split = 0 (terbukti); manifest+seed tersimpan; matriks flag config eksplisit |
| **1** | Baseline terlatih (A–C) | Tabel baseline mean±std; reproduksi B terdokumentasi |
| **2** | Novelty 1: atensi-fisika | Mekanisme diformalkan (melampaui kanal B/R); confound haze diuji; konfusi Sc↔Cb turun **signifikan** vs base |
| **3** | Novelty 2: loss sadar-konfusi | Kontribusi loss terisolasi (confound sampler dikontrol) & signifikan pada recall minoritas |
| **4** | Ablasi penuh (RQ2) | Setiap klaim komponen didukung ablasi + signifikansi (Holm) |
| ~~**5**~~ | ~~Analisis biaya (RQ3)~~ | **Di luar lingkup** (revisi 24 Nov 2026) |
| **6** | Validasi statistik & penulisan | RQ1–RQ2 terjawab dengan signifikansi; Grad-CAM pada kasus Sc↔Cb; draf ditulis ulang dengan klaim jujur |

**Higiene config (Fase 0):** [config.py:50](src/clouddensenet/config.py#L50) saat ini
`USE_BRANCH=False` — konfigurasi split-resmi belum menyalakan CloudViT. Tetapkan secara eksplisit
varian mana menyalakan komponen apa sebelum melatih.

---

## 8. Risiko & Mitigasi
1. **Biaya komputasi** — 5 seed × (3 baseline + ~6 sel ablasi + varian) pada RTX 2060 SUPER 8GB
   itu berat. **Mitigasi:** 3 seed untuk eksplorasi ablasi, **5 seed hanya untuk perbandingan
   headline**; manfaatkan AMP; jadwalkan bertahap.
2. **Reproduksi CloudDenseNet gagal** — temuan sah (memperkuat gap "rigor"), tetapi dibingkai
   jujur.
3. **Novelty fisika ternyata kecil** — inti klaim bergeser ke loss; narasi cadangan sudah
   diantisipasi (§3).
4. **Over-claim B/R** — akui confound aerosol/haze; klaim hanya bertahan bila terikat reduksi
   Sc↔Cb terukur & signifikan.

---

## 9. Referensi (perlu verifikasi sitasi sebelum proposal formal)
> ⚠️ Angka 93–98% masing-masing memakai protokol berbeda dan **belum di-run ulang** di sini.
> Beberapa sitasi di literatur tidak konsisten (mis. dataset GCD). Verifikasi via tool pemeriksa
> referensi / akses primer sebelum dimasukkan ke proposal.

- **GCD dataset** — Liu S., Duan L., Zhang Z., Cao X., Durrani T.S., "Ground-Based Remote Sensing
  Cloud Classification via Context Graph Attention Network," *IEEE TGRS*, 2022,
  doi:10.1109/TGRS.2021.3063255. Rilis: `github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset`
  (19.000 citra, 10.000 train / 9.000 test, 7 kelas).
- **CloudDenseNet** (induk) — Wang dkk., *Sensors* 2023, 23(18):7957. doi:10.3390/s23187957.
- **Transformer-on-GCD** — *Remote Sensing* 2022, 14(16):3978.
- **ALGA-DenseNet** — *PLOS One* 2025, doi:10.1371/journal.pone.0333999.
- **CSSL** — *Remote Sensing* 2022, 14(22):5821.
- **Isyarat B/R** — Long dkk. 2006 (ambang R/B); Heinle dkk. 2010 (R−B); Dev dkk. (studi
  color-space).
- **Konfusi konvektif** — Kopeć dkk., *QJRMS* 2024, doi:10.1002/qj.4865.

---

*Dokumen ini disusun dari kesepahaman bersama hasil sesi grilling. Perubahan keputusan besar
harus memperbarui dokumen ini terlebih dahulu sebelum diterapkan ke kode/eksperimen.*
