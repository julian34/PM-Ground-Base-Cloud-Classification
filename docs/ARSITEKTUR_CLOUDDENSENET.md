# Arsitektur model fusi DenseNet lokal dan CloudViT

> **Lingkup:** penelitian `datasetB_laporan_19-11-2026_0251`, diperbarui 19 November 2026.
> Angka hasil mengikuti [ringkasan bukti](HASIL_DATASET_B.md) dan artefak run tersebut.
> Kode digunakan untuk menjelaskan implementasi, bukan menggantikan konfigurasi historis.
> Laporan mencatat masukan **224×224**; `config.py` saat revisi berisi **512** dan tidak
> diperlakukan sebagai bukti bahwa run ini memakai 512. Tidak ada pelatihan ulang dalam revisi ini.

## 1. Model yang mempunyai hasil evaluasi

Model penelitian ini mempunyai dua cabang paralel: backbone DenseNet-121 yang
dimodifikasi dan ekstraktor CloudViT lokal. Keduanya menerima citra RGB yang
sudah dinormalisasi ImageNet. Istilah input `raw` pada kode berarti cabang membaca
citra sebelum stem DenseNet, bukan RGB sebelum normalisasi.

```text
RGB [B,3,224,224] setelah normalisasi
  ├─ DenseNet lokal: stem → block 1–4 + atensi → BN/ReLU → GAP → [B,1024]
  └─ CloudViT: I → L1 → L2 → G1 → G2 → G3 → GAP             → [B,256]
                              concat → [B,1280]
                    1280 → 512 → 128 → 32 → 16 → 7 logit

RGB [0,1] sebelum normalisasi → peta λ → atensi spasial pada cabang DenseNet
```

Head dengan masukan 1.280 fitur didukung struktur checkpoint yang diperiksa pada
revisi jurnal. Head 1.024 fitur adalah varian tanpa cabang global; head 1.216 fitur
adalah varian dengan ViT sederhana 192 fitur. Keduanya bukan arsitektur checkpoint
yang menghasilkan metrik datasetB ini.

## 2. Dimensi cabang DenseNet pada input 224×224

| Tahap | Susunan | Keluaran tanpa dimensi batch |
|---|---|---|
| Stem | Conv 7×7 stride 2, BN, ReLU, max pool 3×3 stride 2 | 64×56×56 |
| Block 1 | 6 DenseLayer, SE, SpatialAttnBR | 256×56×56 |
| Transisi 1 | BN, ReLU, conv 1×1, average pool 2×2 | 128×28×28 |
| Block 2 | 12 DenseLayer, SE, SpatialAttnBR | 512×28×28 |
| Transisi 2 | BN, ReLU, conv 1×1, average pool 2×2 | 256×14×14 |
| Block 3 | 24 DenseLayer, SE, SpatialAttnBR | 1024×14×14 |
| Transisi 3 | BN, ReLU, conv 1×1, average pool 2×2 | 512×7×7 |
| Block 4 | 16 DenseLayer, SE, SpatialAttnBR | 1024×7×7 |
| Akhir | norm5, ReLU, global average pooling | 1024 fitur |

Growth rate adalah 32. Dalam tiap DenseLayer, BN/ReLU/conv 1×1 menghasilkan
bottleneck 128 kanal. Setelah BN/ReLU dan SE, jalur SCConv menghasilkan 24 kanal,
sedangkan jalur konvolusi `pq` + softplus menghasilkan delapan kanal rasio
`(P−Q)/(P+Q+0.001)`. Konkatenasi keduanya menambah 32 kanal per layer.
ND merupakan rasio fitur yang dipelajari; bukan pengukuran spektral langsung.

## 3. Atensi dan konvolusi lokal

| Komponen | Implementasi terdokumentasi | Batas interpretasi |
|---|---|---|
| SE | GAP → conv 1×1 → ReLU → conv 1×1 → sigmoid, reduksi 16 | Pembobotan kanal; kontribusi akurasinya belum diablasi |
| SpatialAttnBR | Mean dan maksimum lintas kanal, ditambah peta λ; conv 7×7 → sigmoid | Petunjuk warna, bukan segmentasi awan yang sudah divalidasi |
| SCConv | Normalisasi respons konvolusi dengan norma patch/kernel, q dan pangkat tau terlatih | Bukan jaminan invariansi pencahayaan atau sensor |
| Peta λ | Selisih B−R dibagi B+R, dikurangi median valid, lalu tanh | Dihitung sebelum normalisasi ImageNet |

SE dipasang pada bottleneck setiap DenseLayer dan setelah setiap dense block.
SpatialAttnBR mengikuti SE setelah dense block. Peta λ diubah ukurannya dan
dikalikan skala 10 pada atensi menurut kode. Rumus terperinci ada di
[PENJELASAN_MODEL_DETAIL.md](PENJELASAN_MODEL_DETAIL.md).

## 4. Cabang CloudViT dan kepala bersama

CloudViT lokal menghasilkan urutan fitur 16×112×112 → 32×56×56 → 64×28×28 →
128×14×14 → 192×7×7 → 256×4×4 → GAP 256 fitur. Detail modul dan perbedaannya
dari ViT sederhana ada di [CABANG_VIT_DETAIL.md](CABANG_VIT_DETAIL.md).

```text
Concat(GAP(DenseNet), GAP(CloudViT)) = 1280 fitur
Linear(1280,512) → BN1d → ReLU
Linear(512,128)  → BN1d → ReLU
Linear(128,32)   → BN1d → ReLU
Linear(32,16)    → BN1d → ReLU
Dropout(0.8) → Linear(16,7)
```

Dropout 0,8 adalah nilai pada laporan, bukan nilai yang dapat dibaca dari tensor
bobot. Keluaran merupakan logit; kelas dipilih dengan argmax. Softmax dipakai
untuk probabilitas, bukan lapisan tambahan sebelum CrossEntropy pada focal loss.

## 5. Pelatihan dan status pembanding

Backbone menerima bobot DenseNet-121 ImageNet yang dipetakan/disesuaikan ke modul
lokal. SCConv, ND, atensi, cabang CloudViT dan head membuat fungsi model berbeda
dari DenseNet-121 asli; tidak diklaim identik dengan model ImageNet pada inisialisasi.
AdamW, focal loss, tujuh tahap, dan durasi dijelaskan pada
[HASIL_DATASET_B.md](HASIL_DATASET_B.md).

Empat rancangan pada dokumen penelitian mencakup DenseNet-121, DenseNet lokal,
Transformer tunggal dan model fusi. Folder hasil yang ditetapkan hanya berisi
evaluasi model fusi. Tidak ada nilai baseline atau ablasi yang ditambahkan.
Jumlah parameter dan dimensi adalah uraian struktur, bukan hasil uji akurasi,
waktu inferensi, atau penggunaan memori.
