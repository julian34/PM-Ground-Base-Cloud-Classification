# Stratified K-Fold Cross-Validation — Studi Kasus GCD (7 kelas awan)

Penjelasan konsep **Stratified K-Fold Cross-Validation** (validasi silang berlapis)
dan **penerapannya** pada proyek CloudDenseNet: dataset citra langit GCD dengan
**7 kelas awan yang tidak seimbang**.

Dokumen ini dua lapis:
1. **Konsep** — apa, kenapa, dan kapan dipakai (ringan, bisa dibaca tanpa kode).
2. **Penerapan** — kode konkret yang nyambung ke `src/clouddensenet/` (bukan pseudo-code).

> Sumber kebenaran kode: `src/clouddensenet/data.py` (`build_loaders`,
> `_stratified_val_indices`), `src/clouddensenet/config.py` (`SEED`, `VAL_SPLIT`,
> `IMBALANCE`, `PHASES`, `GROUPED_SPLIT`), dan `scripts/train.py` (loop training).
> Angka dataset diambil dari `docs/EDA_AUGMENTASI.md` (split `data/processed/` di mesin ini).

---

## 1. Masalah: kenapa satu kali split tidak cukup

Cara paling umum mengevaluasi model adalah **holdout** — potong data sekali jadi
train/val/test. Proyek ini memang memakai holdout: `build_loaders()` mengiris 15 %
dari `train/` sebagai validasi (`VAL_SPLIT = 0.15`, stratified, `seed=42`).

Holdout punya dua kelemahan:

1. **Estimasi berisik (high variance).** Skor val/F1 bergantung pada *sampel mana
   yang kebetulan masuk val*. Ganti seed → skor bisa bergeser beberapa poin. Kita
   tidak tahu apakah "F1 0.91" itu nyata atau cuma keberuntungan split.
2. **Boros data.** 15 % data "hanya" dipakai untuk validasi, tidak pernah ikut
   melatih model yang dievaluasi.

**K-Fold Cross-Validation** menjawab keduanya: data latih dibagi jadi **K bagian
(fold)** berukuran sama. Model dilatih **K kali**; tiap putaran satu fold jadi
validasi dan K−1 sisanya jadi latih. Setiap sampel dipakai sebagai validasi **tepat
sekali**. Hasilnya K skor → dilaporkan sebagai **rata-rata ± simpangan baku**.

```
K=5, data latih dibagi 5 fold:

Putaran 1:  [ VAL ][train][train][train][train]
Putaran 2:  [train][ VAL ][train][train][train]
Putaran 3:  [train][train][ VAL ][train][train]
Putaran 4:  [train][train][train][ VAL ][train]
Putaran 5:  [train][train][train][train][ VAL ]
                  ↓
   5 skor F1  →  F1 = mean ± std  (estimasi + ketidakpastiannya)
```

---

## 2. Kenapa harus *Stratified* (berlapis)?

K-Fold biasa membagi data **acak**. Untuk dataset seimbang itu aman. Tapi dataset
GCD **sangat timpang**:

| Idx | Kelas             | Jumlah (train) | Porsi   |
|----:|-------------------|---------------:|--------:|
| 0   | 1_cumulus         |            659 |  7,8 %  |
| 1   | 2_altocumulus     |            616 |  7,2 %  |
| 2   | 3_cirrus          |            980 | 11,5 %  |
| 3   | 4_clearsky        |          1.828 | 21,5 %  |
| 4   | 5_stratocumulus   |          1.569 | 18,5 %  |
| 5   | 6_cumulonimbus    |          2.553 | 30,0 %  |
| 6   | 7_mixed           |            296 |  3,5 %  |
|     | **Total**         |      **8.501** | 100 %   |

Rasio ketimpangan ≈ **8,6 : 1** (`6_cumulonimbus` vs `7_mixed`).

Bahaya K-Fold **acak** pada kelas minoritas `7_mixed` (296 citra):

- Dengan K=5, idealnya tiap fold validasi berisi ≈ 59 citra `7_mixed`.
- Secara acak, satu fold bisa kebetulan dapat 40, fold lain 78. Bahkan — pada kelas
  yang lebih kecil lagi — sebuah fold bisa **nyaris tanpa** sampel kelas itu.
- Akibatnya **F1 per fold jadi liar**: fold yang miskin `7_mixed` memberi estimasi
  kelas minoritas yang tidak bisa dipercaya, dan std antar-fold membengkak.

**Stratified K-Fold** memaksa **proporsi tiap kelas sama di setiap fold** seperti di
populasi penuh. Jadi setiap fold validasi "mewakili" seluruh dataset secara mini.

### Distribusi yang dijamin Stratified 5-Fold

Tiap fold validasi (≈ 1.700 citra) selalu berisi kira-kira:

| Kelas            | Per fold (≈ 1/5) |
|------------------|-----------------:|
| 1_cumulus        |             ~132 |
| 2_altocumulus    |             ~123 |
| 3_cirrus         |             ~196 |
| 4_clearsky       |             ~366 |
| 5_stratocumulus  |             ~314 |
| 6_cumulonimbus   |             ~511 |
| 7_mixed          |              ~59 |

Konsisten di kelima fold → F1 antar-fold **lebih stabil** dan estimasinya **jujur**
untuk semua kelas, termasuk minoritas.

> **Analogi:** mau menebak rata-rata nilai satu sekolah. K-Fold acak = ambil 5 kelompok
> siswa sembarang (bisa satu kelompok kebetulan isi anak IPA semua). Stratified =
> tiap kelompok dijaga punya proporsi IPA/IPS/Bahasa yang sama seperti sekolah.

---

## 3. Posisi Stratified K-Fold di proyek ini

Yang **sudah** ada di repo adalah **stratified holdout** (satu kali), bukan K-fold:

```python
# src/clouddensenet/data.py — ini "stratified 1-fold" (holdout), bukan K-fold
def _stratified_val_indices(targets, val_frac, seed=SEED):
    for c in np.unique(targets):          # per kelas...
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))   # ...iris val_frac
        val_idx.extend(idx[:n_val]); train_idx.extend(idx[n_val:])
```

Logika "iris per kelas" ini **persis** inti stratifikasi. Stratified K-Fold tinggal
mengulang ide itu K kali dengan irisan yang saling lepas (`StratifiedKFold` dari
scikit-learn melakukannya untuk kita). **Test set tetap dipegang terpisah** sebagai
holdout final — K-fold hanya dijalankan di dalam pool `train/`.

**Kapan pakai yang mana:**

| Kebutuhan                                            | Pakai              |
|------------------------------------------------------|--------------------|
| Iterasi cepat, prototyping, 1 run training           | Holdout (`build_loaders`) |
| Laporan metrik yang kredibel (skripsi/paper): mean±std | **Stratified K-Fold** |
| Pilih hyperparameter / bandingkan arsitektur         | **Stratified K-Fold** |
| Model produksi final                                 | Retrain di **seluruh** `train/` |

> ⚠️ **Biaya komputasi.** Jadwal training di sini bertahap sampai **80 epoch**
> (`PHASES`). K=5 berarti **5 × training penuh**. Untuk CV, pertimbangkan K=5 dengan
> jadwal dipangkas (mis. `EPOCH_LIMIT`) atau K=3. CV untuk *memilih* konfigurasi;
> model akhir dilatih sekali di seluruh data.

---

## 4. Penerapan — drop-in untuk CloudDenseNet

Pendekatan: `StratifiedKFold` scikit-learn menghasilkan indeks; indeks dipakai untuk
`Subset` di atas `ImageFolder` — **pola yang sama** seperti `build_loaders()` yang
sudah ada (dua `ImageFolder` pada folder yang sama: satu transform augmentasi untuk
train, satu transform polos untuk val; keduanya urut `sorted()` sehingga indeks
sejajar).

### 4.1 Loader per-fold (adaptasi `build_loaders`)

```python
# src/clouddensenet/data_kfold.py
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from .config import (BATCH_SIZE, IMBALANCE, NUM_CLASSES, NUM_WORKERS,
                     SAMPLES_PER_EPOCH, SEED)
from .data import build_transforms, seed_worker


def base_targets(data_root):
    """Target kelas seluruh pool train/ (urutan ImageFolder = sorted, deterministik)."""
    base = ImageFolder(root=str(data_root / "train"))   # tanpa transform: cepat, cuma label
    return np.asarray(base.targets), base.classes


def build_fold_loaders(data_root, device, train_idx, val_idx):
    """Bangun train/val loader untuk SATU fold dari indeks StratifiedKFold.

    Meniru build_loaders(): train_tf untuk latih, val_tf untuk validasi, pada folder
    train/ yang sama. WeightedRandomSampler dihitung ULANG dari distribusi train fold ini.
    """
    train_tf, val_tf = build_transforms()
    g = torch.Generator().manual_seed(SEED)
    pin = (device.type == "cuda")

    base_train = ImageFolder(root=str(data_root / "train"), transform=train_tf)
    base_val   = ImageFolder(root=str(data_root / "train"), transform=val_tf)

    train_ds = Subset(base_train, train_idx.tolist())
    val_ds   = Subset(base_val,   val_idx.tolist())
    val_ds.classes = base_val.classes
    train_targets = np.asarray(base_train.targets)[train_idx]

    if IMBALANCE == "sampler":
        cnt = np.bincount(train_targets, minlength=NUM_CLASSES)
        w = torch.as_tensor(1.0 / cnt[train_targets], dtype=torch.double)
        kw = dict(sampler=WeightedRandomSampler(w, SAMPLES_PER_EPOCH, replacement=True, generator=g))
    else:
        kw = dict(shuffle=True, generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True,
                              worker_init_fn=seed_worker, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS > 0, **kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)
    return train_loader, val_loader, train_ds, val_ds
```

### 4.2 Skrip CV (adaptasi `scripts/train.py`)

```python
# scripts/train_kfold.py
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C
from clouddensenet import (CloudDenseNet, FocalLoss, apply_phase, catat_epoch, evaluate,
                           muat_imagenet, nolkan_input_nd, phase_of, set_seed,
                           train_one_epoch)
from clouddensenet.engine import is_finite
from clouddensenet.data_kfold import base_targets, build_fold_loaders

N_FOLDS = 5
EPOCHS_CV = C.EPOCHS          # pangkas (mis. 30) untuk hemat waktu CV


def train_one_fold(fold, data_root, device, train_idx, val_idx):
    """Satu training penuh pada satu fold. Kembalikan best_f1 + prediksi out-of-fold."""
    set_seed(C.SEED)          # model & sampler identik tiap fold -> yang beda hanya data split
    train_loader, val_loader, _, _ = build_fold_loaders(data_root, device, train_idx, val_idx)

    model = nolkan_input_nd(muat_imagenet(CloudDenseNet().to(device)))
    criterion = FocalLoss() if C.FOCAL_GAMMA > 0 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt = C.CKPT_DIR / f"{C.RUN_NAME}_fold{fold}{C.VERSION}.pth"   # nama per-fold: jangan bentrok
    best_f1, current_phase = -1.0, 0

    for epoch in range(1, EPOCHS_CV + 1):
        phase = phase_of(epoch)
        if phase != current_phase:
            apply_phase(model, optimizer, phase); current_phase = phase
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)
        if not is_finite(tr_loss, va_loss):
            print(f"[fold {fold}] NaN/Inf -> stop"); break
        print(f"[fold {fold}] ep {epoch:02d}/{EPOCHS_CV} | val f1 {va_f1:.4f} acc {va_acc:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_f1": best_f1}, ckpt)
    return best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not C.DEBUG else "cpu")
    data_root = C.DATA_ROOT
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    targets, classes = base_targets(data_root)
    print(f"Pool train: {len(targets)} citra | {len(classes)} kelas")

    # shuffle=True + random_state=SEED  -> fold deterministik & reproducible (konsisten SEED=42 repo)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED)
    X_dummy = np.zeros(len(targets))      # StratifiedKFold hanya butuh label (y) untuk stratifikasi

    fold_f1 = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, targets)):
        # --- bukti stratifikasi: distribusi val fold mirip populasi ---
        dist = np.bincount(targets[va_idx], minlength=len(classes))
        print(f"\n=== Fold {fold} | val={len(va_idx)} train={len(tr_idx)} | dist val: {dist.tolist()}")
        t0 = time.time()
        f1 = train_one_fold(fold, data_root, device, tr_idx, va_idx)
        fold_f1.append(f1)
        print(f"=== Fold {fold} selesai | best val f1 {f1:.4f} | {time.time()-t0:.0f}s")

    arr = np.array(fold_f1)
    print("\n" + "=" * 52)
    print(f"Hasil {N_FOLDS}-Fold Stratified CV (macro-F1):")
    for i, f in enumerate(arr):
        print(f"  fold {i}: {f:.4f}")
    print(f"  ----------------------------------")
    print(f"  F1 = {arr.mean():.4f} ± {arr.std():.4f}")   # <- angka yang dilaporkan


if __name__ == "__main__":
    main()
```

### 4.3 Menjalankan

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_kfold.py
```

Keluaran kunci — perhatikan `dist val` hampir identik tiap fold (bukti stratifikasi
bekerja), lalu baris terakhir:

```
F1 = 0.9xxx ± 0.0xxx     <- rata-rata ± simpangan baku lintas 5 fold
```

---

## 5. Hal penting saat menerapkan di repo ini

1. **Test set jangan disentuh.** K-fold berjalan **hanya** di pool `train/`.
   `data/processed/test/` tetap holdout final, dievaluasi sekali di akhir pakai
   model terbaik. Ini mencegah kebocoran optimistis.

2. **SEED konsisten.** `StratifiedKFold(shuffle=True, random_state=C.SEED)` memakai
   `SEED = 42` yang sama dengan repo → fold reproducible, selaras `set_seed()`.

3. **Sampler dihitung ulang per fold.** `WeightedRandomSampler` (`IMBALANCE="sampler"`)
   bergantung pada distribusi train; pada tiap fold distribusinya beda sedikit, jadi
   `build_fold_loaders` menghitung `bincount` dari indeks train fold itu — **jangan**
   pakai bobot global.

4. **Model & optimizer fresh tiap fold.** Panggil `set_seed()` lalu bangun
   `CloudDenseNet()` + `muat_imagenet()` + optimizer baru di awal setiap fold. Jangan
   membawa bobot antar-fold (itu kebocoran). Checkpoint pakai nama per-fold
   (`..._fold0...pth`) supaya tidak saling menimpa / ter-resume keliru.

5. **Augmentasi hanya di train.** Sudah terjaga: `build_fold_loaders` memakai
   `train_tf` untuk subset latih dan `val_tf` (polos) untuk subset validasi — identik
   dengan pola `build_loaders`.

6. **Biaya.** 5 fold × 80 epoch itu mahal. Untuk CV: pangkas `EPOCHS_CV` (mis. 30),
   atau turunkan ke K=3. Model final tetap dilatih sekali di seluruh `train/` dengan
   jadwal penuh `PHASES`.

### 5.1 Variasi lanjutan (opsional)

- **Out-of-fold (OOF) confusion matrix.** Karena tiap sampel jadi validasi tepat
  sekali, kumpulkan prediksi val tiap fold → gabung → **satu confusion matrix atas
  8.501 citra**. Lebih kaya dari CM holdout. (Tambahkan pengumpulan `y_true, y_pred`
  di `train_one_fold` lalu `plot_confusion_matrix` versi array.)

- **StratifiedGroupKFold — jika ada citra nyaris-duplikat.** Config punya
  `GROUPED_SPLIT`/`GROUP_THR = 0.95`, pertanda ada kekhawatiran citra beruntun/mirip
  (burst dari langit yang sama) bocor antar split. Bila itu nyata, stratifikasi saja
  tak cukup: pakai `StratifiedGroupKFold` agar **satu grup tidak terpecah** ke train
  dan val sekaligus — menjaga proporsi kelas **dan** mencegah kebocoran kemiripan.

- **Repeated Stratified K-Fold.** Ulang CV dengan beberapa seed berbeda
  (`RepeatedStratifiedKFold`) untuk estimasi std yang lebih halus — jika anggaran
  komputasi memungkinkan.

---

## 6. Ringkasan

- **K-Fold CV** = latih K kali, tiap sampel jadi validasi sekali → estimasi
  **mean ± std**, bukan satu angka rapuh.
- **Stratified** = proporsi 7 kelas dijaga sama di tiap fold; **wajib** untuk GCD yang
  timpang 8,6 : 1, terutama demi kelas minoritas `7_mixed` (3,5 %).
- Repo sudah punya **stratified holdout** (`_stratified_val_indices`); K-fold adalah
  perluasan alaminya via `sklearn.StratifiedKFold` + `Subset` di atas `ImageFolder`.
- **Test set tetap holdout**; CV untuk laporan metrik & pemilihan konfigurasi; model
  produksi dilatih ulang di seluruh data latih.
```
