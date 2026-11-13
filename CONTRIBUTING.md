# Panduan Kolaborasi — CloudDenseNet

Panduan ini mengatur **cara berkolaborasi di GitHub** untuk repo
[`PM-Ground-Base-Cloud-Classification`](https://github.com/julian34/PM-Ground-Base-Cloud-Classification).
Tujuannya satu: menjaga `main` **selalu bisa dilatih** (training jalan, `pytest` hijau) sambil
banyak orang bekerja paralel.

> Baru pertama kali setup? Baca dulu **[`docs/Panduan_Instalasi_Clone.md`](docs/Panduan_Instalasi_Clone.md)**
> lalu kembali ke sini. Aturan arsitektur & konvensi kode ada di **[`CLAUDE.md`](CLAUDE.md)**.

---

## Daftar isi

1. [Prinsip utama](#1-prinsip-utama)
2. [Alur kerja singkat (TL;DR)](#2-alur-kerja-singkat-tldr)
3. [Aturan branch](#3-aturan-branch)
4. [Aturan commit](#4-aturan-commit)
5. [Aturan Pull Request (PR)](#5-aturan-pull-request-pr)
6. [Aturan review](#6-aturan-review)
7. [Aturan kode & gaya](#7-aturan-kode--gaya)
8. [Aturan file & folder — JANGAN disentuh / JANGAN di-commit](#8-aturan-file--folder)
9. [Aturan Issue](#9-aturan-issue)
10. [Checklist sebelum push](#10-checklist-sebelum-push)

---

## 1. Prinsip utama

- **`main` adalah cabang suci.** Jangan pernah `push` langsung ke `main`. Semua perubahan masuk
  lewat Pull Request yang sudah di-review.
- **Satu PR = satu tujuan.** Jangan mencampur refaktor, fitur baru, dan perbaikan bug dalam satu PR.
- **Sumber kebenaran hyperparameter = `src/clouddensenet/config.py`.** Jangan hard-code angka di
  script; ubah di `config.py`.
- **Reproducibility dijaga.** `SEED = 42` tidak diubah tanpa alasan kuat yang ditulis di PR.
- **Bahasa:** komentar, docstring, pesan commit, dan deskripsi PR ditulis dalam **Bahasa Indonesia**,
  mengikuti gaya proyek (mis. `muat_imagenet`, `siapkan_batch`).

---

## 2. Alur kerja singkat (TL;DR)

```bash
# 1. Sinkronkan main terbaru
git checkout main
git pull origin main

# 2. Buat branch baru dari main
git checkout -b feat/nama-fitur-singkat

# 3. Kerja, lalu verifikasi WAJIB sebelum commit
pytest

# 4. Commit dengan format Conventional Commits
git add -p
git commit -m "feat(engine): tambah early stopping berbasis val F1"

# 5. Push branch & buka PR ke main
git push -u origin feat/nama-fitur-singkat
# lalu buka PR di GitHub, isi deskripsi sesuai template di §5
```

---

## 3. Aturan branch

### 3.1. Jangan bekerja di `main`

`main` hanya diperbarui lewat merge PR. Selalu buat branch sendiri.

### 3.2. Penamaan branch

Format: `<tipe>/<deskripsi-singkat-kebab-case>`

| Tipe | Untuk | Contoh |
|---|---|---|
| `feat/` | fitur baru | `feat/stratified-kfold` |
| `fix/` | perbaikan bug | `fix/nan-guard-amp` |
| `refactor/` | ubah struktur tanpa ubah perilaku | `refactor/pisah-engine-loop` |
| `docs/` | dokumentasi saja | `docs/panduan-kolaborasi` |
| `exp/` | eksperimen / coba-coba arsitektur | `exp/sconv-tau-tuning` |
| `chore/` | tooling, config repo, dependensi | `chore/bump-torch-2.5.1` |

- Gunakan huruf kecil dan tanda hubung (`-`), bukan spasi atau underscore.
- Satu branch untuk satu topik. Selesai di-merge → hapus branch-nya.

### 3.3. Selalu mulai dari `main` yang terbaru

```bash
git checkout main && git pull origin main
git checkout -b feat/...
```

### 3.4. Menjaga branch tetap sinkron

Kalau `main` bergerak jauh selama kamu kerja, rebase branch-mu di atas `main` terbaru:

```bash
git fetch origin
git rebase origin/main
# selesaikan konflik (jika ada), lalu:
git push --force-with-lease     # HANYA di branch sendiri, JANGAN di main
```

> `--force-with-lease` lebih aman daripada `--force`: ia menolak menimpa kalau ada commit orang
> lain yang belum kamu lihat.

---

## 4. Aturan commit

Proyek ini memakai **[Conventional Commits](https://www.conventionalcommits.org/)** —
konsisten dengan riwayat yang ada (`feat:`, `fix:`, …).

### 4.1. Format

```
<tipe>(<scope opsional>): <ringkasan imperatif, huruf kecil, tanpa titik>

<body opsional: jelaskan "kenapa", bukan "apa">

<footer opsional: refs #12, BREAKING CHANGE: ...>
```

### 4.2. Tipe yang dipakai

`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `style`.

**Scope** mengikuti modul di `src/clouddensenet/`, mis. `model`, `engine`, `data`, `config`,
`attention`, `conv`, `losses`, `finetune`, atau `scripts`.

### 4.3. Contoh baik

```
feat(engine): tambah early stopping berbasis val F1
fix(model): terapkan init LeCun-uniform ke Top Block & cabang ND
refactor(data): pisah transform train/val ke fungsi sendiri
docs: tambah panduan kolaborasi GitHub
chore(deps): kunci numpy ke >=1.26,<2.1
```

### 4.4. Aturan tambahan

- **Ringkasan ≤ 72 karakter**, pakai kata kerja imperatif ("tambah", "perbaiki"), bukan
  "menambahkan"/"ditambahkan".
- **Commit kecil & logis.** Jangan satu commit raksasa "update banyak hal".
- **Jangan commit kode yang gagal `pytest`.** Kalau WIP, tandai jelas dan jangan di-merge.
- Jangan pakai `--no-verify` untuk melewati hook kecuali disepakati.

---

## 5. Aturan Pull Request (PR)

### 5.1. Syarat membuka PR

Sebelum buka PR, pastikan:

- [ ] `pytest` **hijau** di lokal.
- [ ] Branch sudah di-rebase/merge dengan `main` terbaru (tanpa konflik).
- [ ] Tidak ada file terlarang ikut ter-commit (lihat [§8](#8-aturan-file--folder)).
- [ ] Perubahan hyperparameter dilakukan di `config.py`, bukan hard-code.

### 5.2. Judul PR

Pakai format Conventional Commits yang sama dengan commit, mis.:
`feat(engine): early stopping berbasis val F1`.

### 5.3. Template deskripsi PR

Salin ini ke deskripsi PR:

```markdown
## Ringkasan
<!-- Apa yang diubah dan kenapa, 2–4 kalimat -->

## Jenis perubahan
- [ ] feat (fitur baru)
- [ ] fix (perbaikan bug)
- [ ] refactor / chore / docs

## Dampak ke training
<!-- Apakah mengubah arsitektur, loss, config, atau hasil? Jelaskan. -->
<!-- Jika mengubah SEED atau hyperparameter, sebutkan nilai lama → baru. -->

## Cara menguji
<!-- Langkah agar reviewer bisa memverifikasi. Minimal: `pytest`. -->

## Checklist
- [ ] `pytest` hijau
- [ ] Tidak meng-commit data/artefak/.pth (lihat .gitignore)
- [ ] Tidak menyentuh Backup/, GCD.zip, atau notebook asal
- [ ] Hyperparameter diubah di config.py (bukan hard-code)
```

### 5.4. Aturan merge

- **Butuh minimal 1 approval** sebelum merge.
- Resolusikan semua komentar reviewer atau balas dengan alasan.
- Strategi merge disarankan **Squash and merge** agar riwayat `main` tetap bersih (satu PR = satu
  commit rapi dengan judul Conventional Commits).
- **Hapus branch** setelah merge.

---

## 6. Aturan review

Sebagai **reviewer**:

- Fokus pada: kebenaran (terutama shape tensor & stabilitas AMP/NaN), reproducibility, dan apakah
  perubahan sesuai konvensi `CLAUDE.md`.
- Pastikan tidak ada angka hyperparameter yang nyasar ter-hard-code di luar `config.py`.
- Cek tidak ada file terlarang ([§8](#8-aturan-file--folder)) yang ikut masuk.
- Bersikap spesifik dan sopan; sarankan perbaikan konkret.

Sebagai **penulis**:

- Jangan merge PR sendiri tanpa review (kecuali `docs` sepele yang disepakati).
- Balas setiap komentar; jangan menutup diskusi sebelum selesai.

---

## 7. Aturan kode & gaya

- **Ikuti `CLAUDE.md`** untuk aturan arsitektur dan penamaan (Bahasa Indonesia untuk
  komentar/docstring/sebagian nama fungsi).
- **Verifikasi cepat wajib:** jalankan `pytest` setelah mengubah apa pun di `src/clouddensenet/`.
  Smoke test `tests/test_forward.py` memverifikasi output `(N, 7)` tanpa dataset/GPU.
- **DEBUG mode:** set `DEBUG = True` di `config.py` untuk uji cepat 1–3 batch/epoch di CPU sebelum
  menjalankan training penuh.
- **Tambah test** untuk perubahan logika bila memungkinkan (mis. forward pass varian baru).
- Jaga `import` rapi dan hindari dependensi baru tanpa alasan — kalau menambah dependensi, update
  `requirements.txt` dan `pyproject.toml`, lalu sebutkan di PR.

---

## 8. Aturan file & folder

### 8.1. JANGAN disentuh / diubah

| Target | Alasan |
|---|---|
| `Backup/` | Sengaja dibiarkan utuh. |
| `data/raw/GCD.zip` | Arsip dataset asli — jangan dipindah/hapus. |
| `data/raw/GCD/` | Hasil ekstrak cadangan — jangan diubah. |
| `Copy_of_CloudDenseNet.ipynb` | Notebook asal, dibiarkan sebagai referensi. |

### 8.2. JANGAN di-commit (sudah ada di `.gitignore`)

- `.venv/`
- `data/raw/`, `data/processed/`
- `artifacts/checkpoints/`, `artifacts/results/`, `artifacts/runs/`
- `*.pth`, `*.pt` (bobot/checkpoint model)
- `Backup/`

> **Sebelum `git add -A`**, jalankan `git status` dan pastikan tidak ada file besar
> (dataset/checkpoint) yang ikut. Kalau ragu, pakai `git add -p` agar sadar apa yang masuk.
> Jika file besar sudah terlanjur ter-stage: `git restore --staged <file>`.

---

## 9. Aturan Issue

Buka Issue untuk bug, usulan fitur, atau pertanyaan desain. Sertakan:

**Untuk bug:**
- Langkah reproduksi (perintah persis, mis. `python scripts/train.py --epochs 1`).
- Yang diharapkan vs yang terjadi (sertakan traceback / log error).
- Lingkungan: OS, Python, versi PyTorch, GPU/CPU.

**Untuk fitur / eksperimen:**
- Motivasi (masalah apa yang diselesaikan).
- Usulan perubahan (modul mana, pengaruh ke `config.py`).
- Dampak ke reproducibility / hasil training bila ada.

Kaitkan Issue dengan PR lewat footer commit atau deskripsi PR: `refs #12` atau `closes #12`.

---

## 10. Checklist sebelum push

Jalankan ini sebagai ritual terakhir sebelum `git push`:

```bash
pytest                      # harus hijau
git status                  # pastikan tidak ada data/artefak/.pth yang ter-stage
git log --oneline -5        # cek pesan commit rapi & sesuai Conventional Commits
```

- [ ] `pytest` hijau
- [ ] Hanya file yang relevan ter-stage (tidak ada `.pth`, dataset, `artifacts/`)
- [ ] Pesan commit mengikuti Conventional Commits
- [ ] Branch bukan `main`
- [ ] (Jika mengubah config/arsitektur) dampaknya dijelaskan untuk reviewer

---

Terima kasih sudah berkontribusi menjaga `main` tetap sehat dan bisa dilatih. 🌥️
