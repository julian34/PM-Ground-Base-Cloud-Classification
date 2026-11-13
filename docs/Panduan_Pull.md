# Panduan Menarik Pembaruan (git pull) — CloudDenseNet

*Langkah memperbarui salinan lokal dengan perubahan terbaru dari GitHub*

Dokumen ini untuk kolaborator yang **sudah meng-clone** proyek dan ingin mengambil
perubahan terbaru dari repository. `git pull` menggabungkan dua langkah: `git fetch`
(mengunduh commit terbaru dari `origin`) lalu `git merge` (menyatukannya ke branch lokal).

> ⚠️ Yang ditarik hanya **kode & dokumen** yang ikut di-commit. Folder `.venv/`, dataset
> (`data/`), dan artefak training (`artifacts/`) **TIDAK** ada di Git — jadi `pull` tidak
> akan menyentuh venv, dataset, atau checkpoint lokalmu. Lihat Langkah 4 bila ada
> dependensi baru.

## Prasyarat

| Komponen | Keterangan |
|---|---|
| Sudah clone repo | Lihat [`Panduan_Instalasi_Clone.md`](Panduan_Instalasi_Clone.md) bila belum. |
| Git terpasang | Untuk `fetch`/`pull`/`merge`. |
| Koneksi internet | Untuk menghubungi `origin` di GitHub. |

## Langkah 1 — Cek dulu kondisi lokal

Sebelum menarik, pastikan tahu ada di branch apa dan apakah ada perubahan yang belum
di-commit:

```bash
git status
git branch            # tanda * menunjukkan branch aktif
```

> ⚠️ Kalau `git status` menunjukkan ada perubahan belum tersimpan, **jangan langsung
> pull** — bisa bentrok. Selesaikan dulu dengan salah satu pilihan di Langkah 2.

## Langkah 2 — Amankan perubahan lokal (bila ada)

Pilih salah satu sesuai situasi:

**A. Perubahanmu sudah final → commit dulu:**

```bash
git add .
git commit -m "pesan singkat perubahanmu"
```

**B. Perubahan belum selesai → simpan sementara (stash):**

```bash
git stash            # sembunyikan perubahan sementara
# ... lakukan pull (Langkah 3) ...
git stash pop        # kembalikan perubahan setelah pull
```

Kalau working tree sudah **bersih** (`nothing to commit`), langsung ke Langkah 3.

## Langkah 3 — Tarik pembaruan

Tarik perubahan untuk **branch yang sedang aktif**:

```bash
git pull origin <nama-branch>
```

Ganti `<nama-branch>` dengan branch tempatmu bekerja, misalnya:

```bash
git pull origin main
```

Branch lain yang ada di repo ini: `docs/panduan-kolaborasi`,
`Stratified-K-Fold-Cross-Validation`.

> 💡 Jika branch lokalmu sudah "melacak" (tracking) branch remote-nya, cukup `git pull`
> tanpa argumen. Cek dengan `git status` — baris *"Your branch is up to date with
> 'origin/...'"* menandakan tracking sudah aktif.

**Alternatif lebih aman (lihat dulu sebelum menggabung):**

```bash
git fetch origin                 # unduh saja, belum menggabung
git log --oneline HEAD..origin/<nama-branch>   # lihat commit baru yg akan masuk
git merge origin/<nama-branch>   # gabungkan bila sudah yakin
```

## Langkah 4 — Pasang dependensi baru (bila ada)

Kalau pembaruan mengubah `requirements.txt` atau `pyproject.toml` (mis. ada paket baru),
perbarui lingkungan. Aktifkan venv dulu, lalu:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

(Ganti `cu124` → `cpu` bila tanpa GPU.) Jalankan `git log` atau lihat pesan commit untuk
tahu apakah dependensi berubah.

## Langkah 5 — Verifikasi cepat

Pastikan kode masih sehat setelah pembaruan:

```bash
pytest
```

Keluaran harus `passed`. Bila ada kegagalan, baca pesannya — mungkin butuh langkah
tambahan yang disebut di commit terbaru.

## Menangani konflik (merge conflict)

Konflik terjadi bila kamu dan orang lain mengubah **baris yang sama** pada file yang sama.
Git akan berhenti dan menandai filenya:

```bash
git status            # lihat file yang "both modified"
```

1. Buka file yang konflik; cari penanda `<<<<<<<`, `=======`, `>>>>>>>`.
2. Edit manual: pilih/gabungkan bagian yang benar, **hapus ketiga penanda** itu.
3. Tandai selesai lalu rampungkan:

```bash
git add <file-yang-sudah-dibereskan>
git commit            # selesaikan merge (pesan default biasanya sudah cukup)
```

Ingin **membatalkan** merge dan kembali ke keadaan sebelum pull?

```bash
git merge --abort
```

## Ringkasan cepat

1. `git status` — pastikan tahu branch & kondisi lokal.
2. Commit atau `git stash` perubahan yang belum tersimpan.
3. `git pull origin <nama-branch>`.
4. Bila `requirements.txt` berubah: `pip install -r requirements.txt ...` + `pip install -e .`.
5. `pytest` untuk verifikasi.

## Troubleshooting ringkas

- **`Your local changes would be overwritten by merge`:** ada perubahan belum di-commit.
  Jalankan `git stash` (lalu `git stash pop` setelah pull) atau commit dulu.
- **`Already up to date.`:** tidak ada commit baru di remote — lokalmu sudah terbaru.
- **Nyasar/terlanjur di branch lain:** pindah dengan `git switch <nama-branch>` sebelum pull.
- **Ingin menyamakan paksa dengan remote (buang perubahan lokal):** `git fetch origin`
  lalu `git reset --hard origin/<nama-branch>` — ⚠️ **menghapus** semua perubahan lokal
  yang belum di-push, pakai dengan hati-hati.
- **Konflik membingungkan:** `git merge --abort` untuk kembali ke keadaan sebelum pull,
  lalu minta bantuan sebelum mencoba lagi.
