# Web UI — Uji Model CloudDenseNet

Antarmuka web untuk **mencoba model terlatih**. Dua mode:

- **Satu citra** (`/`) — unggah satu foto langit → **kelas awan prediksi + persen
  kepercayaan** untuk ke-7 kelas, **pemilih checkpoint** dari `artifacts/checkpoints/`,
  dan **overlay Grad-CAM** opsional.
- **Satu folder** (`/batch`) — pilih **satu folder** citra langit; setiap gambar diproses
  dan dicatat pada **tabel per-gambar** (nama berkas, jenis awan, angka kepercayaan),
  lengkap dengan **ringkasan sebaran kelas** dan tombol **Unduh CSV**. Gambar yang rusak /
  bukan citra ditandai sebagai baris gagal tanpa menghentikan proses.

Backend FastAPI (ringan) menyajikan frontend statis dan memuat ulang kode inferensi
dari paket `clouddensenet` — jadi preprocessing (resize → ToTensor → normalisasi
ImageNet + channel λ Biru/Merah via `siapkan_batch`) **persis sama** seperti saat
validasi/training.

## Jalankan

Aktifkan venv, lalu pastikan dependensi web terpasang (sudah ada di `requirements.txt`):

```bash
pip install -r requirements.txt           # fastapi, uvicorn, python-multipart, dll
pip install -e .                           # agar paket clouddensenet ter-import
```

Start server (salah satu):

```bash
uvicorn model_ui.app:app --port 8000       # host 127.0.0.1 saja; tambahkan --reload saat dev
python -m model_ui.app                      # host 0.0.0.0 (akses LAN/hotspot) + cetak alamat
```

Buka **http://127.0.0.1:8000** (satu citra) atau **/batch** (satu folder) di browser.
Tautan antar-mode tersedia di pojok kanan atas tiap halaman.

### Akses dari perangkat lain (LAN / hotspot laptop)

`python -m model_ui.app` sudah listen di `0.0.0.0` dan mencetak alamat yang bisa
dibuka dari HP/perangkat lain pada jaringan yang sama (hotspot Windows biasanya
**http://192.168.137.1:8000**). Bila dari perangkat lain gagal padahal lokal jalan,
izinkan port lewat Windows Firewall (PowerShell Administrator, sekali saja):

```powershell
New-NetFirewallRule -DisplayName "CloudDenseNet UI 8000" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

## Endpoint API

| Metode | Path                  | Keterangan                                                           |
|--------|-----------------------|---------------------------------------------------------------------|
| GET    | `/`                   | Halaman satu-citra (`static/index.html`).                           |
| GET    | `/batch`              | Halaman unggah-folder (`static/batch.html`).                        |
| GET    | `/api/checkpoints`    | `{checkpoints: [...], default}` — daftar `.pth` (rekursif).          |
| POST   | `/api/predict`        | multipart: `file`, `checkpoint`, `gradcam` → JSON prediksi 7 kelas.  |
| POST   | `/api/predict-batch`  | multipart: `files[]`, `checkpoint` → kelas teratas + confidence/berkas.|

Respons `/api/predict-batch`: `{checkpoint, device, jumlah, berhasil, hasil: [{nama_file,
kelas, nama, indeks, confidence}` atau `{nama_file, error}]}`. Frontend mengirim folder
**per-potongan** (≤ `MAX_BATCH` gambar/permintaan) agar progres terlihat dan permintaan
tidak membengkak.

Contoh smoke test tanpa browser:

```bash
curl -s -F "file=@data/processed/test/3_cirrus/CONTOH.jpg" \
        -F "checkpoint=cloud_Versi01.pth" \
        http://127.0.0.1:8000/api/predict | python -m json.tool
```

## Catatan

- **Checkpoint:** default memakai checkpoint terbaik (`cloud_Versi01.pth`); file
  `*_last.pth` tetap bisa dipilih tapi tidak dijadikan default. Model di-cache per
  checkpoint, jadi berganti pilihan tidak memuat ulang bobot berulang kali.
- **Perangkat:** GPU (CUDA) dipakai otomatis bila tersedia; jika tidak, CPU (satu
  gambar tetap cepat).
- UI ini **tidak menyentuh** `src/clouddensenet/` — hanya mengimpornya.
