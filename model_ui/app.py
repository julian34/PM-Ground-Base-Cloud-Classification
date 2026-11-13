"""Server FastAPI untuk web UI inferensi CloudDenseNet.

Endpoint:
    GET  /                  -> halaman satu-citra (static/index.html)
    GET  /batch             -> halaman unggah folder (static/batch.html)
    GET  /api/checkpoints   -> daftar checkpoint .pth + default
    POST /api/predict       -> prediksi 7 kelas (+ overlay Grad-CAM opsional)
    POST /api/predict-batch -> prediksi kelas teratas utk banyak gambar sekaligus

Jalankan:
    uvicorn model_ui.app:app --port 8000     # dengan --reload saat dev
    python -m model_ui.app
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import ClassifierService, temukan_checkpoint

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BYTES = 15 * 1024 * 1024        # batas ukuran unggahan ~15 MB
MAX_BATCH = 64                      # maks gambar per-request batch (frontend kirim per-potong)

app = FastAPI(title="CloudDenseNet — Uji Model", version="0.1.0")

# Satu service dibagi lintas request (model di-cache per-checkpoint di dalamnya).
service = ClassifierService()


@app.exception_handler(Exception)
async def _json_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Pastikan error tak-terduga tetap dibalas sebagai JSON {"detail": ...}.

    Tanpa ini, FastAPI membalas 500 dengan body teks polos "Internal Server Error",
    sehingga `res.json()` di frontend gagal dengan "Unexpected token 'I'". Dengan
    handler ini frontend selalu bisa membaca pesan lewat `data.detail`.
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"Kesalahan internal server: {exc}"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/batch")
def batch_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "batch.html")


@app.get("/api/checkpoints")
def list_checkpoints() -> dict:
    return temukan_checkpoint(service.ckpt_dir)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    checkpoint: str | None = Form(None),
    gradcam: bool = Form(False),
) -> JSONResponse:
    # Validasi tipe & isi.
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File kosong.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Gambar terlalu besar (maks 15 MB).")

    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Gambar tidak dapat dibaca / rusak.")

    try:
        hasil = service.predict(pil, checkpoint)
        if gradcam:
            hasil["overlay"] = service.gradcam(pil, checkpoint)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(hasil)


@app.post("/api/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    checkpoint: str | None = Form(None),
) -> JSONResponse:
    """Prediksi kelas teratas untuk banyak gambar sekaligus (satu folder).

    Satu gambar yang gagal (bukan citra / rusak / terlalu besar) TIDAK menggagalkan
    seluruh batch — barisnya diberi field `error` dan proses lanjut. Model dimuat
    sekali untuk seluruh potongan.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada berkas diunggah.")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Terlalu banyak berkas sekaligus (maks {MAX_BATCH} per potongan).",
        )

    # Muat model sekali; checkpoint yang tak ada -> error batch-level (bukan per-baris).
    try:
        model, rel = service.muat_model(checkpoint)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    hasil: list[dict] = []
    for f in files:
        baris: dict = {"nama_file": f.filename or "(tanpa nama)"}
        try:
            if f.content_type and not f.content_type.startswith("image/"):
                raise ValueError("Bukan berkas gambar.")
            data = await f.read()
            if not data:
                raise ValueError("Berkas kosong.")
            if len(data) > MAX_BYTES:
                raise ValueError("Gambar terlalu besar (maks 15 MB).")
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            baris.update(service.predict_top(pil, model))
        except (UnidentifiedImageError, OSError):
            baris["error"] = "Gambar tidak dapat dibaca / rusak."
        except ValueError as e:
            baris["error"] = str(e)
        hasil.append(baris)

    return JSONResponse({
        "checkpoint": rel,
        "device": service.device.type,
        "jumlah": len(hasil),
        "berhasil": sum(1 for b in hasil if "error" not in b),
        "hasil": hasil,
    })


# Static di-mount TERAKHIR agar tidak menimpa rute /api di atas.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
