"""Web UI inferensi CloudDenseNet.

Paket kecil untuk mencoba model terlatih lewat browser: upload satu citra langit,
lalu lihat kelas awan prediksi + persen kepercayaan (7 kelas) dan overlay Grad-CAM.

Jalankan:
    uvicorn model_ui.app:app --port 8000
    # atau
    python -m model_ui.app
"""
from __future__ import annotations

__all__ = ["ClassifierService", "temukan_checkpoint"]


def __getattr__(name):
    # Impor malas agar `import model_ui` tidak langsung memuat torch.
    if name in __all__:
        from .inference import ClassifierService, temukan_checkpoint
        return {"ClassifierService": ClassifierService,
                "temukan_checkpoint": temukan_checkpoint}[name]
    raise AttributeError(f"module {__name__!r} tidak punya atribut {name!r}")
