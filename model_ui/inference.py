"""Logika inferensi untuk web UI — memuat ulang kode paket `clouddensenet`.

Alur sama persis dengan validasi/evaluasi training supaya prediksi konsisten:

    PIL(RGB) -> Resize(IMG_SIZE) -> ToTensor([0,1]) -> siapkan_batch (norm imagenet + λ B/R)
    -> model(x, lam) -> softmax -> confidence per-kelas

Checkpoint dimuat lewat `clouddensenet.bangun_dari_checkpoint`, yang membaca
dimensi Top Block dari bentuk tensor `top.*` sehingga checkpoint dari varian head
lama (07–12-11-2026) tetap bisa dipakai. Model di-cache per-checkpoint agar ganti
pilihan di UI tidak memuat ulang bobot.
"""
from __future__ import annotations

import io
import base64
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Agar paket src/clouddensenet dapat di-import tanpa instalasi (sejalan dgn scripts/*).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    CloudDenseNet, GradCAM, bangun_dari_checkpoint, overlay_cam, siapkan_batch,
)


# --------------------------------------------------------------------------- #
# Penemuan & pemuatan checkpoint
# --------------------------------------------------------------------------- #
def temukan_checkpoint(ckpt_dir: Path = C.CKPT_DIR) -> dict:
    """Pindai `ckpt_dir` (rekursif) untuk file .pth.

    Mengembalikan dict: {"checkpoints": [path relatif str ...], "default": str|None}.
    Checkpoint resume (`*_last.pth`) tetap dilisting tapi tidak dijadikan default.
    Path relatif memakai '/' agar konsisten lintas-OS di sisi frontend.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return {"checkpoints": [], "default": None}

    rel = sorted(p.relative_to(ckpt_dir).as_posix()
                 for p in ckpt_dir.rglob("*.pth"))

    # Default: checkpoint TERBAIK (bukan *_last). Prioritaskan nama default config.
    non_last = [r for r in rel if not r.endswith("_last.pth")]
    disukai = f"{C.RUN_NAME}{C.VERSION}.pth"
    default = next((r for r in non_last if r.endswith(disukai)),
                   non_last[0] if non_last else (rel[0] if rel else None))
    return {"checkpoints": rel, "default": default}


def _nama_rapi(kelas: str) -> str:
    """'1_cumulus' -> 'Cumulus'; '7_mixed' -> 'Mixed'."""
    tanpa_prefiks = kelas.split("_", 1)[-1] if "_" in kelas else kelas
    return tanpa_prefiks.replace("_", " ").title()


class ClassifierService:
    """Pembungkus stateful: cache model per-checkpoint + inferensi satu gambar."""

    def __init__(self, ckpt_dir: Path = C.CKPT_DIR):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict[str, CloudDenseNet] = {}
        # Transform validasi — identik dgn data.build_transforms() -> val_tf.
        self._tf = transforms.Compose([
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    # -- util internal ------------------------------------------------------ #
    def _resolve(self, checkpoint: str | None) -> Path:
        """Ubah path relatif (dari UI) menjadi path absolut aman di dalam ckpt_dir."""
        if not checkpoint:
            info = temukan_checkpoint(self.ckpt_dir)
            checkpoint = info["default"]
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Tidak ada checkpoint .pth di {self.ckpt_dir}. Latih dulu "
                    "(python scripts/train.py) atau letakkan .pth di folder itu."
                )
        base = self.ckpt_dir.resolve()
        path = (base / checkpoint).resolve()
        # Cegah path traversal keluar dari folder checkpoint.
        if base != path and base not in path.parents:
            raise ValueError(f"Checkpoint di luar folder yang diizinkan: {checkpoint}")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
        return path

    def _get_model(self, checkpoint: str | None) -> tuple[CloudDenseNet, str]:
        path = self._resolve(checkpoint)
        key = str(path)
        model = self._cache.get(key)
        if model is None:
            # Head dibangun sesuai checkpoint; ValueError (detail lengkap) -> HTTP 400.
            model, _ = bangun_dari_checkpoint(path, self.device)
            model.eval().float()
            self._cache[key] = model
        rel = path.relative_to(self.ckpt_dir.resolve()).as_posix()
        return model, rel

    def _preprocess(self, pil_img: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PIL -> (x ternormalisasi, lam, x01) di device. x01 untuk tampilan overlay."""
        x01 = self._tf(pil_img.convert("RGB")).unsqueeze(0)      # (1,3,H,W) di [0,1]
        x, lam = siapkan_batch(x01, self.device)                 # norm imagenet + λ B/R
        return x, lam, x01

    # -- API publik --------------------------------------------------------- #
    def muat_model(self, checkpoint: str | None = None) -> tuple[CloudDenseNet, str]:
        """Resolusi + muat model (cache), kembalikan (model, path_relatif).

        Dipakai inferensi batch agar bobot hanya di-resolve sekali untuk banyak
        gambar (lihat `predict_top`).
        """
        return self._get_model(checkpoint)

    @torch.no_grad()
    def predict_top(self, pil_img: Image.Image, model: CloudDenseNet) -> dict:
        """Prediksi kelas teratas saja dengan model yang sudah dimuat.

        Dibedakan dari `predict` (yang mengembalikan seluruh 7 probabilitas) agar
        loop batch ringan: satu forward -> argmax -> {kelas, nama, confidence}.
        """
        x, lam, _ = self._preprocess(pil_img)
        probs = F.softmax(model(x, lam), dim=1)[0].cpu().numpy()
        i = int(probs.argmax())
        return {
            "indeks": i,
            "kelas": C.CLASS_NAMES[i],
            "nama": _nama_rapi(C.CLASS_NAMES[i]),
            "confidence": float(probs[i]),
        }

    @torch.no_grad()
    def predict(self, pil_img: Image.Image, checkpoint: str | None = None) -> dict:
        """Prediksi 7 kelas. Mengembalikan dict JSON-serializable."""
        model, rel = self._get_model(checkpoint)
        x, lam, _ = self._preprocess(pil_img)
        logits = model(x, lam)                                   # (1,7)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        hasil = [
            {"indeks": int(i), "kelas": C.CLASS_NAMES[i],
             "nama": _nama_rapi(C.CLASS_NAMES[i]), "confidence": float(probs[i])}
            for i in range(len(C.CLASS_NAMES))
        ]
        hasil.sort(key=lambda d: d["confidence"], reverse=True)
        return {
            "top": hasil[0],
            "probs": hasil,
            "device": self.device.type,
            "checkpoint": rel,
        }

    def gradcam(self, pil_img: Image.Image, checkpoint: str | None = None) -> str:
        """Overlay Grad-CAM (kelas prediksi) -> data-URI PNG base64."""
        model, _ = self._get_model(checkpoint)
        x, lam, x01 = self._preprocess(pil_img)

        gc = GradCAM(model, model.norm5)
        try:
            cam, _, _ = gc(x, lam)                               # cam (1,H,W) di [0,1]
        finally:
            gc.remove()

        overlay = overlay_cam(x01[0].cpu().numpy(), cam[0])      # (H,W,3) di [0,1]
        arr = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
