"""Ekspor CloudDenseNet (checkpoint .pth) ke ONNX untuk Zetane Viewer / Netron.

Zetane Viewer (aplikasi desktop) & Netron membaca file .onnx, bukan .pth. Script
ini memuat bobot dari checkpoint, menjalankan model dalam mode eval + fp32 (tanpa
AMP), lalu menulis graf ONNX yang bisa dibuka untuk melihat arsitektur sekaligus
tensor/aktivasi antar-layer.

Catatan desain:
  * Ekspor memakai jalur `lam=None` -> SpatialAttnBR berperilaku seperti spatial
    attention biasa (channel λ = 0). Peta λ rasio B/R (spectral.lambda_relatif)
    memakai median per-gambar + masking dinamis yang tidak bisa dilacak ke ONNX,
    jadi sengaja TIDAK disertakan. Struktur & bobot backbone tetap sama persis.
  * Input berupa citra RGB ter-normalisasi (lihat NORM_MODE/MEAN/STD di config),
    shape (N, 3, IMG_SIZE, IMG_SIZE).

Prasyarat: paket `onnx` (eksporter PyTorch membutuhkannya). Pasang via:
    pip install -e .[export]     # atau: pip install onnx

Contoh:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint artifacts/checkpoints/cloud_Versi01.pth
    python scripts/export_onnx.py --opset 11 --output artifacts/onnx/clouddensenet.onnx
    python scripts/export_onnx.py --random   # tanpa checkpoint (bobot init, cek graf)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import CloudDenseNet  # noqa: E402


class LogitsOnly(nn.Module):
    """Bungkus model agar forward hanya menerima 1 input (x) dan 1 output (logits).

    CloudDenseNet.forward(x, lam=None, return_all=False) punya argumen tambahan
    yang membingungkan eksporter ONNX; wrapper ini menetapkan lam=None.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def muat_state_dict(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Muat bobot dari checkpoint, toleran terhadap beberapa format kunci."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        state = (ck.get("model") or ck.get("model_state_dict")
                 or ck.get("state_dict") or ck)
    else:
        state = ck
    hasil = model.load_state_dict(state, strict=False)
    if hasil.missing_keys:
        print(f"[warn] {len(hasil.missing_keys)} kunci tidak ada di checkpoint "
              f"(mis. {hasil.missing_keys[:3]} ...)")
    if hasil.unexpected_keys:
        print(f"[warn] {len(hasil.unexpected_keys)} kunci tak terduga di checkpoint "
              f"(mis. {hasil.unexpected_keys[:3]} ...)")
    epoch = ck.get("epoch") if isinstance(ck, dict) else None
    print(f"Checkpoint dimuat: {ckpt_path.name}" + (f" (epoch {epoch})" if epoch else ""))


def parse_args():
    p = argparse.ArgumentParser(description="Ekspor CloudDenseNet ke ONNX")
    p.add_argument("--checkpoint", type=Path,
                   default=C.CKPT_DIR / f"{C.RUN_NAME}{C.VERSION}.pth",
                   help="Path checkpoint .pth (default: checkpoint terbaik).")
    p.add_argument("--output", type=Path,
                   default=C.PROJECT_DIR / "artifacts" / "onnx" / "clouddensenet.onnx",
                   help="Path file .onnx keluaran.")
    p.add_argument("--opset", type=int, default=13,
                   help="Versi opset ONNX (default 13; coba 11 bila Zetane menolak).")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    p.add_argument("--batch", type=int, default=1, help="Ukuran batch contoh (dummy input).")
    p.add_argument("--random", action="store_true",
                   help="Jangan muat checkpoint; pakai bobot init (untuk cek graf saja).")
    p.add_argument("--static", action="store_true",
                   help="Shape tetap (tanpa dynamic axes) — lebih kompatibel utk viewer lawas (Zetane).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")   # ekspor di CPU, fp32 — stabil & deterministik

    model = CloudDenseNet().to(device)

    if args.random:
        print("[info] Mode --random: ekspor bobot init (tanpa checkpoint).")
    elif args.checkpoint.exists():
        muat_state_dict(model, args.checkpoint, device)
    else:
        raise SystemExit(
            f"Checkpoint tidak ditemukan: {args.checkpoint}\n"
            "Latih dulu (python scripts/train.py), beri --checkpoint <path>, "
            "atau pakai --random untuk mengekspor graf tanpa bobot terlatih."
        )

    model.eval().float()
    wrapped = LogitsOnly(model).eval()

    dummy = torch.randn(args.batch, 3, args.img_size, args.img_size, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None if args.static else {"input": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(
            wrapped, dummy, str(args.output),
            input_names=["input"], output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"[OK] ONNX ditulis -> {args.output}  (opset {args.opset})")

    # Verifikasi graf (paket 'onnx' sudah pasti ada: eksporter di atas membutuhkannya).
    try:
        import onnx  # noqa: PLC0415

        m = onnx.load(str(args.output))
        onnx.checker.check_model(m)
        print("[OK] onnx.checker: graf valid.")
    except Exception as e:  # pragma: no cover - diagnostik
        print(f"[warn] onnx.checker melaporkan masalah: {e}")

    print("\nBuka file di Zetane Viewer (aplikasi desktop) atau https://netron.app")


if __name__ == "__main__":
    main()
