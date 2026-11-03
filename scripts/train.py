"""Entrypoint training CloudDenseNet (lokal, lepas dari Google Colab).

Contoh:
    python scripts/train.py
    python scripts/train.py --data-root data/raw/GCD-dataset --epochs 80

Dataset harus berbentuk ImageFolder: <data-root>/{train,val}/<kelas>/*.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    CloudDenseNet, FocalLoss, apply_phase, build_loaders, catat_epoch, evaluate,
    muat_imagenet, nolkan_input_nd, phase_of, set_seed, siapkan_tensorboard,
    train_one_epoch, tutup_tensorboard,
)
from clouddensenet.engine import is_finite  # noqa: E402
from clouddensenet.viz import plot_confusion_matrix, plot_training_history  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Training CloudDenseNet 7-kelas awan")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(C.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and not C.DEBUG else "cpu")
    print(f"Device: {device} | DEBUG: {C.DEBUG} | SEED: {C.SEED}")

    if not args.data_root.exists():
        raise SystemExit(
            f"Dataset tidak ditemukan di {args.data_root}\n"
            "Letakkan dataset GCD (train/val/test) di sana — lihat data/README.md."
        )

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = C.CKPT_DIR / f"{args.run_name}{C.VERSION}.pth"
    last_path = C.CKPT_DIR / f"{args.run_name}{C.VERSION}_last.pth"

    train_loader, val_loader, train_ds, val_ds = build_loaders(args.data_root, device)
    classes = getattr(val_ds, "classes", C.CLASS_NAMES)
    print(f"Kelas ({len(classes)}): {classes}")

    model = CloudDenseNet().to(device)
    model = muat_imagenet(model)
    model = nolkan_input_nd(model)

    criterion = FocalLoss() if C.FOCAL_GAMMA > 0.0 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    writer = siapkan_tensorboard(
        C.TB_LOG_DIR, args.run_name,
        aktif=C.USE_TENSORBOARD, auto_install=C.TB_AUTO_INSTALL,
        auto_launch=C.TB_AUTO_LAUNCH, port=C.TB_PORT,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, start_epoch, current_phase = -1.0, 1, 0

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if device.type == "cuda" and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        history = ck["history"]
        best_f1 = ck.get("best_f1", max(history["val_f1"], default=-1.0))
        start_epoch = ck["epoch"] + 1
        print(f"Resume dari epoch {start_epoch} | best_f1 {best_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        phase = phase_of(epoch)
        if phase != current_phase:
            apply_phase(model, optimizer, phase)
            current_phase = phase
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _, _lr, _strategi, _desc = C.PHASES[phase - 1]
            print(f"Epoch {epoch:02d}: tahap {phase} – {_strategi}: {_desc} | "
                  f"lr {_lr:g} | trainable {n_tr:,}")

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device,
                                          writer=writer, epoch=epoch)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)

        if not is_finite(tr_loss, va_loss):
            print("[STOP] Loss NaN/Inf -> berhenti; checkpoint terakhir tetap bersih")
            break

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_f1]):
            history[k].append(v)
        catat_epoch(writer, epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1,
                    optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch:02d}/{args.epochs} [tahap {phase}] | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f} | {time.time() - t0:.0f}s")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_f1": best_f1}, ckpt_path)
            print(f"Epoch {epoch:02d}: [OK] model terbaik disimpan (f1={best_f1:.4f})")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best_f1": best_f1}, last_path)

    if history["train_loss"]:
        plot_training_history(history, simpan=str(C.RESULT_DIR / f"{args.run_name}_kurva.png"))
        plot_confusion_matrix(model, val_loader, device,
                              simpan=str(C.RESULT_DIR / f"{args.run_name}_cm_val.png"))

    tutup_tensorboard(writer)


if __name__ == "__main__":
    main()
