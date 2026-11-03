"""Stratified K-Fold Cross-Validation untuk CloudDenseNet (+ Gradient Accumulation).

CV dijalankan HANYA di pool `train/`; folder `test/` tetap holdout final. Tiap fold
melatih model DARI NOL (bobot ImageNet) memakai jadwal fase yang DIKOMPRESI
proporsional 80 -> 30 epoch (PHASES_CV) agar hemat waktu tanpa membuang fase 'full'.

Contoh:
    python scripts/train_kfold.py
    python scripts/train_kfold.py --folds 3 --accum-steps 4
    python scripts/train_kfold.py --data-root data/processed --folds 5

Keluaran:
    artifacts/checkpoints/<run>_fold{k}_Versi01.pth   (model terbaik tiap fold)
    artifacts/results/kfold_<run>.json                (skor per-fold + mean/std + OOF)
    artifacts/results/kfold_<run>_oof.npz             (y_true, y_pred out-of-fold)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    CloudDenseNet, FocalLoss, evaluate, muat_imagenet, nolkan_input_nd, set_seed,
    train_one_epoch,
)
from clouddensenet.data import siapkan_batch  # noqa: E402
from clouddensenet.data_kfold import base_targets, build_fold_loaders  # noqa: E402
from clouddensenet.engine import is_finite  # noqa: E402
from clouddensenet.finetune import bekukan_imagenet, buka_blok_3_4, buka_semua  # noqa: E402

# --------------------------------------------------------------------------- #
# Jadwal CV: 6 fase PHASES dikompresi proporsional 80 -> 30 epoch.
# LR & strategi dipertahankan; hanya panjang tiap fase dipadatkan (4·4·4·4·7·7).
# Batas kumulatif (epoch_akhir, lr, strategi, deskripsi):
# --------------------------------------------------------------------------- #
PHASES_CV = [
    (4,  1e-3, "frozen",  "Top Block (ImageNet beku)"),      # ep 1-4
    (8,  1e-4, "frozen",  "Tahap 2 (ImageNet beku)"),        # ep 5-8
    (12, 1e-5, "partial", "Adaptasi Top Block"),             # ep 9-12
    (16, 5e-5, "partial", "Unfreeze Dense Block 3-4"),       # ep 13-16
    (23, 1e-5, "full",    "Partial Fine-Tuning"),            # ep 17-23
    (30, 1e-6, "full",    "Full Defrost (end-to-end)"),      # ep 24-30
]
EPOCHS_CV = PHASES_CV[-1][0]


def phase_of_cv(epoch, phases=PHASES_CV):
    """Nomor fase (1..n) untuk sebuah epoch pada jadwal CV."""
    return next((i for i, p in enumerate(phases, 1) if epoch <= p[0]), len(phases))


def apply_phase_cv(model, optimizer, phase, phases=PHASES_CV):
    """Terapkan pembekuan + LR fase CV (meniru engine.apply_phase, tanpa global)."""
    strategi = phases[phase - 1][2]
    if strategi == "frozen":
        bekukan_imagenet(model)
    elif strategi == "partial":
        buka_blok_3_4(model)
    else:
        buka_semua(model)
    for grp in optimizer.param_groups:
        grp["lr"] = phases[phase - 1][1]


@torch.no_grad()
def predict_loader(model, loader, device):
    """Prediksi seluruh loader (urut, tanpa shuffle) -> (y_true, y_pred) numpy."""
    model.eval()
    yt, yp = [], []
    for x01, y in loader:
        x, lam = siapkan_batch(x01, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(x, lam).float()
        yt.append(y.numpy())
        yp.append(out.argmax(1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def train_one_fold(fold, data_root, device, train_idx, val_idx, run_name, accum_steps):
    """Satu training penuh (jadwal CV) pada satu fold. Kembalikan (best_f1, val_loader, model)."""
    set_seed(C.SEED)   # model & sampler identik tiap fold -> yang berbeda hanya split data
    train_loader, val_loader, _, _ = build_fold_loaders(data_root, device, train_idx, val_idx)

    model = nolkan_input_nd(muat_imagenet(CloudDenseNet().to(device)))
    criterion = FocalLoss() if C.FOCAL_GAMMA > 0 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt = C.CKPT_DIR / f"{run_name}_fold{fold}{C.VERSION}.pth"
    best_f1, current_phase = -1.0, 0

    for epoch in range(1, EPOCHS_CV + 1):
        phase = phase_of_cv(epoch)
        if phase != current_phase:
            apply_phase_cv(model, optimizer, phase)
            current_phase = phase
            print(f"  [fold {fold}] ep {epoch:02d}: fase {phase} - {PHASES_CV[phase-1][3]} "
                  f"| lr {PHASES_CV[phase-1][1]:g}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                                          device, epoch=epoch, accum_steps=accum_steps)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)
        if not is_finite(tr_loss, va_loss):
            print(f"  [fold {fold}] loss NaN/Inf -> stop fold ini")
            break
        print(f"  [fold {fold}] ep {epoch:02d}/{EPOCHS_CV} | train loss {tr_loss:.4f} acc {tr_acc:.3f}"
              f" | val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_f1": best_f1}, ckpt)

    return best_f1, val_loader, model


def parse_args():
    p = argparse.ArgumentParser(description="Stratified K-Fold CV CloudDenseNet")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--folds", type=int, default=3, help="jumlah fold (default 3)")
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--accum-steps", type=int, default=C.ACCUM_STEPS,
                   help="Gradient Accumulation; batch efektif = BATCH_SIZE x accum")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not C.DEBUG else "cpu")
    print(f"Device: {device} | folds: {args.folds} | epoch/fold: {EPOCHS_CV} "
          f"| accum: {args.accum_steps} (batch efektif {C.BATCH_SIZE * args.accum_steps})")

    if not args.data_root.exists():
        raise SystemExit(f"Dataset tidak ditemukan di {args.data_root} — lihat data/README.md.")

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    targets, classes = base_targets(args.data_root)
    print(f"Pool train: {len(targets)} citra | {len(classes)} kelas: {classes}")

    # shuffle=True + random_state=SEED -> fold deterministik & reproducible (konsisten SEED=42).
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=C.SEED)
    X_dummy = np.zeros(len(targets))        # StratifiedKFold hanya butuh label (y)

    fold_f1, splits = [], list(skf.split(X_dummy, targets))
    oof_true = np.full(len(targets), -1, dtype=np.int64)   # prediksi out-of-fold
    oof_pred = np.full(len(targets), -1, dtype=np.int64)

    t_all = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splits):
        dist = np.bincount(targets[va_idx], minlength=len(classes)).tolist()
        print(f"\n=== Fold {fold}/{args.folds - 1} | train {len(tr_idx)} val {len(va_idx)} "
              f"| dist val per-kelas: {dist}")
        t0 = time.time()
        best_f1, val_loader, model = train_one_fold(
            fold, args.data_root, device, tr_idx, va_idx, args.run_name, args.accum_steps)
        fold_f1.append(best_f1)

        yt, yp = predict_loader(model, val_loader, device)        # OOF pada fold ini
        oof_true[va_idx] = yt
        oof_pred[va_idx] = yp
        print(f"=== Fold {fold} selesai | best val f1 {best_f1:.4f} | {time.time()-t0:.0f}s")

    arr = np.array(fold_f1)
    mask = oof_pred >= 0
    oof_macro = float(f1_score(oof_true[mask], oof_pred[mask], average="macro"))

    print("\n" + "=" * 56)
    print(f"Hasil {args.folds}-Fold Stratified CV (macro-F1), jadwal {EPOCHS_CV} epoch:")
    for i, f in enumerate(arr):
        print(f"  fold {i}: {f:.4f}")
    print("  " + "-" * 34)
    print(f"  F1 = {arr.mean():.4f} ± {arr.std():.4f}   (mean ± std antar-fold)")
    print(f"  OOF macro-F1 (gabungan semua fold) = {oof_macro:.4f}")
    print(f"  total waktu CV: {(time.time()-t_all)/60:.1f} menit")
    print("\nLaporan OOF per-kelas:")
    print(classification_report(oof_true[mask], oof_pred[mask], target_names=classes, digits=4))

    result = {
        "config": {"folds": args.folds, "epochs_per_fold": EPOCHS_CV,
                   "accum_steps": args.accum_steps, "batch_size": C.BATCH_SIZE,
                   "effective_batch": C.BATCH_SIZE * args.accum_steps, "seed": C.SEED,
                   "phases_cv": [list(p) for p in PHASES_CV]},
        "fold_best_f1": [float(f) for f in arr],
        "mean_f1": float(arr.mean()), "std_f1": float(arr.std()),
        "oof_macro_f1": oof_macro, "classes": classes,
    }
    out_json = C.RESULT_DIR / f"kfold_{args.run_name}.json"
    out_npz = C.RESULT_DIR / f"kfold_{args.run_name}_oof.npz"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(out_npz, y_true=oof_true, y_pred=oof_pred)
    print(f"\n[OK] ringkasan -> {out_json}\n[OK] OOF arrays -> {out_npz}")


if __name__ == "__main__":
    main()
