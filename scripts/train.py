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
from datetime import datetime
from pathlib import Path

import torch

# Agar paket src/clouddensenet dapat di-import tanpa instalasi.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet import (  # noqa: E402
    CloudDenseNet, CloudViTOnly, EpochLogger, FocalLoss, apply_phase, buat_laporan_lengkap,
    build_loaders, build_test_loader, catat_epoch, evaluate, muat_imagenet,
    nolkan_input_nd, phase_of, set_seed, siapkan_tensorboard, train_one_epoch,
    tutup_tensorboard,
)
from clouddensenet.engine import is_finite  # noqa: E402
from clouddensenet.viz import plot_confusion_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Training CloudDenseNet 7-kelas awan")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--run-name", default=C.RUN_NAME)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE,
                   help=f"Ukuran batch train/val/test (default {C.BATCH_SIZE} dari config). "
                        "Kecilkan bila VRAM kurang (mis. 16/8); besarkan bila GPU lapang.")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE,
                   help=f"Sisi citra persegi H=W piksel (default {C.IMG_SIZE} dari config). "
                        "Menata ulang Resize/Pad/CenterCrop; selaraskan saat evaluasi.")
    p.add_argument("--seed", type=int, default=C.SEED,
                   help="Seed acak. Untuk uji beberapa seed, pasangkan dengan "
                        "--run-name berbeda agar checkpoint/log tidak saling menimpa.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--test-per-class", type=int, default=None,
                   help="Selain evaluasi test penuh, evaluasi juga subset stratified "
                        "N citra per kelas (mis. 300) untuk pembanding. Default: hanya penuh.")

    # --- Arsitektur keseluruhan ---
    #   cloud    : CloudDenseNet (DenseNet + modifikasi, + CloudViT bila --vit). Default.
    #   vit-only : CloudViT saja -> Dropout -> Linear(7). Tanpa DenseNet & tanpa
    #              bobot ImageNet (latih dari nol). Toggle --se/--scconv/--vit/--br
    #              diabaikan. Contoh:
    #   python scripts/train.py --arch vit-only --run-name abl_vitOnly
    p.add_argument("--arch", choices=("cloud", "vit-only"), default="cloud",
                   help="'cloud' = CloudDenseNet (default); 'vit-only' = cabang CloudViT "
                        "berdiri sendiri sebagai pengklasifikasi (ablasi).")

    # --- Toggle komponen arsitektur untuk ABLASI (default = nilai config.py) ---
    # Pakai --no-xxx untuk mematikan. Contoh baseline polos:
    #   python scripts/train.py --run-name abl_base --no-se --no-scconv --no-vit --no-br
    # Pasangkan selalu dengan --run-name unik agar checkpoint/log tak saling timpa.
    p.add_argument("--se", action=argparse.BooleanOptionalAction, default=C.USE_SE,
                   help="SEBlock (atensi channel). --no-se -> semua SE jadi Identity.")
    p.add_argument("--scconv", action=argparse.BooleanOptionalAction, default=C.USE_SCCONV,
                   help="SCConv. --no-scconv -> conv 3x3 standar (Conv2d).")
    p.add_argument("--vit", action=argparse.BooleanOptionalAction, default=C.USE_VIT,
                   help="Cabang CloudViT + fusi. --no-vit -> hanya backbone DenseNet.")
    p.add_argument("--br", action=argparse.BooleanOptionalAction, default=C.USE_BR,
                   help="Kanal lambda B/R di SpatialAttnBR (kebaruan sadar-fisika). "
                        "--no-br -> lam=None, SpatialAttnBR merosot jadi SpatialAttn biasa.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not C.DEBUG else "cpu")
    print(f"Device: {device} | DEBUG: {C.DEBUG} | SEED: {args.seed} | "
          f"batch {args.batch_size} | img {args.img_size}")

    if not args.data_root.exists():
        raise SystemExit(
            f"Dataset tidak ditemukan di {args.data_root}\n"
            "Letakkan dataset GCD (train/val/test) di sana — lihat data/README.md."
        )

    start_now = datetime.now()
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = C.CKPT_DIR / f"{args.run_name}{C.VERSION}.pth"
    last_path = C.CKPT_DIR / f"{args.run_name}{C.VERSION}_last.pth"

    train_loader, val_loader, train_ds, val_ds = build_loaders(
        args.data_root, device, batch_size=args.batch_size, img_size=args.img_size)
    classes = getattr(val_ds, "classes", C.CLASS_NAMES)
    print(f"Kelas ({len(classes)}): {classes}")

    # Terapkan toggle ablasi. SE/SCConv/ViT di-thread lewat konstruktor (nilai config
    # terikat saat import tak bisa cuma ditimpa). USE_BR disinkronkan ke modul config
    # agar siapkan_batch (data.py) & laporan membacanya saat runtime.
    if args.arch == "vit-only":
        # CloudViT berdiri sendiri: tak ada bobot ImageNet -> fase frozen/partial
        # tak membekukan apa pun (imagenet_keys kosong); jadwal LR PHASES tetap
        # dipakai agar sebanding dengan run fusi. λ B/R dihitung tapi diabaikan model.
        model = CloudViTOnly().to(device)
        print("Arsitektur -> vit-only (CloudViT + head linear, latih dari nol; "
              "toggle --se/--scconv/--vit/--br diabaikan)")
    else:
        C.USE_SE, C.USE_SCCONV, C.USE_VIT, C.USE_BR = args.se, args.scconv, args.vit, args.br
        print(f"Ablasi -> SE:{args.se} | SCConv:{args.scconv} | CloudViT:{args.vit} | "
              f"BR(λ):{args.br}")
        model = CloudDenseNet(use_se=args.se, use_scconv=args.scconv,
                              use_vit=args.vit).to(device)
        model = muat_imagenet(model)
        model = nolkan_input_nd(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Parameter total: {n_par:,}")

    criterion = FocalLoss() if C.FOCAL_GAMMA > 0.0 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    writer = siapkan_tensorboard(
        C.TB_LOG_DIR, args.run_name,
        aktif=C.USE_TENSORBOARD, auto_install=C.TB_AUTO_INSTALL,
        auto_launch=C.TB_AUTO_LAUNCH, port=C.TB_PORT,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_epoch, start_epoch, current_phase = -1.0, None, 1, 0
    laporan_meta = None

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if device.type == "cuda" and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        history = ck["history"]
        best_f1 = ck.get("best_f1", max(history["val_f1"], default=-1.0))
        best_epoch = ck.get("best_epoch")
        start_epoch = ck["epoch"] + 1
        laporan_meta = ck.get("laporan")
        print(f"Resume dari epoch {start_epoch} | best_f1 {best_f1:.4f}")

    # SATU folder keluaran per pelatihan. Saat resume, lanjutkan folder yang sama
    # (tersimpan di checkpoint) agar log, kurva, CM, Grad-CAM, & laporan menyatu.
    resumed = start_epoch > 1
    if laporan_meta and laporan_meta.get("run_dir"):
        run_dir = Path(laporan_meta["run_dir"])
        started_at = laporan_meta.get("started_at", start_now.isoformat())
        sesi = list(laporan_meta.get("sesi", []))
    else:
        run_dir = C.RESULT_DIR / f"{args.run_name}_{start_now:%d-%m-%Y_%H%M}"
        started_at = start_now.isoformat()
        sesi = []
    run_dir.mkdir(parents=True, exist_ok=True)
    sesi.append({"mulai": start_now.isoformat(), "start_epoch": start_epoch, "resume": resumed})
    print(f"Folder keluaran pelatihan: {run_dir}")

    def _laporan_meta():
        return {"run_dir": str(run_dir), "started_at": started_at, "sesi": sesi}

    # Log per-epoch ke satu file CSV di folder run (append; aman untuk resume).
    epoch_logger = EpochLogger(run_dir / f"{args.run_name}_log.csv")

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
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                                                 device, writer=writer, epoch=epoch)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)

        if not is_finite(tr_loss, va_loss):
            print("[STOP] Loss NaN/Inf -> berhenti; checkpoint terakhir tetap bersih")
            break

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_f1]):
            history[k].append(v)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        catat_epoch(writer, epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1, lr_now)
        epoch_logger.catat(epoch, phase, tr_loss, tr_acc, va_loss, va_acc, va_f1, lr_now, dt)
        print(f"Epoch {epoch:02d}/{args.epochs} [tahap {phase}] | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f} | {dt:.0f}s")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_f1": best_f1, "best_epoch": best_epoch}, ckpt_path)
            print(f"Epoch {epoch:02d}: [OK] model terbaik disimpan (f1={best_f1:.4f})")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best_f1": best_f1, "best_epoch": best_epoch,
                    "laporan": _laporan_meta()}, last_path)

    finished_now = datetime.now()
    if history["train_loss"]:
        # Laporan dibuat dari model TERBAIK (checkpoint f1 tertinggi), bukan state
        # epoch terakhir, agar angka cocok dengan performa yang dilaporkan.
        if ckpt_path.exists():
            best = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best["model"])
            best_f1 = best.get("best_f1", best_f1)
            best_epoch = best.get("best_epoch", best_epoch)
            print(f"[eval] muat model terbaik {ckpt_path.name} (f1={best_f1:.4f})")

        # Test penuh (held-out <data_root>/test, 9000 citra). None bila tak ada.
        test_loader, _ = build_test_loader(
            args.data_root, device, batch_size=args.batch_size, img_size=args.img_size)

        # Laporan lengkap (CSV + CM val/test + kurva loss & akurasi terpisah +
        # Grad-CAM benar/salah + Markdown) ke SATU folder pelatihan `run_dir`.
        meta = {
            "checkpoint": str(ckpt_path.relative_to(C.PROJECT_DIR)),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "total_epoch": len(history["train_loss"]),
            "started_at": started_at,
            "finished_at": finished_now.isoformat(),
            "resume": len(sesi) > 1 or resumed,
            "sesi": sesi,
            "log_csv": str(run_dir / f"{args.run_name}_log.csv"),
        }
        buat_laporan_lengkap(
            model, device, classes, run_dir, args.run_name, meta,
            val_loader=val_loader, test_loader=test_loader, history=history)

        # (opsional) Confusion matrix subset test stratified N/kelas — pembanding.
        if args.test_per_class is not None and test_loader is not None:
            sub_loader, _ = build_test_loader(args.data_root, device,
                                              per_class=args.test_per_class,
                                              batch_size=args.batch_size,
                                              img_size=args.img_size)
            if sub_loader is not None:
                print(f"[eval] confusion matrix — test subset ({args.test_per_class}/kelas)")
                plot_confusion_matrix(
                    model, sub_loader, device, classes=classes,
                    simpan=str(run_dir / f"{args.run_name}_cm_test{args.test_per_class}.png"))
    else:
        print("[report] history kosong — laporan metrik dilewati.")

    tutup_tensorboard(writer)


if __name__ == "__main__":
    main()
