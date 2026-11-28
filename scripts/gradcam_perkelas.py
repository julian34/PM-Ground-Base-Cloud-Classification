"""Grad-CAM per kelas (prediksi BENAR & SALAH + tingkat kepercayaan) untuk laporan.

Skrip mandiri — memakai model yang SUDAH DILATIH (checkpoint .pth) dan modul yang
ada (`bangun_dari_checkpoint`, `GradCAM`, `build_transforms`, `siapkan_batch`)
tanpa mengubahnya.

Alur:
  1. Prediksi seluruh citra split (default test) -> probabilitas softmax tiap citra.
  2. Per kelas (label asli): pilih K sampel yang diprediksi benar & K yang salah.
  3. Grad-CAM (layer `norm5`, 7x7 terakhir backbone DenseNet):
       - benar : CAM kelas prediksi
       - salah : CAM kelas prediksi (yang salah) + CAM kelas asli (pembanding)
     plus diagram batang probabilitas 7 kelas (tingkat kepercayaan).

Keluaran (default: laporan/gambar/grad-map/):
    <kelas>/benar/01_<file>.png           panel per sampel (asli | CAM | probabilitas)
    <kelas>/salah/01_<file>.png           panel per sampel (asli | CAM pred | CAM asli | prob)
    <kelas>/<kelas>_benar.png             grid seluruh sampel benar kelas tsb
    <kelas>/<kelas>_salah.png             grid seluruh sampel salah kelas tsb
    <kelas>/polos/...                     (opsional --polos) overlay tanpa teks
    ringkasan_benar.png / ringkasan_salah.png    1 sampel per kelas, semua kelas
    prediksi_<split>.csv                  prediksi + confidence semua citra
    ringkasan_perkelas.csv                jumlah benar/salah, akurasi, rata2 confidence

Contoh:
    python scripts/gradcam_perkelas.py
    python scripts/gradcam_perkelas.py --k 6 --pilih yakin
    python scripts/gradcam_perkelas.py --checkpoint artifacts/checkpoints/cloud_Versi01.pth --polos
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clouddensenet import config as C  # noqa: E402
from clouddensenet.checkpoint import bangun_dari_checkpoint  # noqa: E402
from clouddensenet.data import build_transforms, set_seed, siapkan_batch  # noqa: E402
from clouddensenet.gradcam import GradCAM, overlay_cam  # noqa: E402

HIJAU, MERAH, BIRU, ABU = "#2e7d32", "#c62828", "#1565c0", "#9e9e9e"


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM per kelas: benar/salah + confidence")
    p.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="default: checkpoint .pth terbaru di artifacts/checkpoints/")
    p.add_argument("--out", type=Path, default=ROOT / "laporan" / "gambar" / "grad-map")
    p.add_argument("--k", type=int, default=4, help="jumlah sampel benar & salah per kelas")
    p.add_argument("--pilih", choices=["acak", "yakin", "ragu"], default="acak",
                   help="acak = seeded; yakin = confidence tertinggi; ragu = terendah")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--polos", action="store_true", help="juga simpan overlay tanpa teks")
    p.add_argument("--alpha", type=float, default=0.45, help="transparansi heatmap")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=C.SEED)
    return p.parse_args()


def cari_checkpoint(path):
    if path is not None:
        if not path.exists():
            raise SystemExit(f"Checkpoint tidak ditemukan: {path}")
        return path
    kandidat = sorted((p for p in C.CKPT_DIR.glob("*.pth") if not p.stem.endswith("_last")),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not kandidat:
        raise SystemExit(f"Tidak ada checkpoint .pth di {C.CKPT_DIR} (pakai --checkpoint)")
    return kandidat[0]


def pendek(nama):
    """'1_cumulus' -> 'cumulus' (label ringkas untuk grafik)."""
    return nama.split("_", 1)[1] if "_" in nama and nama.split("_", 1)[0].isdigit() else nama


# --------------------------------------------------------------------------- #
# 1. Prediksi seluruh split
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prediksi_semua(model, ds, device, batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=C.NUM_WORKERS,
                        pin_memory=device.type == "cuda")
    probs, labels = [], []
    for x01, y in tqdm(loader, desc="prediksi", unit="batch"):
        x, lam = siapkan_batch(x01, device)
        probs.append(model(x, lam).float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def pilih_indeks(idx, skor, k, mode, rng):
    if len(idx) == 0:
        return idx
    if mode == "acak":
        return rng.choice(idx, size=min(k, len(idx)), replace=False)
    urut = idx[np.argsort(skor[idx])]
    return (urut[::-1] if mode == "yakin" else urut)[:k]


# --------------------------------------------------------------------------- #
# 2. Grad-CAM
# --------------------------------------------------------------------------- #
@torch.enable_grad()
def hitung_cam(model, gc, x01, kelas, device):
    """x01 [B,3,H,W] -> cam [B,H,W] untuk `kelas` [B] (tensor long)."""
    x, lam = siapkan_batch(x01, device)
    cam, _, _ = gc(x, lam, kelas=kelas.to(device))
    return cam


# --------------------------------------------------------------------------- #
# 3. Gambar
# --------------------------------------------------------------------------- #
def gambar_prob(ax, prob, y, p, classes):
    n = len(classes)
    warna = [ABU] * n
    warna[y] = BIRU
    warna[p] = HIJAU if p == y else MERAH
    pos = np.arange(n)[::-1]
    ax.barh(pos, prob, color=warna, height=0.7)
    ax.set_yticks(pos)
    ax.set_yticklabels([pendek(c) for c in classes], fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("probabilitas (softmax)", fontsize=7)
    for b, v in zip(pos, prob):
        if v >= 0.01:
            ax.text(min(v + 0.02, 0.86), b, f"{v:.2f}", va="center", fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def isi_baris(ax, img, cam_pred, cam_asli, prob, y, p, classes, alpha, judul_kolom):
    """Isi satu baris panel. ax: list axes (3 kolom bila benar, 4 bila salah)."""
    benar = (p == y)
    warna = HIJAU if benar else MERAH
    ax[0].imshow(img)
    ax[0].set_title(f"Asli: {pendek(classes[y])}" if not judul_kolom else
                    f"Citra asli\nlabel: {pendek(classes[y])}", fontsize=8)
    ax[1].imshow(overlay_cam(img, cam_pred, alpha=alpha))
    ax[1].set_title(f"Grad-CAM prediksi: {pendek(classes[p])}\nconf {prob[p]:.1%} "
                    f"[{'BENAR' if benar else 'SALAH'}]", fontsize=8, color=warna)
    j = 2
    if cam_asli is not None:
        ax[2].imshow(overlay_cam(img, cam_asli, alpha=alpha))
        ax[2].set_title(f"Grad-CAM label asli: {pendek(classes[y])}\nprob {prob[y]:.1%}",
                        fontsize=8, color=BIRU)
        j = 3
    for a in ax[:j]:
        a.axis("off")
    gambar_prob(ax[j], prob, y, p, classes)


def simpan_panel(img, cam_pred, cam_asli, prob, y, p, classes, nama_file, path, alpha, dpi):
    ncol = 4 if cam_asli is not None else 3
    fig, ax = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.2),
                           gridspec_kw={"width_ratios": [1] * (ncol - 1) + [1.1]})
    isi_baris(ax, img, cam_pred, cam_asli, prob, y, p, classes, alpha, judul_kolom=True)
    fig.suptitle(nama_file, fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def simpan_grid(baris, classes, path, alpha, dpi, judul):
    """baris: list dict(img, cam_pred, cam_asli, prob, y, p)."""
    if not baris:
        return False
    ncol = 4 if baris[0]["cam_asli"] is not None else 3
    fig, ax = plt.subplots(len(baris), ncol, figsize=(3.0 * ncol, 3.0 * len(baris)),
                           squeeze=False,
                           gridspec_kw={"width_ratios": [1] * (ncol - 1) + [1.1]})
    for r, b in enumerate(baris):
        isi_baris(ax[r], b["img"], b["cam_pred"], b["cam_asli"], b["prob"], b["y"], b["p"],
                  classes, alpha, judul_kolom=False)
    fig.suptitle(judul, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def simpan_polos(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = cari_checkpoint(args.checkpoint)
    model, info = bangun_dari_checkpoint(ckpt, device=device)
    model.eval()
    print(f"Device: {device} | model: {ckpt.name} (epoch={info['epoch']}, "
          f"best_f1={info['best_f1']})")

    folder = args.data_root / args.split
    if not folder.is_dir():
        raise SystemExit(f"Folder split tidak ada: {folder}")
    _, val_tf = build_transforms(args.img_size)
    ds = ImageFolder(root=str(folder), transform=val_tf)
    classes = ds.classes
    print(f"{args.split}: {len(ds)} citra, {len(classes)} kelas")

    probs, labels = prediksi_semua(model, ds, device, args.batch_size)
    preds = probs.argmax(1)
    conf = probs.max(1)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- CSV prediksi & ringkasan ---
    with open(args.out / f"prediksi_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "prediksi", "benar", "conf_prediksi", "prob_label"]
                   + [f"p_{c}" for c in classes])
        for i, (path, _) in enumerate(ds.samples):
            w.writerow([Path(path).name, classes[labels[i]], classes[preds[i]],
                        int(preds[i] == labels[i]), f"{conf[i]:.4f}",
                        f"{probs[i, labels[i]]:.4f}"] + [f"{v:.4f}" for v in probs[i]])

    ringkas = []
    print(f"\n{'kelas':<18}{'n':>6}{'benar':>7}{'salah':>7}{'akurasi':>9}"
          f"{'conf_B':>8}{'conf_S':>8}")
    for c, nama in enumerate(classes):
        m = labels == c
        b, s = m & (preds == c), m & (preds != c)
        row = dict(kelas=nama, n=int(m.sum()), benar=int(b.sum()), salah=int(s.sum()),
                   akurasi=b.sum() / max(m.sum(), 1),
                   conf_rata_benar=conf[b].mean() if b.any() else float("nan"),
                   conf_rata_salah=conf[s].mean() if s.any() else float("nan"))
        ringkas.append(row)
        print(f"{nama:<18}{row['n']:>6}{row['benar']:>7}{row['salah']:>7}"
              f"{row['akurasi']:>9.2%}{row['conf_rata_benar']:>8.3f}{row['conf_rata_salah']:>8.3f}")
    print(f"{'TOTAL':<18}{len(labels):>6}{int((preds == labels).sum()):>7}"
          f"{int((preds != labels).sum()):>7}{(preds == labels).mean():>9.2%}\n")
    with open(args.out / "ringkasan_perkelas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0]))
        w.writeheader()
        for r in ringkas:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # --- Grad-CAM per kelas ---
    target = getattr(model, "gradcam_target", None) or model.norm5
    gc = GradCAM(model, target)
    wakil = {"benar": [], "salah": []}
    try:
        for c, nama in enumerate(classes):
            d_kelas = args.out / nama
            for jenis in ("benar", "salah"):
                idx_semua = np.where((labels == c) & ((preds == c) == (jenis == "benar")))[0]
                idx = pilih_indeks(idx_semua, conf, args.k, args.pilih, rng)
                if len(idx) == 0:
                    print(f"[{nama}] tidak ada sampel {jenis} — dilewati")
                    continue
                d = d_kelas / jenis
                d.mkdir(parents=True, exist_ok=True)

                x01 = torch.stack([ds[int(i)][0] for i in idx])
                p_t = torch.as_tensor(preds[idx], dtype=torch.long)
                cam_pred = hitung_cam(model, gc, x01, p_t, device)
                cam_asli = None
                if jenis == "salah":
                    y_t = torch.as_tensor(labels[idx], dtype=torch.long)
                    cam_asli = hitung_cam(model, gc, x01, y_t, device)

                baris = []
                for j, i in enumerate(idx):
                    img = x01[j].permute(1, 2, 0).numpy()
                    nama_file = Path(ds.samples[int(i)][0]).name
                    b = dict(img=img, cam_pred=cam_pred[j],
                             cam_asli=None if cam_asli is None else cam_asli[j],
                             prob=probs[i], y=int(labels[i]), p=int(preds[i]))
                    baris.append(b)
                    stem = f"{j + 1:02d}_{Path(nama_file).stem}"
                    simpan_panel(**b, classes=classes, nama_file=nama_file,
                                 path=d / f"{stem}.png", alpha=args.alpha, dpi=args.dpi)
                    if args.polos:
                        dp = d_kelas / "polos"
                        dp.mkdir(exist_ok=True)
                        simpan_polos(img, dp / f"{jenis}_{stem}_asli.png")
                        simpan_polos(overlay_cam(img, b["cam_pred"], alpha=args.alpha),
                                     dp / f"{jenis}_{stem}_cam_pred-{pendek(classes[b['p']])}.png")
                        if b["cam_asli"] is not None:
                            simpan_polos(overlay_cam(img, b["cam_asli"], alpha=args.alpha),
                                         dp / f"{jenis}_{stem}_cam_asli-{pendek(nama)}.png")

                judul = (f"{nama} — prediksi {jenis.upper()} "
                         f"({len(idx)} dari {len(idx_semua)}, pilih={args.pilih})")
                simpan_grid(baris, classes, d_kelas / f"{nama}_{jenis}.png",
                            args.alpha, args.dpi, judul)
                wakil[jenis].append(baris[0])
                print(f"[{nama}] {jenis}: {len(idx)} sampel -> {d}")
    finally:
        gc.remove()

    for jenis, baris in wakil.items():
        simpan_grid(baris, classes, args.out / f"ringkasan_{jenis}.png", args.alpha, args.dpi,
                    f"Grad-CAM — contoh prediksi {jenis.upper()} tiap kelas ({args.split})")
    print(f"[OK] tersimpan di {args.out}")


if __name__ == "__main__":
    main()
