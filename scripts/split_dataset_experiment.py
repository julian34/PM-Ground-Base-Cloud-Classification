"""Split ulang dataset PENELITIAN B -> data/processed/<DATASET_B_NAME>/{train,val,test}.

Berbeda dari scripts/split_dataset.py (dataset GCD asli). Script ini untuk dataset
BARU yang arsipnya sudah tersusun per-kelas, dengan alur:

    1) ekstrak DATASET_B_ZIP,
    2) gabung semua gambar jadi satu pool per-kelas  -> <out>/_merged/<kelas>/,
    3) split 3-arah stratified & seeded               -> <out>/{train,val,test}/<kelas>/.

Nilai default dibaca dari src/clouddensenet/config.py:
    DATASET_B_ZIP, DATASET_B_ROOT, DATASET_B_SPLIT (train,val,test), DATASET_B_SEED.
Atur rasio/jumlah aktual di config.py, lalu jalankan script ini.

Dataset ASLI (data/processed/{train,val,test}) TIDAK pernah disentuh: script ini
hanya menulis & menghapus di dalam sub-folder DATASET_B_ROOT, dan menolak berjalan
bila output ternyata sama dengan atau di luar data/processed.

Contoh:
    python scripts/split_dataset_experiment.py                 # copy (zip tetap utuh)
    python scripts/split_dataset_experiment.py --move          # hemat disk (pindah)
    python scripts/split_dataset_experiment.py --force          # timpa output lama
    python scripts/split_dataset_experiment.py --keep-merged    # simpan pool _merged
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    """Muat src/clouddensenet/config.py terisolasi (ringan: hanya pathlib).

    Sengaja TIDAK `import clouddensenet` agar tak menarik torch lewat __init__.py —
    split data cukup butuh konstanta path/rasio, bukan model.
    """
    cfg_path = ROOT / "src" / "clouddensenet" / "config.py"
    spec = importlib.util.spec_from_file_location("cdn_config", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load_config()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MERGED_DIRNAME = "_merged"
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(
        description="Split ulang dataset Penelitian B -> processed/<name>/{train,val,test}")
    p.add_argument("--zip", type=Path, default=config.DATASET_B_ZIP,
                   help="Arsip dataset baru (struktur per-kelas). Default: config.DATASET_B_ZIP")
    p.add_argument("--out", type=Path, default=config.DATASET_B_ROOT,
                   help="Folder output eksperimen. Default: config.DATASET_B_ROOT")
    p.add_argument("--split", type=float, nargs=3, metavar=("TR", "VA", "TE"),
                   default=list(config.DATASET_B_SPLIT),
                   help="Rasio train val test (jumlah=1.0). Default: config.DATASET_B_SPLIT")
    p.add_argument("--seed", type=int, default=config.DATASET_B_SEED)
    p.add_argument("--move", action="store_true",
                   help="Pindahkan file (bukan copy) saat menggabung. Hemat disk.")
    p.add_argument("--keep-merged", action="store_true",
                   help="Simpan pool <out>/_merged/ setelah split (default: dihapus).")
    p.add_argument("--force", action="store_true",
                   help="Timpa folder output bila sudah ada.")
    return p.parse_args()


def guard_output(out: Path) -> Path:
    """Pastikan output adalah SUB-folder di dalam data/processed, bukan processed itu sendiri.

    Mencegah script tak sengaja menghapus/menimpa dataset asli (processed/{train,val,test}).
    """
    out = out.resolve()
    processed = config.DATA_ROOT.resolve()
    if out == processed:
        sys.exit(f"[ERROR] output tidak boleh = {processed} (itu dataset asli). "
                 f"Pakai sub-folder, mis. {processed / 'dataset_b'}")
    if processed not in out.parents:
        sys.exit(f"[ERROR] output {out} harus berada di dalam {processed}.")
    return out


def ensure_extracted(zip_path: Path, extract_dir: Path) -> Path:
    """Ekstrak zip ke extract_dir (idempoten). Mengembalikan folder hasil ekstrak."""
    if extract_dir.is_dir() and any(extract_dir.iterdir()):
        print(f"[ekstrak] sudah ada, pakai ulang: {extract_dir}")
        return extract_dir
    if not zip_path.exists():
        sys.exit(f"[ERROR] zip tidak ditemukan: {zip_path}\n"
                 f"        Taruh arsip dataset baru di sana, atau beri --zip <path>.")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ekstrak] {zip_path} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    print("[ekstrak] selesai")
    return extract_dir


def collect_by_class(src: Path, classes: list[str]) -> dict[str, list[Path]]:
    """Kumpulkan semua gambar di bawah src, dikelompokkan per-kelas.

    Sebuah gambar masuk kelas X bila SALAH SATU folder leluhurnya bernama X
    (mis. .../train/1_cumulus/a.jpg dan .../test/1_cumulus/b.jpg sama-sama -> 1_cumulus).
    Folder yang namanya bukan salah satu kelas diabaikan (dilaporkan di akhir).
    """
    class_set = set(classes)
    pools: dict[str, list[Path]] = {c: [] for c in classes}
    unknown: set[str] = set()
    for f in src.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in IMG_EXT):
            continue
        hit = next((p for p in f.parts if p in class_set), None)
        if hit is None:
            unknown.add(f.parent.name)
        else:
            pools[hit].append(f)
    if unknown:
        print(f"[warn] folder/kelas tak dikenal (dilewati): {sorted(unknown)}")
    for c in classes:
        pools[c].sort()
    return pools


def unique_dst(dst_dir: Path, name: str) -> Path:
    """Hindari tabrakan nama saat menggabung dari beberapa sub-folder."""
    cand = dst_dir / name
    if not cand.exists():
        return cand
    stem, suf = Path(name).stem, Path(name).suffix
    i = 1
    while (dst_dir / f"{stem}_{i}{suf}").exists():
        i += 1
    return dst_dir / f"{stem}_{i}{suf}"


def transfer(src: Path, dst_dir: Path, move: bool) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = unique_dst(dst_dir, src.name)
    (shutil.move if move else shutil.copy2)(str(src), str(dst))
    return dst


def split_counts(n: int, frac: tuple[float, float, float]) -> tuple[int, int, int]:
    """Bagi n item -> (train, val, test) sesuai frac; sisa pembulatan masuk ke train."""
    n_val = min(int(round(n * frac[1])), n)
    n_test = min(int(round(n * frac[2])), n - n_val)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def main():
    a = parse_args()
    out = guard_output(a.out)

    frac = tuple(a.split)
    if abs(sum(frac) - 1.0) > 1e-6:
        sys.exit(f"[ERROR] rasio split {frac} harus berjumlah 1.0 (sekarang {sum(frac):.3f}).")

    if out.exists():
        if not a.force:
            sys.exit(f"[ERROR] {out} sudah ada. Pakai --force untuk menimpa "
                     f"(hanya folder ini yang dihapus, dataset asli aman).")
        print(f"[bersih] menghapus {out}")
        shutil.rmtree(out)

    # 1) ekstrak
    extract_dir = out / "_extracted"
    ensure_extracted(a.zip, extract_dir)

    # 2) gabung per-kelas -> _merged/<kelas>/
    classes = config.CLASS_NAMES
    pools = collect_by_class(extract_dir, classes)
    merged = out / MERGED_DIRNAME
    print(f"\n[gabung] menyusun pool per-kelas di {merged} ...")
    merged_pools: dict[str, list[Path]] = {}
    for c in classes:
        dst_files = [transfer(f, merged / c, a.move) for f in pools[c]]
        merged_pools[c] = sorted(dst_files)
        print(f"  {c:<18}{len(dst_files):>6} gambar")

    # 3) split 3-arah stratified & seeded, dari pool _merged
    rng = np.random.default_rng(a.seed)
    print(f"\n{'kelas':<18}{'train':>8}{'val':>7}{'test':>8}")
    print("-" * 41)
    tot = {"train": 0, "val": 0, "test": 0}
    empty = []
    for c in classes:
        files = merged_pools[c]
        if not files:
            empty.append(c)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_tr, n_va, n_te = split_counts(len(files), frac)
        groups = {
            "train": idx[:n_tr],
            "val": idx[n_tr:n_tr + n_va],
            "test": idx[n_tr + n_va:n_tr + n_va + n_te],
        }
        for split in SPLITS:
            for i in sorted(groups[split].tolist()):
                # pindahkan dari _merged ke split (pool jadi hasil akhir)
                transfer(files[i], out / split / c, move=True)
            tot[split] += len(groups[split])
        print(f"{c:<18}{n_tr:>8}{n_va:>7}{n_te:>8}")

    print("-" * 41)
    print(f"{'TOTAL':<18}{tot['train']:>8}{tot['val']:>7}{tot['test']:>8}")
    if empty:
        print(f"[warn] kelas tanpa gambar: {empty} — cek struktur zip & CLASS_NAMES.")

    # bersih-bersih intermediate
    shutil.rmtree(extract_dir, ignore_errors=True)
    if a.keep_merged:
        print(f"[info] pool _merged dipertahankan: {merged}")
    else:
        shutil.rmtree(merged, ignore_errors=True)

    print(f"\n[selesai] split={frac}, seed={a.seed}, metode gabung="
          f"{'pindah' if a.move else 'copy'}")
    print(f"[hasil]   {out}/{{train,val,test}}")
    print(f"[latih]   python scripts/train.py --data-root "
          f"{out.relative_to(config.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
