"""Bandingkan beberapa hasil Stratified K-Fold CV (read-only).

Membaca file `artifacts/results/kfold_<run>.json` yang ditulis `train_kfold.py`,
mencetak tabel `mean ± std` + flag arsitektur tiap run, lalu menerapkan **aturan
tumpang-tindih ±std** terhadap baseline (run pertama): selisih dianggap NYATA hanya
bila |Δmean| > (std_a + std_b); selain itu kemungkinan cuma derau.

Tidak menulis/mengubah apa pun.

Contoh:
    python scripts/compare_kfold.py                      # semua kfold_*.json (baseline = pertama)
    python scripts/compare_kfold.py baseline br_off aug_on   # run tertentu; yang pertama = baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:                                   # konsol Windows (cp1252) tak bisa cetak ± / Δ -> paksa UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from clouddensenet import config as C  # noqa: E402

ARCH_KEYS = ["USE_BR", "STRONG_AUG", "USE_SE", "USE_SCCONV", "IMBALANCE"]


def _load(run_or_path):
    """Terima nama run ('baseline') atau path JSON; kembalikan (nama, dict) atau None."""
    p = Path(run_or_path)
    if not p.suffix:
        p = C.RESULT_DIR / f"kfold_{run_or_path}.json"
    if not p.exists():
        print(f"[lewati] tidak ditemukan: {p}")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    nama = p.stem.replace("kfold_", "")
    return nama, data


def _fmt_arch(arch):
    if not arch:
        return "(flag tak tercatat — run lama)"
    return " ".join(f"{k}={arch.get(k)}" for k in ARCH_KEYS if k in arch)


def main():
    args = sys.argv[1:]
    if args:
        items = [_load(a) for a in args]
    else:
        files = sorted((C.RESULT_DIR).glob("kfold_*.json"))
        items = [_load(str(f)) for f in files]
    items = [it for it in items if it]

    if not items:
        raise SystemExit("Tidak ada hasil CV untuk dibandingkan. Jalankan train_kfold.py dulu.")

    # ---- tabel ringkasan ----
    print("\n" + "=" * 78)
    print(f"{'run':<16}{'mean_f1':>9}{'std':>8}{'OOF':>9}   flag")
    print("-" * 78)
    for nama, d in items:
        print(f"{nama:<16}{d['mean_f1']:>9.4f}{d['std_f1']:>8.4f}"
              f"{d.get('oof_macro_f1', float('nan')):>9.4f}   {_fmt_arch(d.get('arch'))}")
    print("=" * 78)

    if len(items) == 1:
        print("Hanya satu run — tidak ada perbandingan.")
        return

    # ---- perbandingan vs baseline (run pertama) ----
    base_nama, base = items[0]
    print(f"\nPerbandingan terhadap baseline = '{base_nama}' "
          f"(F1 {base['mean_f1']:.4f} ± {base['std_f1']:.4f}):\n")
    bm, bs = base["mean_f1"], base["std_f1"]
    for nama, d in items[1:]:
        m, s = d["mean_f1"], d["std_f1"]
        delta = m - bm
        gabung_std = bs + s
        nyata = abs(delta) > gabung_std
        arah = "lebih baik" if delta > 0 else "lebih buruk"
        if nyata:
            verdict = f"NYATA ({arah}): |Δ|={abs(delta):.4f} > std_gabungan={gabung_std:.4f}"
        else:
            verdict = (f"mungkin DERAU: |Δ|={abs(delta):.4f} <= std_gabungan={gabung_std:.4f} "
                       f"-> rentang ±std tumpang-tindih")
        print(f"  {nama:<16} F1 {m:.4f} ± {s:.4f} | Δmean {delta:+.4f} -> {verdict}")

    print("\nCatatan: 'DERAU' = perbedaan belum tentu nyata; pilih yang lebih sederhana/"
          "stabil/murah. 'NYATA' = aman memilih yang mean-nya lebih tinggi.")
    print("Angka CV untuk MEMILIH konfigurasi; performa final dilaporkan dari test/ "
          "setelah model final dilatih di seluruh train/.")


if __name__ == "__main__":
    main()
