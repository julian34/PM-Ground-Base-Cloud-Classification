"""Integrasi TensorBoard: pasang & aktifkan otomatis saat training.

Modul ini menjaga agar paket `tensorboard` tersedia (pasang otomatis bila belum
ada), membuat `SummaryWriter`, dan—opsional—menjalankan server TensorBoard di
latar belakang sehingga dasbor langsung aktif begitu training dimulai.

Semua fungsi aman-gagal (fail-safe): bila TensorBoard tidak tersedia atau gagal
dijalankan, training TIDAK boleh ikut gagal — fungsi hanya mengembalikan None
dan mencetak peringatan. Logika ini dipisah dari `engine.py` agar loop training
tetap murni/parameterized dan mudah diuji.
"""
from __future__ import annotations

import atexit
import subprocess
import sys
from pathlib import Path

# Handle proses server yang sedang berjalan (agar bisa ditutup saat keluar).
_server_proc: subprocess.Popen | None = None


def _pastikan_terpasang(auto_install: bool) -> bool:
    """Pastikan paket `tensorboard` bisa di-import; pasang via pip bila perlu."""
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        if not auto_install:
            print("[TB] paket 'tensorboard' tidak ada (TB_AUTO_INSTALL=False) — dilewati")
            return False
    print("[TB] 'tensorboard' belum terpasang — memasang otomatis via pip ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "tensorboard>=2.14"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[TB] gagal memasang tensorboard ({e}) — pemantauan dinonaktifkan")
        return False
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        print("[TB] tensorboard tetap tak bisa di-import setelah instalasi — dilewati")
        return False


def _jalankan_server(log_dir: Path, port: int) -> None:
    """Jalankan `tensorboard --logdir ... --port ...` di latar belakang (sekali)."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        return  # sudah berjalan
    try:
        _server_proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        atexit.register(tutup_tensorboard)
        print(f"[TB] server aktif  ->  http://localhost:{port}  (logdir: {log_dir})")
    except OSError as e:
        print(f"[TB] gagal menjalankan server ({e}) — event file tetap ditulis")
        _server_proc = None


def siapkan_tensorboard(log_dir, run_name, *, aktif=True, auto_install=True,
                        auto_launch=True, port=6006):
    """Siapkan SummaryWriter (+ server opsional). Mengembalikan writer atau None.

    Parameter sengaja eksplisit (bukan baca global config) agar mudah diuji.
    """
    if not aktif:
        return None
    if not _pastikan_terpasang(auto_install):
        return None

    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(log_dir)
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    if auto_launch:
        _jalankan_server(log_dir, port)   # logdir induk -> semua run tampil
    return writer


def catat_epoch(writer, epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1, lr):
    """Tulis scalar satu epoch. No-op bila writer None."""
    if writer is None:
        return
    writer.add_scalar("Loss/train", tr_loss, epoch)
    writer.add_scalar("Loss/val", va_loss, epoch)
    writer.add_scalar("Acc/train", tr_acc, epoch)
    writer.add_scalar("Acc/val", va_acc, epoch)
    writer.add_scalar("F1/val", va_f1, epoch)
    writer.add_scalar("LR", lr, epoch)
    writer.flush()


def tutup_tensorboard(writer=None):
    """Tutup writer & hentikan server (dipanggil manual atau via atexit)."""
    global _server_proc
    if writer is not None:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 — penutupan tak boleh menggagalkan training
            pass
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None
