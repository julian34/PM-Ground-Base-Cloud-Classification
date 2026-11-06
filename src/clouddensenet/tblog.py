"""Integrasi TensorBoard: pasang & aktifkan otomatis saat training.

Modul ini menjaga agar paket `tensorboard` tersedia (pasang otomatis bila belum
ada), membuat `SummaryWriter`, dan—opsional—menjalankan server TensorBoard di
latar belakang sehingga dasbor langsung aktif begitu training dimulai.

Server sengaja dijalankan *lepas* (detached) dari proses training dan TIDAK
dimatikan saat training selesai, supaya dasbor tetap bisa dibuka sesudahnya.
Hentikan manual lewat Ctrl+C di jendelanya atau panggil `hentikan_server()`.

Semua fungsi aman-gagal (fail-safe): bila TensorBoard tidak tersedia atau gagal
dijalankan, training TIDAK boleh ikut gagal — fungsi hanya mengembalikan None
dan mencetak peringatan. Logika ini dipisah dari `engine.py` agar loop training
tetap murni/parameterized dan mudah diuji.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

# Handle proses server yang dijalankan proses ini (untuk hentikan_server()).
_server_proc: subprocess.Popen | None = None


def _port_listen(host: str, port: int) -> bool:
    """True bila sudah ada proses yang mendengarkan di host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _flags_detached() -> dict:
    """kwargs Popen agar server lepas dari proses training (tak ikut mati)."""
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        return {"creationflags": flags}
    return {"start_new_session": True}


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


def _jalankan_server(log_dir: Path, port: int, host: str = "127.0.0.1") -> None:
    """Jalankan server TensorBoard di latar belakang (sekali), lepas dari training.

    - `--host` di-set eksplisit agar cocok dengan http://localhost (IPv4), bukan
      bind ke IPv6/`::` yang kadang tak terjangkau browser di Windows.
    - Bila port sudah dipakai, anggap ada TensorBoard lain yang aktif -> pakai ulang
      (tidak menjalankan server kedua).
    - stdout/stderr ditangkap ke `<log_dir>/_tensorboard.log`, lalu dicek apakah
      proses langsung mati — kegagalan ditampilkan, bukan diam-diam.
    - Server TIDAK didaftarkan ke atexit: dibiarkan hidup agar dasbor tetap terbuka
      setelah training selesai (hentikan via hentikan_server() atau Ctrl+C).
    """
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        return  # sudah kita jalankan di proses ini

    if _port_listen(host, port):
        print(f"[TB] port {port} sudah dipakai — memakai server yang ada  ->  "
              f"http://localhost:{port}")
        return

    log_dir = Path(log_dir)
    log_file = log_dir / "_tensorboard.log"
    try:
        _log_handle = open(log_file, "w", encoding="utf-8")  # noqa: SIM115 — diwarisi anak
        _server_proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(log_dir), "--host", host, "--port", str(port)],
            stdout=_log_handle, stderr=subprocess.STDOUT,
            **_flags_detached(),
        )
    except OSError as e:
        print(f"[TB] gagal menjalankan server ({e}) — event file tetap ditulis")
        _server_proc = None
        return

    # Beri waktu start (maks ~5 dtk), lalu pastikan benar-benar mendengarkan.
    for _ in range(20):
        if _server_proc.poll() is not None:
            break
        if _port_listen(host, port):
            print(f"[TB] server aktif  ->  http://localhost:{port}  (logdir: {log_dir})")
            return
        time.sleep(0.25)

    if _server_proc.poll() is not None:
        # Mati seketika -> tampilkan alasan sebenarnya (mis. port bentrok).
        pesan = ""
        try:
            pesan = "\n      ".join(
                log_file.read_text(encoding="utf-8").strip().splitlines()[-3:]
            )
        except OSError:
            pass
        print(f"[TB] server gagal start (lihat {log_file}):\n      {pesan}")
        _server_proc = None
    else:
        # Masih hidup tapi belum merespons; tetap beri tahu URL-nya.
        print(f"[TB] server dimulai, belum merespons  ->  http://localhost:{port}  "
              f"(bila tak terbuka, cek {log_file})")


def siapkan_tensorboard(log_dir, run_name, *, aktif=True, auto_install=True,
                        auto_launch=True, port=6006, host="127.0.0.1"):
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
        _jalankan_server(log_dir, port, host)   # logdir induk -> semua run tampil
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
    """Tutup writer saja. Server TensorBoard sengaja DIBIARKAN hidup agar dasbor
    tetap bisa dibuka setelah training selesai — hentikan lewat `hentikan_server()`
    atau Ctrl+C di jendelanya."""
    if writer is not None:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 — penutupan tak boleh menggagalkan training
            pass


def hentikan_server():
    """Hentikan server TensorBoard yang dijalankan proses ini (bila ada)."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None
