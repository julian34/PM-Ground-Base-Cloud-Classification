"""Grad-CAM untuk CloudDenseNet.

Menyorot wilayah citra langit yang paling memengaruhi prediksi kelas awan.
Target default: `model.norm5` — feature map 1024x7x7 terakhir (sebelum global
pooling di TopBlock), lokasi standar Grad-CAM pada backbone gaya DenseNet.

Forward CloudDenseNet butuh argumen `lam` (peta rasio Biru/Merah); modul ini
meneruskannya apa adanya agar atensi sadar-fisika ikut terefleksi di peta CAM.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES


class GradCAM:
    """Hitung peta Grad-CAM lewat hook aktivasi + gradien di satu layer target.

    Pemakaian::

        gc = GradCAM(model, model.norm5)
        cam, kelas, logits = gc(x, lam)   # x, lam dari siapkan_batch
        gc.remove()                       # lepas hook saat selesai
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.acts = None
        self.grads = None
        self._h_fwd = target_layer.register_forward_hook(self._simpan_act)
        self._h_bwd = target_layer.register_full_backward_hook(self._simpan_grad)

    def _simpan_act(self, module, inp, out):
        self.acts = out.detach()

    def _simpan_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def remove(self):
        """Lepas kedua hook (wajib dipanggil agar tidak bocor memori/efek samping)."""
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __call__(self, x, lam=None, kelas=None):
        """x [B,3,H,W] ternormalisasi -> (cam [B,H,W] di [0,1], kelas [B], logits [B,C]).

        `kelas=None` memakai kelas prediksi (argmax); beri tensor [B] untuk
        menjelaskan kelas tertentu. Dijalankan dalam mode grad (bukan no_grad).
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x, lam)
        if kelas is None:
            kelas = logits.argmax(1)
        skor = logits.gather(1, kelas.view(-1, 1)).sum()
        skor.backward()

        w = self.grads.mean(dim=(2, 3), keepdim=True)              # GAP gradien -> bobot channel
        cam = F.relu((w * self.acts).sum(1, keepdim=True))         # [B,1,h,w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        flat = cam.flatten(1)                                      # normalisasi per-gambar ke [0,1]
        cmin = flat.min(1).values.view(-1, 1, 1)
        cmax = flat.max(1).values.view(-1, 1, 1)
        cam = (cam - cmin) / (cmax - cmin + 1e-8)
        return cam.cpu().numpy(), kelas.cpu().numpy(), logits.detach()


def overlay_cam(img01, cam, alpha=0.5, cmap="jet"):
    """Gabung citra asli [0,1] dengan heatmap CAM. img01: [3,H,W]/[H,W,3]; cam: [H,W]."""
    if img01.ndim == 3 and img01.shape[0] == 3:
        img01 = np.transpose(img01, (1, 2, 0))
    heat = plt.get_cmap(cmap)(cam)[..., :3]                       # [H,W,3] RGB dari colormap
    return np.clip((1 - alpha) * img01 + alpha * heat, 0, 1)


@torch.enable_grad()
def plot_gradcam(model, imgs01, x, lam, device, classes=None, y_true=None,
                 target_layer=None, simpan=None):
    """Grid Grad-CAM: baris = sampel, kolom = [citra asli, overlay CAM].

    imgs01 [B,3,H,W] di [0,1] (untuk tampilan), x & lam dari siapkan_batch (untuk model).
    """
    classes = classes or CLASS_NAMES
    target_layer = target_layer or model.norm5
    gc = GradCAM(model, target_layer)
    try:
        cam, pred, logits = gc(x.to(device), None if lam is None else lam.to(device))
    finally:
        gc.remove()
    prob = logits.softmax(1).cpu().numpy()
    imgs01 = imgs01.cpu().numpy()

    n = imgs01.shape[0]
    fig, ax = plt.subplots(n, 2, figsize=(6, 3 * n))
    ax = np.atleast_2d(ax)
    for i in range(n):
        p = int(pred[i])
        judul = f"pred: {classes[p]} ({prob[i, p]:.2f})"
        if y_true is not None:
            tanda = "OK" if int(y_true[i]) == p else "SALAH"
            judul = f"asli: {classes[int(y_true[i])]} | {judul} [{tanda}]"
        ax[i, 0].imshow(np.transpose(imgs01[i], (1, 2, 0)))
        ax[i, 0].set_title("citra asli", fontsize=9)
        ax[i, 1].imshow(overlay_cam(imgs01[i], cam[i]))
        ax[i, 1].set_title(judul, fontsize=9)
        for a in ax[i]:
            a.axis("off")
    plt.tight_layout()
    if simpan:
        fig.savefig(simpan, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":                 # hindari warning di mode simpan/headless
        plt.show()
    plt.close(fig)
    return cam, pred
