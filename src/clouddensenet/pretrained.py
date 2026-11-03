"""Transfer bobot DenseNet-121 ImageNet ke backbone CloudDenseNet.

- muat_imagenet    : salin bobot torchvision DenseNet121 (IMAGENET1K_V1).
- nolkan_input_nd  : nol-kan bobot pembaca channel ND agar awal = ImageNet murni.
"""
from __future__ import annotations

import torch
from torchvision import models

from .config import ND_CH


def muat_imagenet(model):
    """Salin bobot DenseNet121 ImageNet ke backbone CloudDenseNet."""
    tv = models.densenet121(weights="IMAGENET1K_V1").features.state_dict()
    own = model.state_dict()
    nama = {"norm1": "h.0", "conv1": "h.2", "norm2": "h.3", "conv2": "conv"}
    sd = {}
    for k, v in tv.items():
        k = (k.replace("conv0", "conv").replace("norm0", "bn")
              .replace("transition", "trans").replace("trans1.norm", "trans1.bn")
              .replace("trans2.norm", "trans2.bn").replace("trans3.norm", "trans3.bn"))
        if k.startswith("denseblock"):                       # denseblock1.denselayer1.norm1.weight
            b, l, m, w = k.split(".")
            k = f"block{b[-1]}.layers.{int(l[10:]) - 1}.{nama[m]}.{w}"
        if k in own and v.shape != own[k].shape:
            v = v[:own[k].shape[0]]                          # 32 -> 24 filter
        sd[k] = v
    hasil = model.load_state_dict(sd, strict=False)
    model.imagenet_keys = {k for k in sd if k in own}
    print("ImageNet dimuat:", len(sd) - len(hasil.unexpected_keys), "tensor")
    return model


@torch.no_grad()
def nolkan_input_nd(model, g=32, nd=ND_CH):
    """Bobot yang membaca channel ND diisi 0 -> awal training = ImageNet murni."""
    for b, c0 in [(1, 64), (2, 128), (3, 256), (4, 512)]:
        blk = getattr(model, f"block{b}")

        def posisi(n):
            return [c0 + j * g + (g - nd) + k for j in range(n) for k in range(nd)]

        for i, layer in enumerate(blk.layers):
            layer.h[2].weight[:, posisi(i)] = 0                     # conv1×1 layer berikutnya
        if hasattr(model, f"trans{b}"):
            getattr(model, f"trans{b}").conv.weight[:, posisi(len(blk.layers))] = 0
    return model
