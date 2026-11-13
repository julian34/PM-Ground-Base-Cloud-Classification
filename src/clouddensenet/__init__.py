"""CloudDenseNet — DenseNet-121 termodifikasi untuk klasifikasi 7 kelas awan.

Modifikasi utama vs DenseNet-121 standar:
  * SEBlock (Squeeze-and-Excitation) pada bottleneck tiap DenseLayer & tiap block.
  * SCConv — konvolusi cosine ter-normalisasi dengan skala τ per-channel.
  * SpatialAttnBR — atensi spasial dengan channel λ rasio Biru/Merah (sadar-fisika).
  * Top Block: GAP -> [FC -> BN -> ReLU] x N -> Dropout -> FC(->7) (lebar: TOP_HIDDEN).
  * Transfer learning ImageNet 5 tahap (freeze bertahap) + Focal Loss.
"""
from __future__ import annotations

from . import config
from .attention import EdgeLayer, SEBlock, SpatialAttn, SpatialAttnBR
from .blocks import (
    DenseBlock1, DenseBlock2, DenseBlock3, DenseBlock4,
    DenseLayer, TopBlock, TransitionLayer1, TransitionLayer2, TransitionLayer3,
)
from .checkpoint import ambil_state_dict, bangun_dari_checkpoint, dimensi_top
from .conv import SCConv
from .data import build_loaders, build_test_loader, build_transforms, set_seed, siapkan_batch
from .engine import apply_phase, evaluate, phase_of, train_one_epoch
from .epochlog import EpochLogger
from .gradcam import GradCAM, overlay_cam, plot_gradcam
from .finetune import bekukan_imagenet, buka_blok_3_4, buka_semua, set_bn_beku_eval
from .losses import FocalLoss, init_new_layers, lecun_uniform_
from .model import CloudDenseNet
from .pretrained import muat_imagenet, nolkan_input_nd
from .spectral import br_ratio_lambda, lambda_relatif, valid_mask
from .tblog import catat_epoch, hentikan_server, siapkan_tensorboard, tutup_tensorboard

__all__ = [
    "config",
    "CloudDenseNet",
    "ambil_state_dict", "dimensi_top", "bangun_dari_checkpoint",
    "DenseLayer", "DenseBlock1", "DenseBlock2", "DenseBlock3", "DenseBlock4",
    "TransitionLayer1", "TransitionLayer2", "TransitionLayer3", "TopBlock",
    "SEBlock", "SpatialAttn", "SpatialAttnBR", "EdgeLayer", "SCConv",
    "FocalLoss", "init_new_layers", "lecun_uniform_",
    "muat_imagenet", "nolkan_input_nd",
    "bekukan_imagenet", "buka_blok_3_4", "buka_semua", "set_bn_beku_eval",
    "build_loaders", "build_test_loader", "build_transforms", "set_seed", "siapkan_batch",
    "train_one_epoch", "evaluate", "phase_of", "apply_phase",
    "EpochLogger",
    "GradCAM", "overlay_cam", "plot_gradcam",
    "br_ratio_lambda", "lambda_relatif", "valid_mask",
    "siapkan_tensorboard", "catat_epoch", "tutup_tensorboard", "hentikan_server",
]

__version__ = "0.1.0"
