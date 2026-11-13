"""Blok penyusun CloudDenseNet: DenseLayer, DenseBlock1-4, Transition1-3, TopBlock."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SEBlock
from .config import NUM_CLASSES, P_DROP_TOP, TOP_HIDDEN, USE_SCCONV
from .conv import SCConv


class DenseLayer(nn.Module):
    """BN-ReLU-Conv1x1(4k)-BN-ReLU -> SE -> SCConv3x3(k). Opsional: ND (nd channel)."""

    def __init__(self, in_channels, growth_rate=32, nd=0, use_se=False, eps=1e-3):
        super().__init__()
        mid = 4 * growth_rate  # 128
        self.h = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.se = SEBlock(mid) if use_se else nn.Identity()           # SE pada bottleneck (128)
        self.conv = (SCConv(mid, growth_rate - nd, 3, padding=1) if USE_SCCONV
                     else nn.Conv2d(mid, growth_rate - nd, 3, padding=1, bias=False))
        self.pq = nn.Conv2d(mid, 2 * nd, 3, padding=1, bias=False) if nd else None
        self.eps = eps

    def forward(self, x):
        x = torch.cat(x, 1) if isinstance(x, (list, tuple)) else x
        h = self.se(self.h(x))                                        # H_l dulu, lalu SE
        out = self.conv(h)                                            # SCConv
        if self.pq is not None:
            P, Q = F.softplus(self.pq(h)).chunk(2, 1)
            out = torch.cat([out, (P - Q) / (P + Q + self.eps)], 1)
        return out


class DenseBlock1(nn.Module):
    """Wadah untuk menyusun N DenseLayer secara berurutan (concat dense)."""

    def __init__(self, num_layers=6, in_channels=64, growth_rate=32, nd=0, use_se=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate, nd=nd, use_se=use_se)
            for i in range(num_layers)])

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(feats))
        return torch.cat(feats, 1)


class DenseBlock2(DenseBlock1):
    """12 dense layer: 128 -> 128 + 12·32 = 512 channel, spasial tetap 64×64."""

    def __init__(self, num_layers=12, in_channels=128, growth_rate=32, nd=0, use_se=False):
        super().__init__(num_layers, in_channels, growth_rate, nd, use_se)


class DenseBlock3(DenseBlock1):
    """24 dense layer: 256 -> 256 + 24·32 = 1024 channel, spasial tetap 32×32."""

    def __init__(self, num_layers=24, in_channels=256, growth_rate=32, nd=0, use_se=False):
        super().__init__(num_layers, in_channels, growth_rate, nd, use_se)


class DenseBlock4(DenseBlock1):
    """16 dense layer: 512 -> 512 + 16·32 = 1024 channel, spasial tetap 16×16."""

    def __init__(self, num_layers=16, in_channels=512, growth_rate=32, nd=0, use_se=False):
        super().__init__(num_layers, in_channels, growth_rate, nd, use_se)


class TransitionLayer1(nn.Module):
    """Mengecilkan channel 50% dan mereduksi ukuran spasial menjadi setengahnya."""

    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class TransitionLayer2(nn.Module):
    """BN-ReLU-Conv1x1 (512 -> 256) -> AvgPool 2×2 (64×64 -> 32×32)."""

    def __init__(self, in_channels=512, out_channels=256):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class TransitionLayer3(TransitionLayer2):
    """BN-ReLU-Conv1x1 (1024 -> 512) -> AvgPool 2×2 (32×32 -> 16×16)."""

    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__(in_channels, out_channels)


class TopBlock(nn.Module):
    """GAP -> [FC -> BN -> ReLU] x len(hidden) -> Dropout -> FC(hidden[-1] -> n_cls).

    `hidden` = lebar lapisan tersembunyi (default `config.TOP_HIDDEN`). Urutan modul
    di `bottleneck` (Linear, BN, ReLU per lapisan, lalu Dropout) menentukan nama kunci
    state_dict (`top.bottleneck.0/3/6/...`) — jangan diubah agar checkpoint lama
    tetap bisa dimuat.
    """

    def __init__(self, in_channels=1024, n_cls=NUM_CLASSES, p=P_DROP_TOP, hidden=None,
                 extra_dim=0):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # extra_dim = lebar vektor cabang tambahan (mis. ViT) yang di-concat ke
        # vektor DenseNet sebelum MLP. extra_dim=0 -> perilaku lama persis.
        self.extra_dim = extra_dim

        hidden = tuple(TOP_HIDDEN if hidden is None else hidden)
        self.hidden = hidden
        lapisan, d_in = [], in_channels + extra_dim      # +extra_dim utk fusi concat
        for d_out in hidden:
            lapisan += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
            d_in = d_out
        lapisan.append(nn.Dropout(p))
        self.bottleneck = nn.Sequential(*lapisan)
        self.fc = nn.Linear(d_in, n_cls)

    def forward(self, x, extra=None):
        x = self.global_pool(x)        # [B, 1024, 1, 1]
        x = self.flatten(x)            # [B, 1024]
        if extra is not None:          # fusi: gabung vektor DenseNet + vektor cabang
            x = torch.cat([x, extra], dim=1)   # [B, 1024 + extra_dim]
        x = self.bottleneck(x)         # [B, hidden[-1]]
        return self.fc(x)              # [B, n_cls] (logits untuk CrossEntropy/Focal)
