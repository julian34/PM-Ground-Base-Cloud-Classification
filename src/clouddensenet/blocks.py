"""Blok penyusun CloudDenseNet: DenseLayer, DenseBlock1-4, Transition1-3, TopBlock."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SEBlock
from .config import NUM_CLASSES, P_DROP_TOP, USE_SCCONV
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
    """GAP -> FC(1024->32) -> BN -> ReLU -> Dropout -> FC(32->n_cls)."""

    def __init__(self, in_channels=1024, n_cls=NUM_CLASSES, p=P_DROP_TOP):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        mid_features = 32
        self.bottleneck = nn.Sequential(
            nn.Linear(in_channels, mid_features),
            nn.BatchNorm1d(mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )
        self.fc = nn.Linear(mid_features, n_cls)

    def forward(self, x):
        x = self.global_pool(x)   # [B, 1024, 1, 1]
        x = self.flatten(x)       # [B, 1024]
        x = self.bottleneck(x)    # [B, 32]
        return self.fc(x)         # [B, n_cls] (logits untuk CrossEntropy/Focal)
