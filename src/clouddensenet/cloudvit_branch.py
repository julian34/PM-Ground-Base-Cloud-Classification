"""Cabang CloudViT sebagai pengganti cabang ViT sederhana.

Reimplementasi backbone **CloudViT** (Wei dkk., *Computers, Materials & Continua*
2025, DOI 10.32604/cmc.2025.061402) — model hybrid CNN+Transformer ringan untuk
klasifikasi citra awan. Di repo ini CloudViT dipakai sebagai **Cabang B** yang
berjalan paralel dengan backbone DenseNet; keluarannya (vektor 256-dim dari Global
Average Pooling, yaitu C_Block TANPA Fc) di-concat dengan vektor DenseNet di dalam
`TopBlock`. Jadi C_Block/Fc asli paper DIGANTI oleh TopBlock kita (7 kelas GCD).

Struktur mengikuti Tabel 1 paper (input 224x224x3):

    I_Block            112x112   16   (Conv s2 + BN + SiLU)
    L_Block_1  D+E      56x56    32
    L_Block_2  D+E      28x28    64
    G_Block_1  D+G      14x14   128
    G_Block_2  D+G       7x7    192   (paper menulis 8x8; conv s2 memberi 7 — tak
    G_Block_3  D+G       4x4    256    masalah karena diakhiri GAP)
    C_Block    GAP       1x1    256  -> (ke TopBlock, bukan Fc-11)

Definisi modul mengikuti Gbr 3-5 & Pers. (1)-(8):
  * ECA         — lightweight channel attention (Gbr 3): GAP -> Conv1D -> sigmoid.
  * D_Module    — downsampling dua-cabang + channel shuffle (Gbr 4a, Pers. 1-3).
  * E_Module    — channel split + residual (Gbr 4b, Pers. 4-5).
  * G_Module    — DWSep local rep + separable self-attention + fusi (Gbr 5, Pers. 6-8).

Catatan kejujuran (bagian yang TIDAK diangkakan eksplisit oleh paper, dipilih wajar):
  - Kernel I_Block diasumsikan 3x3 (lazim; paper hanya menyebut "conv s2").
  - Pers. (1)/(4) menulis stride yang tampak tak konsisten untuk concat; di sini
    D_Module memakai stride 2 di KEDUA cabang (agar shape cocok) dan E_Module
    stride 1 (sesuai Gbr 4b yang tak men-downsample).
  - Kedalaman transformer per G_Module (`gdepth`) & dropout tidak disebut paper;
    default 2 & 0.1. Tokenisasi G_Module memakai patch 1 (tiap piksel = 1 token)
    alih-alih unfold/fold MobileViT — ekuivalen-atau-lebih-global pada resolusi
    kecil (14/7/4) dan lebih ringkas; matematika separable self-attention tetap
    persis Pers. (6)-(8).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Aduk channel antar-grup (ShuffleNet) agar info dua cabang bertukar."""
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ECA(nn.Module):
    """Efficient Channel Attention (Gbr 3): GAP -> Conv1D -> sigmoid -> skala."""

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                         # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)        # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)   # (B, C, 1, 1)
        return x * y


class DWSep(nn.Module):
    """Depthwise separable conv ('DWConv' pada legenda paper): DW k x k + PW 1x1."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride, k // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class IBlock(nn.Module):
    """Initial Feature Extraction Block: Conv s2 -> BN -> SiLU (3 -> 16, 224->112)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DModule(nn.Module):
    """Downsampling Module (Gbr 4a, Pers. 1-3): dua cabang -> concat -> shuffle.

    Keduanya stride 2 agar keluaran separuh resolusi & shape cocok untuk concat.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 2 == 0, "out_ch D_Module harus genap (dibagi dua cabang)"
        half = out_ch // 2

        # Cabang 1 (G1): DWSep s2 -> BN -> SiLU
        self.b1 = nn.Sequential(
            DWSep(in_ch, half, stride=2), nn.BatchNorm2d(half), nn.SiLU(inplace=True))

        # Cabang 2 (G2): PW -> DWSep s2 -> BN -> SiLU -> ECA
        self.pw = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.b2 = nn.Sequential(
            DWSep(in_ch, half, stride=2), nn.BatchNorm2d(half), nn.SiLU(inplace=True))
        self.eca = ECA(half)

    def forward(self, x):
        g1 = self.b1(x)
        g2 = self.eca(self.b2(self.pw(x)))
        return channel_shuffle(torch.cat([g1, g2], dim=1), groups=2)


class EModule(nn.Module):
    """Feature Extraction Module (Gbr 4b, Pers. 4-5): channel split + residual.

    Stride 1 (tak men-downsample). Separuh channel lewat jalur konvolusi, separuh
    lagi jadi skip-connection, lalu di-concat kembali.
    """

    def __init__(self, ch: int):
        super().__init__()
        assert ch % 2 == 0, "channel E_Module harus genap (channel split)"
        half = ch // 2
        self.pw = nn.Conv2d(half, half, 1, bias=False)
        self.body = nn.Sequential(
            DWSep(half, half, stride=1), nn.BatchNorm2d(half), nn.SiLU(inplace=True))
        self.eca = ECA(half)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)               # skip half, jalur proses half
        g2 = self.eca(self.body(self.pw(x2)))
        return torch.cat([x1, g2], dim=1)


class SeparableSelfAttention(nn.Module):
    """Separable self-attention MobileViTv2 (Pers. 6-8) — kompleksitas linear O(N).

    C_s = softmax(x W_I) ; context = sum_N(C_s * x W_K) ; y = (ReLU(x W_V) * context) W_O.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.to_i = nn.Linear(dim, 1)            # W_I -> skor konteks per token
        self.to_k = nn.Linear(dim, dim)          # W_K
        self.to_v = nn.Linear(dim, dim)          # W_V
        self.to_out = nn.Linear(dim, dim)        # W_O

    def forward(self, x):                        # x: (B, N, d)
        cs = torch.softmax(self.to_i(x), dim=1)  # (B, N, 1) context scores
        k = self.to_k(x)                         # (B, N, d)
        context = (cs * k).sum(dim=1, keepdim=True)   # (B, 1, d) context vector
        v = torch.relu(self.to_v(x))             # (B, N, d)
        return self.to_out(v * context)          # broadcast -> (B, N, d)


class SepTransformerBlock(nn.Module):
    """Satu blok transformer: SeparableSelfAttention + MLP(SiLU), pre-LN + residual."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SeparableSelfAttention(dim)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class GModule(nn.Module):
    """Global Feature Extraction Module (Gbr 5, Pers. 6-8).

    DWSep (local rep) -> tokenisasi -> transformer separable x gdepth -> fusi PW.
    Stride 1 (D_Module di blok yang sama sudah men-downsample).
    """

    def __init__(self, in_ch: int, out_ch: int, gdepth: int = 2,
                 mlp_ratio: float = 2.0, drop: float = 0.1):
        super().__init__()
        self.local = DWSep(in_ch, out_ch, stride=1)          # DWConv pada Gbr 5
        self.blocks = nn.ModuleList(
            [SepTransformerBlock(out_ch, mlp_ratio, drop) for _ in range(gdepth)])
        self.norm = nn.LayerNorm(out_ch)
        self.proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)  # Conv fusi pada Gbr 5

    def forward(self, x):
        x = self.local(x)                        # (B, C, H, W)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)         # (B, N=H*W, C)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(x)


class CloudViTBranch(nn.Module):
    """Backbone CloudViT sebagai cabang paralel; keluaran vektor `out_dim` (256).

    Membaca CITRA MENTAH (B, 3, IMG_SIZE, IMG_SIZE) secara mandiri (jalur independen
    dari DenseNet) dan menghasilkan satu vektor ringkasan untuk difusi di TopBlock.
    """

    def __init__(self, in_ch: int = 3, channels=(16, 32, 64, 128, 192, 256),
                 gdepth: int = 2, drop: float = 0.1):
        super().__init__()
        c0, c1, c2, c3, c4, c5 = channels
        self.out_dim = c5

        self.i_block = IBlock(in_ch, c0)                     # -> 112, c0

        # L_Block = D_Module (downsample + ganti channel) + E_Module (preserve)
        self.l1 = nn.Sequential(DModule(c0, c1), EModule(c1))   # -> 56,  c1
        self.l2 = nn.Sequential(DModule(c1, c2), EModule(c2))   # -> 28,  c2

        # G_Block = D_Module + G_Module
        self.g1 = nn.Sequential(DModule(c2, c3), GModule(c3, c3, gdepth, drop=drop))  # 14
        self.g2 = nn.Sequential(DModule(c3, c4), GModule(c4, c4, gdepth, drop=drop))  # 7
        self.g3 = nn.Sequential(DModule(c4, c5), GModule(c5, c5, gdepth, drop=drop))  # 4

        self.gap = nn.AdaptiveAvgPool2d(1)                   # C_Block (tanpa Fc)

    def forward(self, x):
        x = self.i_block(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.gap(x).flatten(1)                           # (B, out_dim)
        return x
