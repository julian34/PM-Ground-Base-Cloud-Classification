"""SCConv — konvolusi ter-normalisasi (cosine) dengan skala per-channel.

Menghitung konvolusi lalu menormalkannya terhadap norma patch input dan norma
kernel (cosine similarity), diskalakan per-channel dengan τ. Dipaksa fp32 agar
stabil di bawah AMP.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 q_init=1e-2, tau_init=1.0, eps=1e-6):
        super().__init__()
        self.stride, self.padding, self.eps = stride, padding, eps
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("ones", torch.ones(1, in_ch, kernel_size, kernel_size))
        self.log_q = nn.Parameter(torch.tensor(math.log(q_init)))                # q > 0
        self.log_tau = nn.Parameter(torch.full((out_ch,), math.log(tau_init)))   # τ > 0 per channel

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):   # paksa fp32
            x = F.pad(x.float(), (self.padding,) * 4, mode="reflect")
            w = self.weight.float()
            k = w.shape[-1]
            dot = F.conv2d(x, w, stride=self.stride)
            # ||x|| per patch = jumlah x² lintas channel, lalu jumlah jendela k×k.
            sq = x.pow(2).sum(1, keepdim=True)
            x_norm = (F.avg_pool2d(sq, k, stride=self.stride, divisor_override=1)
                      .clamp_min(0) + self.eps).sqrt()
            w_norm = w.flatten(1).norm(dim=1).view(1, -1, 1, 1)
            q = self.log_q.exp()
            tau = self.log_tau.exp().view(1, -1, 1, 1)
            cos = (dot / ((x_norm + q) * (w_norm + self.eps))).clamp(-1.0, 1.0)
            return torch.sign(dot) * (cos.abs() + self.eps).pow(tau)
