"""Focal Loss (Lin et al., 2017) dan inisialisasi LeCun Uniform untuk lapisan baru."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        ce_loss = self.ce_loss(logits, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def lecun_uniform_(tensor):
    fan_in = tensor[0].numel()
    limit = math.sqrt(3.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def init_new_layers(m):
    """Hanya untuk lapisan BARU (SE, cabang B1, Top Block). Jangan .apply ke seluruh model."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        lecun_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
