"""Visualisasi hasil: kurva training dan confusion matrix."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

from .config import PHASES
from .data import siapkan_batch


def plot_training_history(history, simpan=None):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    if any(np.isnan(history[k]).any() for k in history):
        print("[!] Ada NaN di history — kurva bisa terputus/kosong.")
    panel = [("Loss", ["train_loss", "val_loss"]),
             ("Accuracy", ["train_acc", "val_acc"]),
             ("F1 Macro", ["val_f1"])]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, (judul, keys) in zip(ax, panel):
        for k in keys:
            a.plot(ep, history[k], "o-" if k.startswith("train") else "o--", label=k)
        for batas, *_ in PHASES[:-1]:                 # garis batas antar tahap
            if batas < len(ep):
                a.axvline(batas + 0.5, color="gray", ls=":", alpha=0.7)
        a.set(title=judul, xlabel="Epoch", ylabel=judul)
        a.grid(True, ls="--", alpha=0.6)
        a.legend()
    plt.tight_layout()
    if simpan:
        fig.savefig(simpan, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_confusion_matrix(model, loader, device, classes=None, simpan=None):
    model.eval()
    classes = classes or loader.dataset.classes
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="conf-mat", leave=False, dynamic_ncols=True):
        x, lam = siapkan_batch(x, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(x, lam).argmax(1)
        y_pred += pred.cpu().tolist()
        y_true += y.tolist()

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))  # selalu n×n
    cm_n = cm / cm.sum(1, keepdims=True).clip(min=1)                   # recall per baris
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    for a, data, fmt, judul in [(ax[0], cm, "d", "Jumlah"),
                                (ax[1], cm_n, ".2f", "Ternormalisasi (recall)")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=a,
                    xticklabels=classes, yticklabels=classes)
        a.set(title=judul, xlabel="Prediksi", ylabel="Label asli")
    plt.tight_layout()
    if simpan:
        fig.savefig(simpan, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Akurasi {np.trace(cm) / cm.sum():.4f} | "
          f"F1 macro {f1_score(y_true, y_pred, average='macro'):.4f}")
