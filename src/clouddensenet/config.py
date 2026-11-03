"""Konfigurasi global CloudDenseNet.

Nilai di sini adalah sumber kebenaran untuk arsitektur, augmentasi, dan jadwal
training. Diambil dari notebook `Copy_of_CloudDenseNet.ipynb` dan dilepaskan dari
Google Colab/Drive agar bisa dijalankan lokal.

Path default menunjuk ke folder `data/` dan `artifacts/` di root proyek.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Mode & reproducibility
# --------------------------------------------------------------------------- #
DEBUG = False           # True = uji cepat di CPU, False = training penuh (GPU)
DEBUG_BATCHES = 3       # jumlah batch per epoch saat DEBUG
SEED = 42

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
NUM_CLASSES = 7
IMG_SIZE = 224                       # 224 x 224 piksel, RGB
MEAN = [0.485, 0.456, 0.406]         # normalisasi ImageNet
STD = [0.229, 0.224, 0.225]
EPS = 1e-6
NUM_WORKERS = 0
NORM_MODE = "imagenet"               # 'imagenet' = ÷255 lalu mean/std | '01' = hanya ÷255

N_SPLIT = (8340, 1471, 9000)         # train/val/test (lihat DATASET_SPLIT.md)
IMBALANCE = "sampler"                # 'sampler' | 'weights' | 'none'
SAMPLES_PER_EPOCH = 7 * 3000         # hanya untuk 'sampler'

# Dataset GCD yang disediakan hanya punya train/ dan test/ (tanpa val/).
# Jika folder val/ tidak ada, val di-carve dari train secara stratified & seeded.
VAL_SPLIT = 0.15                     # proporsi train yang dijadikan validasi

# Nama 7 kelas (urutan sesuai folder ImageFolder)
CLASS_NAMES = [
    "1_cumulus", "2_altocumulus", "3_cirrus", "4_clearsky",
    "5_stratocumulus", "6_cumulonimbus", "7_mixed",
]

# --------------------------------------------------------------------------- #
# Arsitektur
# --------------------------------------------------------------------------- #
USE_SE = True
USE_SCCONV = True
USE_BRANCH = False
USE_BR = True
BRANCH_PAPER = False
ND_CH = 8
LAM_SCALE = 10.0

SE_REDUCTION = 16        # rasio reduksi r (Hu et al., 2018)
SE_INIT_BIAS = 3.0       # sigmoid(3) ≈ 0.953 -> SE awal mendekati identitas
P_DROP = 0.8             # dropout cabang Block 1
P_DROP_TOP = 0.5         # dropout Top Block
FUSION_ALPHA_INIT = 1.0
TOP_POOL = 7
TOP_DIMS = [TOP_POOL * TOP_POOL * 1024, 632, 512, 32]
BRANCH_DIMS = [256, 2048, 1024]

# --------------------------------------------------------------------------- #
# Optimasi (AdamW + Focal Loss)
# --------------------------------------------------------------------------- #
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0        # faktor fokus γ (Lin et al., 2017)
GRAD_CLIP = 5.0          # batas norma gradien

# Gradient Accumulation — akumulasi gradien beberapa mikro-batch sebelum satu
# langkah optimizer. Efektif saat VRAM terbatas: batch efektif = BATCH_SIZE x
# ACCUM_STEPS tanpa menambah memori puncak. Naikkan saat OOM, terutama di fase
# 'full' (defrost end-to-end) yang paling boros VRAM. 1 = nonaktif (perilaku lama).
ACCUM_STEPS = 1

# --------------------------------------------------------------------------- #
# Perbaikan generalisasi (v2)
# --------------------------------------------------------------------------- #
GROUPED_SPLIT = False
GROUP_THR = 0.95
STRONG_AUG = False
LABEL_SMOOTHING = 0.0
MIXUP_P = 0.0
MIXUP_ALPHA = 0.2
LECUN_UNIFORM = 3

# --------------------------------------------------------------------------- #
# Jadwal training (epoch_akhir, learning_rate, deskripsi)
# --------------------------------------------------------------------------- #

PHASES = [
    (10, 1e-3, "frozen",  "Top Block (bobot ImageNet beku)"),
    (20, 1e-4, "frozen",  "Tahap 2 (bobot ImageNet beku)"),
    (30, 1e-5, "partial", "Adaptasi Top Block"),
    (40, 5e-5, "partial", "Unfreeze Dense Block 3-4"),
    (60, 1e-5, "full",    "Partial Fine-Tuning Optimization"),
    (80, 1e-6, "full",    "Full Defrost (end-to-end)"),
]

BATCH_SIZE = 2 if DEBUG else 8
EPOCHS = PHASES[-1][0]
EPOCH_LIMIT = None           # None = jadwal penuh

LR = PHASES[0][1]
PATIENCE = 7
EARLY_STOP = False

# --------------------------------------------------------------------------- #
# Path (lokal, bukan Google Drive)
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
# Dataset hasil split: data/processed/{train,val,test} (lihat scripts/split_dataset.py).
# Raw asli tetap di data/raw/GCD/{train,test} sebagai cadangan.
DATA_ROOT = DATA_DIR / "processed"
CKPT_DIR = PROJECT_DIR / "artifacts" / "checkpoints"
RESULT_DIR = PROJECT_DIR / "artifacts" / "results"

VERSION = "_Versi01"
RUN_NAME = "cloud"

# --------------------------------------------------------------------------- #
# TensorBoard (pemantauan training — pasang & aktif otomatis)
# --------------------------------------------------------------------------- #
USE_TENSORBOARD = True            # tulis scalar (loss/acc/f1/lr) ke event file
TB_AUTO_INSTALL = True            # pip install tensorboard otomatis bila belum ada
TB_AUTO_LAUNCH = True             # jalankan server TensorBoard otomatis saat training
TB_LOG_DIR = PROJECT_DIR / "artifacts" / "runs"   # satu subfolder per run-name
TB_PORT = 6006                    # port server (http://localhost:6006)
