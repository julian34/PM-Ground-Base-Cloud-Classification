# Panduan Menjalankan CloudDenseNet di Google Colab

Panduan integrasi proyek **CloudDenseNet** ke Google Colab untuk memanfaatkan GPU
gratis (T4). Inti alurnya: **clone repo → install paket → sediakan dataset → simpan
hasil ke Drive → training**.

> **Catatan konteks**
> - Dataset `GCD.zip` **tidak ikut di-commit** (lihat `.gitignore`), jadi harus kamu
>   sediakan sendiri lewat Google Drive.
> - `config.py` memakai path relatif ke root repo (`PROJECT_DIR = parents[2]`), jadi
>   selama kamu `cd` ke folder repo semua path jalan tanpa diubah.
> - Device otomatis `cuda` kalau tersedia dan `DEBUG=False` — tidak perlu edit kode
>   (lihat `scripts/train.py`).

---

## 1. Aktifkan GPU

Menu Colab → **Runtime → Change runtime type → Hardware accelerator: GPU (T4)**.

## 2. Clone repo

```python
# Repo publik:
!git clone -b ViT-branch https://github.com/julian34/PM-Ground-Base-Cloud-Classification.git
%cd PM-Ground-Base-Cloud-Classification
```

Kalau repo **private**, pakai Personal Access Token (jangan commit token ini):

```python
# !git clone -b ViT-branch https://<USERNAME>:<TOKEN>@github.com/julian34/PM-Ground-Base-Cloud-Classification.git
```

## 3. Install paket

Colab sudah membawa PyTorch versi CUDA-nya sendiri. **Jangan reinstall
`torch==2.5.1+cu124`** (berat & bisa bentrok). Cukup pasang paket pakai torch bawaan
Colab:

```python
# Pasang clouddensenet + dependensi (torch bawaan Colab dipakai apa adanya)
!pip install -q -e . --no-deps
!pip install -q "numpy>=1.26,<2.1" pandas scikit-learn matplotlib seaborn Pillow tqdm PyYAML tensorboard
```

> ⚠️ **Penting:** pin `numpy<2.1` sering men-downgrade numpy bawaan Colab (biasanya
> 2.x). Kalau muncul peringatan *"restart runtime"*, jalankan **Runtime → Restart
> session**, lalu `%cd` lagi ke folder repo dan lanjut ke langkah berikutnya (tidak
> perlu install ulang).

Cek GPU terbaca:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 4. Sediakan dataset lewat Google Drive

Taruh `GCD.zip` di Drive-mu (mis. `MyDrive/CloudDenseNet/GCD.zip`), lalu:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
os.makedirs('data/raw', exist_ok=True)
shutil.copy('/content/drive/MyDrive/CloudDenseNet/GCD.zip', 'data/raw/GCD.zip')
```

Ekstrak + split (stratified, seed 42 → `data/processed/`):

```python
!python scripts/split_dataset.py --force
```

## 5. (Disarankan) Arahkan output ke Drive biar tidak hilang

Session Colab bersifat sementara. Supaya checkpoint & hasil aman, symlink folder
`artifacts/` ke Drive **sebelum** training:

```python
import os
os.makedirs('/content/drive/MyDrive/CloudDenseNet/artifacts', exist_ok=True)
!rm -rf artifacts
!ln -s /content/drive/MyDrive/CloudDenseNet/artifacts artifacts
```

## 6. Training

```python
!python scripts/train.py --data-root data/processed --epochs 80 --run-name cloud_colab
```

- Checkpoint terbaik → `artifacts/checkpoints/cloud_colab_Versi01.pth`
- Checkpoint terakhir (resume) → `..._last.pth` — resume otomatis kalau session putus
  lalu kamu jalankan lagi (`--no-resume` untuk matikan).
- Kurva & confusion matrix → `artifacts/results/`

## 7. (Opsional) TensorBoard di dalam Colab

```python
%load_ext tensorboard
%tensorboard --logdir artifacts/runs
```

---

## Ringkasan alur

| Langkah | Perintah inti |
| --- | --- |
| Clone | `git clone -b ViT-branch ... && cd PM-...` |
| Install | `pip install -e . --no-deps` + dependensi sisanya |
| Dataset | mount Drive → copy `GCD.zip` → `python scripts/split_dataset.py --force` |
| Output ke Drive | symlink `artifacts/` → `MyDrive/.../artifacts` |
| Training | `python scripts/train.py --run-name cloud_colab` |

Yang **wajib** kamu siapkan sendiri: file `GCD.zip` (via Drive) dan menyimpan hasil ke
Drive. Sisanya auto.
