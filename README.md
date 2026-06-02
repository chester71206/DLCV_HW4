# PromptIR Rain and Snow Image Restoration

## Introduction

This project implements an all-in-one image restoration system for removing rain and snow degradation from images. It was developed for **Visual Recognition using Deep Learning, Homework 4 (Spring 2026)**.

The task requires a **single model** that can restore both rain-degraded and snow-degraded images. The required architecture is [PromptIR](https://github.com/va1shn9v/PromptIR), an all-in-one blind image restoration model that uses degradation-aware prompts to dynamically guide the restoration network.

The model is trained from scratch:

- No external dataset is used.
- No pretrained weights are used.
- A single PromptIR model is used for both rain and snow restoration.
- The final public leaderboard PSNR is **29.92 dB**.

The training pipeline contains two stages:

1. **Validation-based training:** use a 90% / 10% training-validation split to select a stable checkpoint.
2. **Full-data fine-tuning:** resume from the best checkpoint and fine-tune on all 3,200 training image pairs for 10 additional epochs.

---

## Method Overview

### PromptIR

PromptIR is designed for all-in-one blind image restoration. Instead of training a separate model for each degradation type, PromptIR uses learnable prompts to encode degradation-related information. These prompts dynamically guide the restoration network according to the input image.

In this project, the same model processes:

- Rain-degraded images
- Snow-degraded images

### Stage 1: Validation-Based Training

The first training stage randomly divides the 3,200 image pairs into:

- 90% training data
- 10% validation data

The model is trained with Charbonnier Loss:

```text
Charbonnier Loss = average of sqrt((prediction - target)^2 + epsilon^2)
epsilon = 0.001
```

The checkpoint with the lowest validation loss is stored as:

```text
checkpoints/best_model.pth
```

### Stage 2: Full-Data Fine-Tuning

The second stage resumes from `checkpoints/best_model.pth` and fine-tunes the model using all 3,200 training image pairs.

The second stage uses a mixed restoration loss:

```text
Mixed Loss = Charbonnier Loss + 0.1 * Mean Squared Error
```

The following techniques are also applied:

- Gradient accumulation
- Gradient clipping
- Mixed-precision training
- Exponential Moving Average (EMA) weights
- Warm-up followed by cosine learning-rate decay
- Fixed random seed for reproducibility

The final submission uses:

```text
checkpoints_train_all/ema_epoch_10.pth
```

---

## Project Structure

```text
.
├── train.py                 # Stage 1: validation-based PromptIR training
├── train_all.py             # Stage 2: full-data fine-tuning
├── inference.py             # Generate pred.npz for submission
├── net/
│   └── model.py             # PromptIR model definition
├── requirements.txt         # Python package dependencies
├── README.md
└── .gitignore
```

The dataset and model checkpoints should not be pushed to GitHub.

---

## Dataset Structure

Place the dataset in the following structure:

```text
hw4_realse_dataset/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png
│   │   ├── ...
│   │   ├── rain-1600.png
│   │   ├── snow-1.png
│   │   ├── ...
│   │   └── snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png
│       ├── ...
│       ├── rain_clean-1600.png
│       ├── snow_clean-1.png
│       ├── ...
│       └── snow_clean-1600.png
└── test/
    └── degraded/
        ├── 0.png
        ├── ...
        └── 99.png
```

The training set contains:

| Degradation Type | Number of Degraded Images | Number of Clean Images |
|---|---:|---:|
| Rain | 1,600 | 1,600 |
| Snow | 1,600 | 1,600 |
| Total | 3,200 | 3,200 |

---

## Environment Setup

### Recommended Environment

- Python 3.9 or higher
- PyTorch
- CUDA-capable GPU recommended
- Conda or another virtual environment tool

### Create a Conda Environment

```bash
conda create -n cv_hw4 python=3.9 -y
conda activate cv_hw4
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

A minimal set of required packages is:

```text
numpy
Pillow
torch
torchvision
```

To export the packages from an existing Conda environment:

```bash
conda activate cv_hw4
pip list --format=freeze > requirements.txt
```

Using `pip list --format=freeze` avoids local build paths such as:

```text
package-name @ file:///local/build/path
```

---

## Configuration

Before running the code, update the dataset and checkpoint paths in the Python files.

### Stage 1 Paths

In `train.py`, configure:

```python
train_degraded_dir = "/path/to/hw4_realse_dataset/train/degraded"
train_clean_dir = "/path/to/hw4_realse_dataset/train/clean"
save_dir = "checkpoints"
```

### Stage 2 Paths

In `train_all.py`, configure:

```python
TRAIN_DEG_DIR = "/path/to/hw4_realse_dataset/train/degraded"
TRAIN_CLN_DIR = "/path/to/hw4_realse_dataset/train/clean"
RESUME_WEIGHTS = "/path/to/checkpoints/best_model.pth"
SAVE_DIR = "checkpoints_train_all"
```

### Inference Paths

In `inference.py`, configure:

```python
test_dir = "/path/to/hw4_realse_dataset/test/degraded"
ckpt_path = "/path/to/checkpoints_train_all/ema_epoch_10.pth"
output_file = "pred.npz"
```

---

## Usage

### Stage 1: Validation-Based Training

Run:

```bash
python train.py
```

Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Maximum epochs | 100 |
| Training-validation split | 90% / 10% |
| Batch size | 1 |
| Gradient accumulation steps | 16 |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Scheduler | Cosine annealing |
| Minimum learning rate | 0.000001 |
| Loss function | Charbonnier Loss |
| Early stopping patience | 15 |
| Mixed-precision training | Enabled |

The best checkpoint is stored as:

```text
checkpoints/best_model.pth
```

### Stage 2: Full-Data Fine-Tuning

Run:

```bash
python train_all.py
```

Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Initial checkpoint | `checkpoints/best_model.pth` |
| Fine-tuning epochs | 10 |
| Number of training images | 3,200 |
| Batch size | 1 |
| Gradient accumulation steps | 16 |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.00005 |
| Weight decay | 0.0001 |
| Scheduler | One warm-up epoch followed by cosine decay |
| Minimum learning rate | 0.000001 |
| Loss function | Charbonnier Loss + 0.1 * Mean Squared Error |
| Maximum gradient norm | 1.0 |
| EMA decay | 0.999 |
| Mixed-precision training | Enabled |
| Random seed | 42 |

Both raw weights and EMA weights are stored after every epoch:

```text
checkpoints_train_all/raw_epoch_1.pth
checkpoints_train_all/ema_epoch_1.pth
...
checkpoints_train_all/raw_epoch_10.pth
checkpoints_train_all/ema_epoch_10.pth
```

### Inference

Run:

```bash
python inference.py
```

The inference script:

1. Loads `ema_epoch_10.pth`.
2. Restores each test image with a single forward pass.
3. Clamps output values into the range from 0 to 1.
4. Converts each restored image into an unsigned 8-bit NumPy array.
5. Stores all images in `pred.npz`.

No test-time augmentation is used.

### Prepare the CodaBench Submission

The required output filename inside the ZIP archive is:

```text
pred.npz
```

Compress the prediction file:

```bash
zip submission.zip pred.npz
```

Upload `submission.zip` to CodaBench.

---

## Output Format

The output file is a dictionary-like NumPy archive:

```python
np.savez("pred.npz", **images_dict)
```

Each key is an original test filename:

```text
0.png
1.png
...
99.png
```

Each value is a restored image array with the following format:

```text
Shape: (3, H, W)
Data type: uint8
Value range: 0 to 255
```

---

## Performance Snapshot

<img width="1446" height="277" alt="image" src="https://github.com/user-attachments/assets/012b626f-e808-453f-9bef-2df26cdb2bf2" />


The final public leaderboard score is:

```text
29.92 dB
```

The second-stage refinement pipeline improves the public score by:

```text
29.92 - 29.89 = 0.03 dB
```

The second-stage improvement should be interpreted as the combined effect of:

- Full-data fine-tuning
- Mixed restoration loss
- Gradient clipping
- EMA weights
- Smaller learning rate

Because these components were introduced together, the gain cannot be attributed exclusively to a single modification without an isolated ablation experiment.

---

## Code Reliability

The implementation follows these reliability practices:

- Fixed random seed during Stage 2
- Gradient accumulation to reduce GPU memory usage
- Mixed-precision training to reduce memory consumption
- Gradient clipping to prevent unstable updates
- EMA checkpoints for stable inference
- Separate training, fine-tuning, and inference scripts
- Model checkpoints excluded from GitHub
- Dataset files excluded from GitHub

A recommended `.gitignore` file is:

```text
__pycache__/
*.pyc
*.pth
*.pt
*.npz
*.zip
checkpoints/
checkpoints_train_all/
hw4_realse_dataset/
.DS_Store
```

---

## Python Coding Style Guide Reference

The Python code should be formatted and reviewed according to the following coding style references.

### PEP8

PEP8 is the official Python style guide:

- https://peps.python.org/pep-0008/

Recommended linting command:

```bash
pip install flake8
flake8 .
```

### Google Python Style Guide

Google Python Style Guide:

- https://google.github.io/styleguide/pyguide.html

Optional linting command:

```bash
pip install pylint
pylint train.py train_all.py inference.py
```

---

## References

1. V. Potlapalli, S. W. Zamir, S. Khan, and F. S. Khan, **"PromptIR: Prompting for All-in-One Blind Image Restoration,"** Advances in Neural Information Processing Systems, 2023.  
   Paper: https://arxiv.org/abs/2306.13090

2. Official PromptIR PyTorch implementation:  
   https://github.com/va1shn9v/PromptIR

3. PEP8 Style Guide for Python Code:  
   https://peps.python.org/pep-0008/

4. Google Python Style Guide:  
   https://google.github.io/styleguide/pyguide.html
