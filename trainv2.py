import os
import math
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision import transforms

from net.model import PromptIR
import torch.cuda.amp as amp


# ============================================================
# 0. 參數設定
# ============================================================

SEED = 42

TRAIN_DEG_DIR = "/disk2/ccchen/DL_CV_class/HW/HW4/hw4_realse_dataset/train/degraded"
TRAIN_CLN_DIR = "/disk2/ccchen/DL_CV_class/HW/HW4/hw4_realse_dataset/train/clean"

RESUME_WEIGHTS = "/disk2/ccchen/DL_CV_class/HW/HW4/checkpoints/best_model.pth"

SAVE_DIR = "checkpoints_train_all"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 1
ACCUM_ITER = 16

CROP_SIZE = None

LEARNING_RATE = 5e-5

# ============================================================
# 只 train 10 epoch
# ============================================================

NUM_EPOCHS = 10

WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

WARMUP_EPOCHS = 1

USE_EMA = True
EMA_DECAY = 0.999

# ============================================================
# 每個 epoch 都存
# ============================================================

SAVE_EVERY = 1

NUM_WORKERS = 4


# ============================================================
# 1. 固定 Seed
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 2. Loss
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class MixedRestorationLoss(nn.Module):
    def __init__(self, mse_weight=0.1):
        super().__init__()
        self.char_loss = CharbonnierLoss()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        return self.char_loss(pred, target) + self.mse_weight * self.mse_loss(pred, target)


# ============================================================
# 3. Dataset
# ============================================================

class RestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, file_list, is_train=True, crop_size=128):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.file_list = file_list
        self.is_train = is_train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.file_list)

    def get_clean_name(self, deg_name):
        if deg_name.startswith("rain-"):
            return deg_name.replace("rain-", "rain_clean-")
        elif deg_name.startswith("snow-"):
            return deg_name.replace("snow-", "snow_clean-")
        else:
            raise ValueError(f"Unknown filename: {deg_name}")

    def __getitem__(self, idx):
        deg_name = self.file_list[idx]
        clean_name = self.get_clean_name(deg_name)

        deg_path = os.path.join(self.degraded_dir, deg_name)
        clean_path = os.path.join(self.clean_dir, clean_name)

        deg_img = Image.open(deg_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.is_train and self.crop_size is not None:

            i, j, h, w = transforms.RandomCrop.get_params(
                deg_img,
                output_size=(self.crop_size, self.crop_size),
            )

            deg_img = TF.crop(deg_img, i, j, h, w)
            clean_img = TF.crop(clean_img, i, j, h, w)

            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                clean_img = TF.hflip(clean_img)

            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                clean_img = TF.vflip(clean_img)

            if random.random() > 0.5:
                deg_img = TF.rotate(deg_img, 180)
                clean_img = TF.rotate(clean_img, 180)

        deg_tensor = TF.to_tensor(deg_img)
        clean_tensor = TF.to_tensor(clean_img)

        return deg_tensor, clean_tensor


# ============================================================
# 4. EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for key in ema_state.keys():
            model_value = model_state[key].detach()
            ema_value = ema_state[key]

            if ema_value.dtype.is_floating_point:
                ema_value.copy_(
                    ema_value * self.decay + model_value * (1.0 - self.decay)
                )
            else:
                ema_value.copy_(model_value)


# ============================================================
# 5. Scheduler
# ============================================================

def build_scheduler(optimizer):

    def lr_lambda(epoch):

        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)

        progress = float(epoch - WARMUP_EPOCHS) / float(
            max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        )

        progress = min(progress, 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        min_lr_ratio = MIN_LR / LEARNING_RATE

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 6. Load
# ============================================================

def load_model_weights(model, ckpt_path, device):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    print(f"[Info] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)


# ============================================================
# 7. Main
# ============================================================

def main():

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Device: {device}")

    # ========================================================
    # 使用全部資料
    # ========================================================

    train_files = sorted([
        f for f in os.listdir(TRAIN_DEG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"[Info] Total train images: {len(train_files)}")

    # ========================================================
    # Dataset / DataLoader
    # ========================================================

    train_dataset = RestorationDataset(
        TRAIN_DEG_DIR,
        TRAIN_CLN_DIR,
        train_files,
        is_train=True,
        crop_size=CROP_SIZE,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # ========================================================
    # Model
    # ========================================================

    model = PromptIR(
        inp_channels=3,
        out_channels=3,
        decoder=True,
    ).to(device)

    load_model_weights(model, RESUME_WEIGHTS, device)

    ema_model = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    # ========================================================
    # Loss / Optimizer
    # ========================================================

    criterion = MixedRestorationLoss(mse_weight=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer)

    scaler = amp.GradScaler()

    print("[Info] Start Training")

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for i, (deg_imgs, clean_imgs) in enumerate(train_loader):

            deg_imgs = deg_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)

            with amp.autocast():

                outputs = model(deg_imgs)

                loss = criterion(outputs, clean_imgs)

                loss = loss / ACCUM_ITER

            scaler.scale(loss).backward()

            do_step = (
                ((i + 1) % ACCUM_ITER == 0)
                or
                ((i + 1) == len(train_loader))
            )

            if do_step:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                if ema_model is not None:
                    ema_model.update(model)

            train_loss += loss.item() * ACCUM_ITER

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {avg_train_loss:.6f}"
        )

        # ====================================================
        # 每個 epoch 都存
        # ====================================================

        raw_path = os.path.join(
            SAVE_DIR,
            f"raw_epoch_{epoch+1}.pth"
        )

        torch.save(model.state_dict(), raw_path)

        if ema_model is not None:

            ema_path = os.path.join(
                SAVE_DIR,
                f"ema_epoch_{epoch+1}.pth"
            )

            torch.save(
                ema_model.ema.state_dict(),
                ema_path
            )

        print(f"  => Saved epoch {epoch+1}")

    print("[Info] Training Finished")


if __name__ == "__main__":
    main()