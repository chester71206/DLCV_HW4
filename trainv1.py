import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from net.model import PromptIR
import torch.cuda.amp as amp

# ==========================================
# 1. Dataset 與 Loss 定義
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps**2))
        return loss

class RestorationDataset(Dataset):
    # 修改 init，允許直接傳入 file_list
    def __init__(self, degraded_dir, clean_dir, file_list, is_train=True):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.is_train = is_train
        self.image_files = file_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        deg_name = self.image_files[idx]
        
        if deg_name.startswith('rain'):
            clean_name = deg_name.replace('rain-', 'rain_clean-')
        elif deg_name.startswith('snow'):
            clean_name = deg_name.replace('snow-', 'snow_clean-')
        else:
            clean_name = deg_name

        deg_path = os.path.join(self.degraded_dir, deg_name)
        clean_path = os.path.join(self.clean_dir, clean_name)

        deg_img = Image.open(deg_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        if self.is_train:
            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                clean_img = TF.hflip(clean_img)
                
            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                clean_img = TF.vflip(clean_img)
                
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                deg_img = TF.rotate(deg_img, angle)
                clean_img = TF.rotate(clean_img, angle)

        deg_img = TF.to_tensor(deg_img)
        clean_img = TF.to_tensor(clean_img)

        return deg_img, clean_img

# ==========================================
# 2. 主程式
# ==========================================
def main():
    # --- 參數設定 ---
    batch_size = 1
    accum_iter = 16             # 梯度累積步數：模擬 1 x 16 = 16 的大 Batch Size
    num_epochs = 100
    learning_rate = 2e-4
    patience = 15
    save_dir = 'checkpoints'
    
    os.makedirs(save_dir, exist_ok=True)

    train_degraded_dir = "/disk2/ccchen/DL_CV_class/HW/HW4/hw4_realse_dataset/train/degraded"
    train_clean_dir = "/disk2/ccchen/DL_CV_class/HW/HW4/hw4_realse_dataset/train/clean"
    
    # --- 正確的 Train/Val Split 寫法 ---
    all_files = [f for f in os.listdir(train_degraded_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(all_files) # 打亂順序
    
    val_size = int(len(all_files) * 0.1)
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]
    
    # 分別實例化，互不干擾
    train_dataset = RestorationDataset(train_degraded_dir, train_clean_dir, file_list=train_files, is_train=True)
    val_dataset = RestorationDataset(train_degraded_dir, train_clean_dir, file_list=val_files, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"資料準備完畢: 訓練集 {len(train_dataset)} 張 | 驗證集 {len(val_dataset)} 張")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(inp_channels=3, out_channels=3, decoder=True).to(device) 
    
    # 修改點：換上 CharbonnierLoss
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = amp.GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    print("開始訓練...")
    for epoch in range(num_epochs):
        # ------------------- 訓練階段 -------------------
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad() # 確保在迴圈開始前梯度清零
        
        for i, (deg_imgs, clean_imgs) in enumerate(train_loader):
            deg_imgs, clean_imgs = deg_imgs.to(device), clean_imgs.to(device)
            
            with amp.autocast():
                outputs = model(deg_imgs)
                loss = criterion(outputs, clean_imgs)
                # 修改點：除以累積步數，標準化 Loss
                loss = loss / accum_iter 
            
            scaler.scale(loss).backward()
            
            # 修改點：累積滿 accum_iter 步，或是達到 epoch 結尾時，才進行權重更新
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 更新完後清空梯度
            
            # 還原顯示用的 Loss 值 (乘以 accum_iter)
            train_loss += (loss.item() * accum_iter)

        avg_train_loss = train_loss / len(train_loader)

        # ------------------- 驗證階段 -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for deg_imgs, clean_imgs in val_loader:
                deg_imgs, clean_imgs = deg_imgs.to(device), clean_imgs.to(device)
                
                # 修改點：驗證階段也要開 AMP 防止 OOM
                with amp.autocast():
                    outputs = model(deg_imgs)
                    loss = criterion(outputs, clean_imgs)
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ------------------- 儲存機制與 Early Stop -------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("  => 發現更好的模型，已儲存 best_model.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            print(f"  => 已儲存 epoch {epoch+1} 的模型")

        if patience_counter >= patience:
            print(f"\n[!] 連續 {patience} 個 Epoch 驗證誤差沒有下降，Early Stopping 觸發！")
            break

        scheduler.step()

if __name__ == "__main__":
    main()