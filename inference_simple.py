import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
from net.model import PromptIR

def main():
    # ==========================================
    # 1. 路徑與參數設定
    # ==========================================
    test_dir = "/disk2/ccchen/DL_CV_class/HW/HW4/hw4_realse_dataset/test/degraded"
    output_file = "pred.npz"
    ckpt_path = "/disk2/ccchen/DL_CV_class/HW/HW4/checkpoints_train_all/ema_epoch_10.pth"  # 確保這是你目前最好的模型
    
    if not os.path.exists(ckpt_path):
        print(f"[錯誤] 找不到權重檔案: {ckpt_path}")
        return

    # ==========================================
    # 2. 模型初始化
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用設備: {device}")
    
    # 建立模型並載入權重 (必須開啟 decoder=True)
    model = PromptIR(inp_channels=3, out_channels=3, decoder=True).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 取得所有測試檔案並排序
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))])
    images_dict = {}

    print(f"開始執行純淨版推論 (無 TTA)，共 {len(test_files)} 張...")

    # ==========================================
    # 3. 推論主迴圈
    # ==========================================
    with torch.no_grad():
        for filename in test_files:
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert('RGB')
            
            # 轉換為 Tensor 並搬移到 GPU
            x = TF.to_tensor(img).unsqueeze(0).to(device)
            
            # 使用混合精度加速 (AMP)
            with amp.autocast():
                # --- 單次預測，無任何 TTA 操作 ---
                output = model(x)
                
                # 數值限制在 [0.0, 1.0]
                output = torch.clamp(output, 0.0, 1.0)
            
            # ==========================================
            # 4. 格式轉換與存檔準備
            # ==========================================
            # 移除 Batch 維度 -> [3, 256, 256]
            output_np = output.squeeze(0).cpu().float().numpy()
            
            # 依照作業規定轉為 uint8 (0-255)
            output_np = (output_np * 255.0).round().astype(np.uint8)
            
            # 存入字典
            images_dict[filename] = output_np
            
            if len(images_dict) % 20 == 0:
                print(f"進度: {len(images_dict)}/{len(test_files)}")

    # ==========================================
    # 5. 儲存為 npz 檔案
    # ==========================================
    print(f"\n正在將結果存至 {output_file}...")
    np.savez(output_file, **images_dict)
    print("🎉 處理完成！")

if __name__ == "__main__":
    main()