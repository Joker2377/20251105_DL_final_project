import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# 導入您現有的 dataset 和 utils
# 這些檔案提供了必要的變數，如：
# VAL_IMAGE_PATH, VAL_MASK_PATH, TRANSFORMS_VAL, NUM_CLASSES, BCSSDataset
from dataset import BCSSDataset
from utils import * # --- 設定 ---
N_SAMPLES = 5  # 您想要顯示的隨機樣本數量 (即 n ROWS)
MODEL_DIR = Path('./models') # 儲存所有訓練模型的根目錄

# ----------------------------------------------------------------------------

def find_latest_model_path(base_dir: Path) -> Path:
    """在 models/ 中尋找最新的模型資料夾 (根據時間戳) 並返回 best.pt 的路徑"""
    try:
        # 獲取所有子目錄 (即每個 timestamp_str)
        all_run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not all_run_dirs:
            raise FileNotFoundError("在 'models' 目錄下找不到任何模型資料夾。")
        
        # 根據名稱 (YYYY-MM-DD_HH-MM-SS) 排序，找到最新的
        latest_run_dir = sorted(all_run_dirs)[-1]
        model_path = latest_run_dir / 'best.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"在 {latest_run_dir} 中找不到 'best.pt'。")
            
        print(f"✅ 成功找到最新的模型: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ 尋找模型時發生錯誤: {e}")
        print("請確保您已經訓練過模型，並且 'best.pt' 存在於 './models/<timestamp_folder>/' 中。")
        exit()

def unnormalize_image(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    將 Albumentations/PyTorch 標準化後的影像張量還原，以便 Matplotlib 顯示。
    輸入: (C, H, W) 的張量
    輸出: (H, W, C) 的 Numpy 陣列
    """
    # 複製張量，避免修改原始數據
    tensor = tensor.clone().cpu() 
    
    # 標準化時的均值和標準差
    mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    
    # 反標準化: tensor = (tensor * std) + mean
    tensor.mul_(std).add_(mean)
    
    # 將張量轉換為 (H, W, C) 格式
    np_image = tensor.permute(1, 2, 0).numpy()
    
    # 將數值裁剪到 [0, 1] 範圍內，這是 Matplotlib 顯示 RGB 影像的標準範圍
    np_image = np.clip(np_image, 0, 1)
    
    return np_image

def main():
    # 0. 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 找到並加載模型
    model_path = find_latest_model_path(MODEL_DIR)
    
    # train.py 使用 torch.save(model, ...) 儲存了整個模型，
    # 而不是 model.state_dict()，所以我們用 torch.load() 直接加載
    try:
        # 初始化模型架構
        model = smp.UnetPlusPlus(classes=NUM_CLASSES)
        # 加載 state_dict
        state_dict = torch.load(model_path, map_location=device)
        
        # 移除 'module.' 前綴 (適用於使用 DataParallel 訓練的模型)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # 設置為評估模式
    except Exception as e:
        print(f"❌ 加載模型失敗: {e}")
        print("您的 'best.pt' 可能已損壞或與 DeepLabV3Plus 結構不符。")
        print("請確認模型架構與保存的 state_dict 匹配。")
        exit()

    # 2. 準備驗證資料集
    # 我們使用 TRANSFORMS_VAL，它只包含 Resize 和 Normalize
    val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=TRANSFORMS_VAL)
    print(f"已加載驗證資料集，共 {len(val_dataset)} 張影像。")

    # 3. 隨機選取 N_SAMPLES 個索引
    if len(val_dataset) < N_SAMPLES:
        print(f"警告: 樣本數 (N_SAMPLES={N_SAMPLES}) 大於資料集大小 ({len(val_dataset)})。")
        print(f"將顯示所有 {len(val_dataset)} 張影像。")
        indices = list(range(len(val_dataset)))
        n_rows = len(val_dataset)
    else:
        indices = random.sample(range(len(val_dataset)), N_SAMPLES)
        n_rows = N_SAMPLES

    # 4. 建立繪圖網格
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5))
    
    # 處理 n_rows=1 的特殊情況，axes 不是 2D 陣列
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # 設定第一行的標題
    axes[0, 0].set_title("Original Image", fontsize=16)
    axes[0, 1].set_title("Predicted Mask", fontsize=16)
    axes[0, 2].set_title("Ground Truth Mask", fontsize=16)

    # 5. 迭代、預測並繪圖
    with torch.no_grad(): # 關閉梯度計算
        for i, idx in enumerate(indices):
            # 從資料集獲取數據
            # image_tensor: (3, H, W), gt_mask: (H, W)
            image_tensor, gt_mask = val_dataset[idx] 
            
            # 準備模型輸入：(1, 3, H, W) 並移至 device
            input_tensor = image_tensor.unsqueeze(0).to(device)

            # 執行預測
            output = model(input_tensor)
            
            # 處理輸出
            # output: (1, NUM_CLASSES, H, W)
            # pred_mask: (H, W)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 將標準化的影像還原
            display_image = unnormalize_image(image_tensor, 
                                              mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

            # --- 繪圖 ---
            
            # 欄 1: 原始影像
            ax = axes[i, 0]
            ax.imshow(display_image)
            ax.set_ylabel(f"Sample #{idx}", fontsize=12)
            ax.axis('off')

            # 欄 2: 預測遮罩
            ax = axes[i, 1]
            ax.imshow(pred_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            ax.axis('off')

            # 欄 3: 真實遮罩
            ax = axes[i, 2]
            ax.imshow(gt_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'./{model_path.parent.name}_predictions.png') # 以模型名稱儲存
    print(f"\n✅ 繪圖完成！已儲存至 ./{model_path.parent.name}_predictions.png")
    plt.show()


if __name__ == '__main__':
    main()