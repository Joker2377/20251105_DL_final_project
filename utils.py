import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

TRAIN_IMAGE_PATH = "./BCSS_512/train_512/"
TRAIN_MASK_PATH = "./BCSS_512/train_mask_512/"
VAL_IMAGE_PATH = "./BCSS_512/val_512/"
VAL_MASK_PATH = "./BCSS_512/val_mask_512/"
BATCH_SIZE = 64
NUM_CLASSES = 21
LR = 1e-5
EPOCHS = 200
WEIGHT_DECAY =0.01
NUM_SUBSET=100000
NUM_VAL_SUBSET = 100000
TOL = 1000
WARMUP_EPOCHS = 5

# Define transformations using Albumentations
TRANSFORMS_TRAIN = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(p=0.2),

    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    # A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

TRANSFORMS_VAL = A.Compose([
    A.Resize(224, 224),     
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def apply_warmup_lr(optimizer, epoch, warmup_epochs=WARMUP_EPOCHS, base_lr=LR):
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_loss_lr(train_losses, val_losses, lrs, base_path: Path):
    """
    繪製兩個子圖:
    1. 左圖: 訓練 (Train) vs 驗證 (Validation) 的 Loss
    2. 右圖: 學習率 (Learning Rate) 變化
    """
    # 清理數據
    train_losses_clean = train_losses
    val_losses_clean = val_losses
    lrs_clean = lrs

    # 創建 epoch 軸 (用於 Loss)
    epochs = range(1, len(train_losses_clean) + 1)
    # 創建 step 軸 (用於 Learning Rate)
    steps = range(1, len(lrs_clean) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- 子圖 1: Loss ---
    ax1.plot(epochs, train_losses_clean, label='Train Loss')
    ax1.plot(epochs, val_losses_clean, label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- 子圖 2: Learning Rate ---
    # 注意: lrs 是 per-step 記錄的, 所以 x 軸是 steps
    ax2.plot(steps, lrs_clean, label='Learning Rate', color='green')
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(base_path / "loss.png")
    plt.show()

def plot_metrics(train_iou, val_iou, train_acc, val_acc, base_path: Path):
    """
    繪製兩個子圖:
    1. 左圖: 訓練 (Train) vs 驗證 (Validation) 的 mIoU
    2. 右圖: 訓練 (Train) vs 驗證 (Validation) 的 Accuracy
    """
    # 清理數據
    train_iou_clean = train_iou
    val_iou_clean = val_iou
    train_acc_clean = train_acc
    val_acc_clean = val_acc

    # 創建 epoch 軸
    epochs = range(1, len(train_iou_clean) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- 子圖 1: mIoU ---
    ax1.plot(epochs, train_iou_clean, label='Train mIoU')
    ax1.plot(epochs, val_iou_clean, label='Validation mIoU')
    ax1.set_title('mIoU (Jaccard Index)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('mIoU')
    ax1.legend()
    ax1.grid(True)

    # --- 子圖 2: Accuracy ---
    ax2.plot(epochs, train_acc_clean, label='Train Accuracy')
    ax2.plot(epochs, val_acc_clean, label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(base_path / "metrics.png")
    plt.show()

def get_subset(train_dataset, val_dataset, num_train=NUM_SUBSET, num_val=NUM_VAL_SUBSET):
    total_samples = len(train_dataset)
    subset_size = min(num_train, total_samples)
    indices = torch.randperm(total_samples)[:subset_size]

    train_dataset = Subset(train_dataset, indices)
    print(f'Train Sample: {len(train_dataset)}')

    total_samples = len(val_dataset)
    subset_size = min(num_val, total_samples)
    indices = torch.randperm(total_samples)[:subset_size]

    val_dataset = Subset(val_dataset, indices)
    print(f'Validation Sample: {len(val_dataset)}')
    
    return train_dataset, val_dataset

def write_log(info_file_path, timestamp_str, max_lr, num_epochs, weight_decay, \
              min_loss, epoch_val_iou, epoch_val_acc, epoch_train_loss, epoch_train_iou,\
                epoch_train_acc, e, model_name):
    with open(info_file_path, "w", encoding='utf-8') as f:
            f.write(f"Training run: {timestamp_str}\n")
            f.write(f"Using model: {model_name}\n")
            f.write("="*40 + "\n\n")

            f.write("## Hyperparameters\n")
                        # 假設 BATCH_SIZE, NUM_CLASSES, NUM_SUBSET 是可用的全域常數
            f.write(f"  - Max Learning Rate: {max_lr}\n")
            f.write(f"  - Epochs: {num_epochs}\n")
            f.write(f"  - Batch Size: {BATCH_SIZE}\n") 
            f.write(f"  - Weight Decay: {weight_decay}\n")
            f.write(f"  - Num Classes: {NUM_CLASSES}\n")
            f.write(f"  - Subset Size: {NUM_SUBSET}\n")
            f.write(f"  - Val Subset Size: {NUM_VAL_SUBSET}\n")
            f.write("\n")

            f.write(f"## Best Model Metrics (Found at Epoch {e + 1})\n")
            f.write(f"  - Validation Loss: {min_loss:.4f}\n")
            f.write(f"  - Validation mIoU: {epoch_val_iou:.4f}\n")
            f.write(f"  - Validation Acc: {epoch_val_acc:.4f}\n")
            f.write("\n")

            f.write(f"## Corresponding Train Metrics (at Epoch {e + 1})\n")
            f.write(f"  - Train Loss: {epoch_train_loss:.4f}\n")
            f.write(f"  - Train mIoU: {epoch_train_iou:.4f}\n")
            f.write(f"  - Train Acc: {epoch_train_acc:.4f}\n")