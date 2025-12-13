import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 引入進度條

# 假設 dataset 和 utils 在您的專案結構中是可用的
from dataset import BCSSDataset
from utils import TRANSFORMS_VAL, NUM_CLASSES

def find_model_path(provided_path: str = None):
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return str(p)
        else:
            raise FileNotFoundError(f"Provided model path does not exist: {provided_path}")

    base = Path('./models')
    if not base.exists():
        raise FileNotFoundError("No `models/` directory found in project root. Provide --model-path.")

    candidates = []
    for sub in base.iterdir():
        if sub.is_dir():
            for name in ('best.pt', 'last.pt'):
                p = sub / name
                if p.exists():
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No model files (best.pt or last.pt) found under `models/`. Provide --model-path.")

    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])

def save_mask(mask_arr, out_path: Path, cmap='tab20'):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(mask_arr, cmap=cmap, interpolation='nearest')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def overlay_image(image_tensor, mask_arr, out_path: Path, alpha=0.5, cmap='tab20'):
    img = image_tensor.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(mask_arr, cmap=cmap, alpha=alpha, interpolation='nearest')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_iou_distribution(ious, m_iou, out_path: Path, title_prefix=""):
    """
    繪製每個類別的 IoU 分佈長條圖
    """
    plot_values = [x if not np.isnan(x) else 0.0 for x in ious]
    colors = ['skyblue' if not np.isnan(x) else 'lightgray' for x in ious]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(ious)), plot_values, color=colors)
    
    plt.xlabel('Class ID')
    plt.ylabel('IoU Score')
    plt.title(f'{title_prefix} Per-class IoU (mIoU: {m_iou:.4f})')
    plt.xticks(range(len(ious)))
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, val in zip(bars, ious):
        if not np.isnan(val):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def evaluate_dataset(model, dataloader, device, num_classes):
    """
    評估整個 Dataset 的全域 mIoU 和 Accuracy。
    這是「算總帳」的方式：先加總所有的 Intersection 和 Union，最後才除。
    """
    model.eval()
    
    # 初始化全域計數器
    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_correct = 0
    total_pixels = 0

    print(f"Evaluating on {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device).long() # BxHxW
            
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[0]

            # 預測
            if output.shape[1] > 1:
                preds = torch.argmax(output, dim=1) # BxHxW
            else:
                preds = (torch.sigmoid(output) > 0.5).squeeze(1).long()

            # 轉為 numpy 處理 (也可以用純 torch 實作，但 numpy 對邏輯運算很直觀)
            preds_np = preds.cpu().numpy()
            gt_np = masks.cpu().numpy()

            # 更新 Pixel Accuracy
            total_correct += (preds_np == gt_np).sum()
            total_pixels += gt_np.size

            # 更新每個類別的 Intersection 和 Union
            for c in range(num_classes):
                # 這裡使用位元運算加速
                pred_mask = (preds_np == c)
                gt_mask = (gt_np == c)
                
                inter = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                
                total_inter[c] += inter
                total_union[c] += union

    # 計算全域指標
    global_acc = total_correct / total_pixels
    
    # 計算全域 Per-class IoU
    # 處理除以零的情況：如果某個類別在整個 Dataset 的 GT 和 Pred 都沒出現 (Union=0)，
    # 按照標準通常設為 NaN 或忽略，但在訓練指標中通常視為 0 或忽略。
    # 這裡使用 np.divide 安全除法
    global_ious = []
    for c in range(num_classes):
        if total_union[c] == 0:
            global_ious.append(float('nan')) # 完全沒出現過的類別
        else:
            global_ious.append(total_inter[c] / total_union[c])
            
    global_miou = np.nanmean(global_ious)
    
    return global_acc, global_miou, global_ious

def main():
    parser = argparse.ArgumentParser(description='Test BCSS dataset metrics (Global & Single Sample)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--image-dir', type=str, default='./BCSS/val/', help='Path to BCSS images')
    parser.add_argument('--mask-dir', type=str, default='./BCSS/val_mask/', help='Path to BCSS masks')
    parser.add_argument('--index', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--random', action='store_true', help='Pick a random sample to visualize')
    parser.add_argument('--device', type=str, default=None, help='Torch device (cpu or cuda)')
    parser.add_argument('--outdir', type=str, default='./outputs/test_dataset/', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--skip-global', action='store_true', help='Skip global dataset evaluation (fast mode)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # 1. Load Model
    model_path = find_model_path(args.model_path)
    print(f"Loading model from: {model_path}")
    
    loaded = None
    try:
        loaded = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        try:
            loaded = torch.load(model_path, map_location=device)
        except Exception:
            raise
    except Exception as e:
        print(f'Warning: full-object load failed ({e}). Trying weights-only load...')
        try:
            loaded = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            raise
        except Exception:
            raise

    model = None
    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        state_dict = loaded.get('state_dict', loaded)
        try:
            import segmentation_models_pytorch as smp
            print('Building UnetPlusPlus(encoder="timm-efficientnet-b3") to load state_dict')
            model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=NUM_CLASSES)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f'Failed to reconstruct model: {e}')

    model = model.to(device)
    model.eval()

    # 2. Setup Dataset & Loader
    ds = BCSSDataset(args.image_dir, args.mask_dir, transform=TRANSFORMS_VAL)
    if len(ds) == 0:
        raise RuntimeError(f'No samples found in {args.image_dir}')
    
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Output setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.outdir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'report.txt'

    # 3. Global Evaluation (Optional but requested)
    global_acc, global_miou, global_ious = 0.0, 0.0, []
    if not args.skip_global:
        print("\n--- Starting Global Evaluation ---")
        global_acc, global_miou, global_ious = evaluate_dataset(model, loader, device, NUM_CLASSES)
        print(f"Global Pixel Acc: {global_acc:.4f}")
        print(f"Global mIoU:      {global_miou:.4f}")
        
        # Plot Global IoU
        plot_iou_distribution(global_ious, global_miou, out_dir / 'global_iou_bar_plot.png', title_prefix="Global Dataset")
    else:
        print("\nSkipping global evaluation.")

    # 4. Single Sample Visualization
    if args.random:
        idx = int(torch.randint(0, len(ds), (1,)).item())
    else:
        idx = args.index
    idx = max(0, min(idx, len(ds) - 1))
    
    print(f"\n--- Visualizing Sample Index: {idx} ---")
    image, mask = ds[idx]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_batch)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output.shape[1] > 1:
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        else:
            pred = (torch.sigmoid(output).squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype(np.int32)

    gt = mask.cpu().numpy()
    
    # Calculate single sample metrics
    correct = (pred == gt).sum()
    sample_acc = correct / gt.size
    sample_ious = []
    for c in range(NUM_CLASSES):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union == 0:
            sample_ious.append(float('nan'))
        else:
            sample_ious.append(inter / union)
    sample_miou = np.nanmean(sample_ious)

    # Save visualizations
    save_mask(pred, out_dir / 'sample_pred_mask.png')
    save_mask(gt, out_dir / 'sample_gt_mask.png')
    overlay_image(image, pred, out_dir / 'sample_overlay_pred.png')
    overlay_image(image, gt, out_dir / 'sample_overlay_gt.png')
    plot_iou_distribution(sample_ious, sample_miou, out_dir / 'sample_iou_bar_plot.png', title_prefix=f"Sample {idx}")

    # 5. Write Report
    with open(report_path, 'w') as f:
        f.write(f'Model: {model_path}\n')
        f.write(f'Device: {device}\n')
        f.write('=' * 30 + '\n')
        
        if not args.skip_global:
            f.write('GLOBAL DATASET METRICS:\n')
            f.write(f'  Global Pixel Acc: {global_acc:.6f}\n')
            f.write(f'  Global mIoU:      {global_miou:.6f}\n')
            f.write('  Per-class IoU:\n')
            for cid, iou in enumerate(global_ious):
                val_str = f"{iou:.6f}" if not np.isnan(iou) else "N/A"
                f.write(f'    Class {cid}: {val_str}\n')
            f.write('=' * 30 + '\n')

        f.write(f'SINGLE SAMPLE METRICS (Index {idx}):\n')
        f.write(f'  Sample Pixel Acc: {sample_acc:.6f}\n')
        f.write(f'  Sample mIoU:      {sample_miou:.6f}\n')
        f.write('  Per-class IoU:\n')
        for cid, iou in enumerate(sample_ious):
            val_str = f"{iou:.6f}" if not np.isnan(iou) else "N/A"
            f.write(f'    Class {cid}: {val_str}\n')

    print(f'Wrote outputs and report to: {out_dir}')

if __name__ == '__main__':
    main()