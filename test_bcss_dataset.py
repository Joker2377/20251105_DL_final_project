import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 引入 torchmetrics ===
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassF1Score

try:
    from dataset import BCSSDataset
    from utils import TRANSFORMS_VAL
except ImportError:
    print("警告: 找不到 dataset.py 或 utils.py，請確保這些檔案存在於專案目錄中。")
    BCSSDataset = None
    TRANSFORMS_VAL = None

def find_model_path(provided_path: str = None):
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return str(p)
        else:
            raise FileNotFoundError(f"Provided model path does not exist: {provided_path}")

    base = Path('./models')
    if not base.exists():
        raise FileNotFoundError("No `models/` directory found. Provide --model-path.")

    candidates = []
    for sub in base.iterdir():
        if sub.is_dir():
            for name in ('best.pt', 'last.pt'):
                p = sub / name
                if p.exists():
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No model files (best.pt/last.pt) found. Provide --model-path.")

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
    plot_values = [x if not np.isnan(x) else 0.0 for x in ious]
    colors = ['skyblue' if not np.isnan(x) else 'lightgray' for x in ious]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(ious)), plot_values, color=colors)
    
    plt.xlabel('Class ID')
    plt.ylabel('Score')
    plt.title(f'{title_prefix} Per-class Metrics (Mean: {m_iou:.4f})')
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
    model.eval()
    
    metrics = MetricCollection({
        'iou_per_class': MulticlassJaccardIndex(num_classes=num_classes, average=None),
        'dice_per_class': MulticlassF1Score(num_classes=num_classes, average=None),
        'pixel_acc': MulticlassAccuracy(num_classes=num_classes, average='micro')
    }).to(device)

    print(f"Evaluating on {len(dataloader.dataset)} samples with num_classes={num_classes}...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device).long()
            
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[0]

            if output.shape[1] > 1:
                preds = torch.argmax(output, dim=1)
            else:
                preds = (torch.sigmoid(output) > 0.5).squeeze(1).long()

            metrics.update(preds, masks)

    results = metrics.compute()
    
    global_acc = results['pixel_acc'].item()
    iou_per_class = results['iou_per_class'].cpu().numpy()
    dice_per_class = results['dice_per_class'].cpu().numpy()
    
    global_miou = np.nanmean(iou_per_class)
    global_mdice = np.nanmean(dice_per_class)

    metrics.reset()
    
    return global_acc, global_miou, iou_per_class, global_mdice, dice_per_class

def main():
    parser = argparse.ArgumentParser(description='Test BCSS dataset metrics')
    parser.add_argument('--model-path', type=str, default=None, help='Path to .pt model')
    parser.add_argument('--image-dir', type=str, default='./BCSS/val/', help='Path to images')
    parser.add_argument('--mask-dir', type=str, default='./BCSS/val_mask/', help='Path to masks')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes for metric calculation (Default: 3)')
    
    parser.add_argument('--index', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--random', action='store_true', help='Visualize random sample')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--outdir', type=str, default='./outputs/test_dataset/')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--skip-global', action='store_true', help='Skip full dataset evaluation')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # 1. Load Model (Robust Loading)
    model_path = find_model_path(args.model_path)
    print(f"Loading model from: {model_path}")
    print(f"Target Num Classes: {args.num_classes}")
    
    loaded = None
    try:
        loaded = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Standard load failed, trying weights_only=True. Error: {e}")
        loaded = torch.load(model_path, map_location=device, weights_only=True)

    model = None
    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        state_dict = loaded.get('state_dict', loaded)
        import segmentation_models_pytorch as smp
        print(f'Building UnetPlusPlus(encoder="timm-efficientnet-b3") with classes={args.num_classes}...')
        model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=args.num_classes)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 2. Setup Dataset
    ds = BCSSDataset(args.image_dir, args.mask_dir, transform=TRANSFORMS_VAL)
    if len(ds) == 0:
        raise RuntimeError(f'No samples found in {args.image_dir}')
    
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Output setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = args.name if args.name else timestamp
    if args.name and (Path(args.outdir) / folder_name).exists():
        folder_name = f"{folder_name}_{timestamp}"

    out_dir = Path(args.outdir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'report.txt'

    # 3. Global Evaluation
    global_acc, global_miou, global_ious, global_mdice, global_dices = 0.0, 0.0, [], 0.0, []
    
    if not args.skip_global:
        print("\n" + "="*40)
        print(" STARTING GLOBAL EVALUATION ")
        print("="*40)
        
        # 傳入 args.num_classes
        global_acc, global_miou, global_ious, global_mdice, global_dices = evaluate_dataset(model, loader, device, args.num_classes)
        
        print("-" * 30)
        print(f"Global Pixel Acc: {global_acc:.4f}")
        print(f"Global mDice:     {global_mdice:.4f}")
        print(f"Global mIoU:      {global_miou:.4f}")
        print("-" * 30)
        print("Per-class IoU:")
        for idx, iou in enumerate(global_ious):
            val_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  Class {idx}: {val_str}")
        print("-" * 30)
        
        plot_iou_distribution(global_ious, global_miou, out_dir / 'global_iou_bar_plot.png', title_prefix="Global IoU")
        plot_iou_distribution(global_dices, global_mdice, out_dir / 'global_dice_bar_plot.png', title_prefix="Global Dice")
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
    mask_batch = mask.unsqueeze(0).to(device).long()

    with torch.no_grad():
        output = model(image_batch)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output.shape[1] > 1:
            preds = torch.argmax(output, dim=1)
        else:
            preds = (torch.sigmoid(output) > 0.5).long().squeeze(1)

    pred_np = preds.squeeze(0).cpu().numpy()
    gt_np = mask.cpu().numpy()

    from torchmetrics.functional import jaccard_index, f1_score, accuracy
    
    s_acc = accuracy(preds, mask_batch, task="multiclass", num_classes=args.num_classes, average='micro').item()
    s_miou = jaccard_index(preds, mask_batch, task="multiclass", num_classes=args.num_classes, average='macro').item()
    s_mdice = f1_score(preds, mask_batch, task="multiclass", num_classes=args.num_classes, average='macro').item()
    s_ious = jaccard_index(preds, mask_batch, task="multiclass", num_classes=args.num_classes, average='none').cpu().numpy()

    save_mask(pred_np, out_dir / 'sample_pred_mask.png')
    save_mask(gt_np, out_dir / 'sample_gt_mask.png')
    overlay_image(image, pred_np, out_dir / 'sample_overlay_pred.png')
    overlay_image(image, gt_np, out_dir / 'sample_overlay_gt.png')
    plot_iou_distribution(s_ious, s_miou, out_dir / 'sample_iou_bar_plot.png', title_prefix=f"Sample {idx}")

    # 5. Write Report
    with open(report_path, 'w') as f:
        f.write(f'Model: {model_path}\n')
        f.write(f'Device: {device}\n')
        f.write(f'Num Classes: {args.num_classes}\n')
        f.write('=' * 30 + '\n')
        
        if not args.skip_global:
            f.write('GLOBAL DATASET METRICS:\n')
            f.write(f'  Global Pixel Acc: {global_acc:.6f}\n')
            f.write(f'  Global mIoU:      {global_miou:.6f}\n')
            f.write(f'  Global mDice:     {global_mdice:.6f}\n')
            f.write('  Per-class IoU:\n')
            for cid, iou in enumerate(global_ious):
                val_str = f"{iou:.6f}" if not np.isnan(iou) else "N/A"
                f.write(f'    Class {cid}: {val_str}\n')
            f.write('  Per-class Dice:\n')
            for cid, dice in enumerate(global_dices):
                val_str = f"{dice:.6f}" if not np.isnan(dice) else "N/A"
                f.write(f'    Class {cid}: {val_str}\n')
            f.write('=' * 30 + '\n')

        f.write(f'SINGLE SAMPLE METRICS (Index {idx}):\n')
        f.write(f'  Sample Pixel Acc: {s_acc:.6f}\n')
        f.write(f'  Sample mIoU:      {s_miou:.6f}\n')
        f.write(f'  Sample mDice:     {s_mdice:.6f}\n')
        f.write('  Per-class IoU:\n')
        for cid, iou in enumerate(s_ious):
            val_str = f"{iou:.6f}" if not np.isnan(iou) else "N/A"
            f.write(f'    Class {cid}: {val_str}\n')

    print(f'\nWrote outputs and report to: {out_dir}')

if __name__ == '__main__':
    main()