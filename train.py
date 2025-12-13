import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import time
import tqdm
from datetime import datetime
from pathlib import Path
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, Accuracy
from torch.amp import GradScaler, autocast
import numpy as np # Added numpy just in case

from utils import *
from dataset import *
from model import *

def main():
    # load data
    train_dataset = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=TRANSFORMS_TRAIN)
    val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=TRANSFORMS_VAL)

    train_dataset, val_dataset = get_subset(train_dataset, val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=5, pin_memory=True)

    encoder_name="timm-efficientnet-b3"
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        classes=NUM_CLASSES,
        decoder_attention_type="scse",  
        decoder_use_batchnorm=True,
        activation=None,
        encoder_depth=4,
        decoder_channels=(256, 128, 64, 32) # 14x14 from 224x224
    )

    model_name = model.__class__.__name__

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # hyperparameters
    max_lr = LR
    num_epochs = EPOCHS
    weight_decay = WEIGHT_DECAY

    # Compute class weights
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device, ignore_index=None)
    
    # loss function
    criterion0 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05).to(device)
    criterion1 = smp.losses.DiceLoss(mode='multiclass').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    
    steps_per_epoch = len(train_loader)

    # Use OneCycleLR scheduler: steps per epoch must be provided so scheduler
    # can compute the learning rate for each training batch across epochs.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
    )
    scaler = GradScaler()
    torch.cuda.empty_cache()

    train_losses, val_losses, val_iou, val_acc, train_iou, train_acc, lrs = [], [], [], [], [], [], []

    min_loss, max_iou = np.inf, 0
    decrease, not_improve = 1, 0
    best_val_iou = 0  
    start_time = time.time()

    m_iou = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = timestamp_str
    folder_path = Path('./models/') / folder_name
    folder_path.mkdir(exist_ok=True, parents=True)

    try: # Add try block to catch KeyboardInterrupt safely
        for e in range(num_epochs):
            #apply_warmup_lr(optimizer, e)
            running_loss, running_iou, running_acc = 0, 0, 0
            model.train()
            train_loop = tqdm.tqdm(train_loader, desc='Train', leave=True)
            
            for i, data in enumerate(train_loop):
                image, mask = data
                image, mask = image.to(device), mask.to(device).long()

                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    output = model(image)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    loss = 0.5 * criterion0(output, mask) + 0.5 * criterion1(output, mask) 

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

                # Step the OneCycleLR scheduler once per optimizer step (per batch)
                try:
                    scheduler.step()
                except Exception:
                    pass

                with torch.no_grad():
                    m_iou.reset()
                    accuracy.reset()
                    m_iou.update(output, mask)
                    accuracy.update(output, mask)
                    batch_iou = m_iou.compute().item()
                    batch_acc = accuracy.compute().item()
                
                running_loss += loss.item()
                running_iou += batch_iou
                running_acc += batch_acc

                if i % 2 ==0:
                    train_loop.set_postfix(loss=(running_loss/(i+1)),
                                        mIoU=(running_iou/(i+1)),  
                                        acc=(running_acc/(i+1)),
                                        lr=(get_lr(optimizer))
                                        ) 
            
                # scheduling is handled per-batch with OneCycleLR
            lrs.append(get_lr(optimizer))
            
            epoch_train_loss = running_loss/len(train_loader)
            epoch_train_iou = running_iou/len(train_loader)
            epoch_train_acc = running_acc/len(train_loader)

            model.eval()
            m_iou.reset()
            accuracy.reset()

            val_loss = 0
            val_loop = tqdm.tqdm(val_loader, desc='Val', leave=True)
            with torch.no_grad():
                for i, data in enumerate(val_loop):
                    image, mask = data
                    image, mask = image.to(device), mask.to(device).long()

                    with autocast(device_type='cuda'):
                        output = model(image)
                        if isinstance(output, (list, tuple)):
                            output = output[0]
                        loss = 0.5 * criterion0(output, mask) + 0.5 * criterion1(output, mask) 

                    m_iou.update(output.detach(), mask)
                    accuracy.update(output.detach(), mask)
                    val_loss += loss.item()

                    val_loop.set_postfix(loss=val_loss/(i+1),
                                        mIoU=m_iou.compute().item(),
                                        acc=accuracy.compute().item()
                                        )
            
            epoch_val_loss = val_loss/len(val_loader)
            epoch_val_iou = m_iou.compute()
            epoch_val_acc = accuracy.compute()

            # --- Update History Lists ---
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_iou.append(epoch_train_iou)
            train_acc.append(epoch_train_acc)
            val_iou.append(epoch_val_iou.item())
            val_acc.append(epoch_val_acc.item())
            
            # --- START: UPDATE PLOTS IMMEDIATELY ---
            # We call the plot functions here so they update at the end of every epoch
            plot_loss_lr(train_losses, val_losses, lrs, folder_path)
            plot_metrics(train_iou, val_iou, train_acc, val_acc, folder_path)
            # --- END: UPDATE PLOTS IMMEDIATELY ---

            # Save best model
            if epoch_val_iou.item() > best_val_iou:
                best_val_iou = epoch_val_iou.item()
                min_loss = epoch_val_loss
                decrease += 1
                not_improve = 0
                # Save the full model object so it can be loaded without rebuilding
                torch.save(model, folder_path / 'best.pt')
                info_file_path = folder_path / "info.txt"
                write_log(info_file_path, timestamp_str, max_lr, num_epochs, weight_decay, \
                    min_loss, epoch_val_iou, epoch_val_acc, epoch_train_loss, epoch_train_iou,\
                        epoch_train_acc, e, model_name, encoder_name)
                print(f"✓ New best model! Val mIoU: {best_val_iou:.4f}")
            else:
                not_improve += 1
                print(f"Not improving: {not_improve}/{TOL} (Best Val mIoU: {best_val_iou:.4f})")
                if not_improve >= TOL:
                    print("Early stopping triggered!")
                    break
            
            print("Epoch:{}/{}..".format(e+1, num_epochs),
                "Train Loss: {:.3f}..".format(epoch_train_loss),
                "Val Loss: {:.3f}..".format(epoch_val_loss),
                "Train mIoU:{:.3f}..".format(epoch_train_iou),
                "Val mIoU: {:.3f}..".format(epoch_val_iou),
                "Train Acc:{:.3f}..".format(epoch_train_acc),
                "Val Acc:{:.3f}..".format(epoch_val_acc))

    except KeyboardInterrupt:
        print("\nStopping early. Saving last model state...")
    
    # Final cleanup
    print(f"Finish, time taken: {(time.time()-start_time)/60:.2f}")
    # Run one last time just in case
    plot_loss_lr(train_losses, val_losses, lrs, folder_path)
    plot_metrics(train_iou, val_iou, train_acc, val_acc, folder_path)
    # Save the last model (full object) as well
    torch.save(model, folder_path / 'last.pt')

if __name__ == '__main__':
    main()