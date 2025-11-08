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

from utils import *
from dataset import *
from model import *

def main():
    # load data
    train_dataset = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=TRANSFORMS_TRAIN)
    val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=TRANSFORMS_VAL)

    train_dataset, val_dataset = get_subset(train_dataset, val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    # configure model
    #model = Model(num_channels=3, num_classes=NUM_CLASSES)
    #model = smp.DeepLabV3Plus(classes=NUM_CLASSES)
    model = smp.Segformer(
        encoder_name="mit_b2",                 # 最優：mit_b5 / 次佳 mit_b4；若無可用 pretrained，可選 mit_b5 並載 imagenet 或專用 pretrained
        encoder_depth=5,
        encoder_weights="imagenet",            # 若有 SegFormer 專用 pretrained 權重，改用該權重
        decoder_segmentation_channels=512,     # 頻道數由 256 提升為 512，提升 decoder 表示力
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,                       # 返回 logits，搭配混合損失
        upsampling=4,                          # 輸出尺寸可用後處理 resize；若需要一步到位改為 upsampling=input/4
        aux_params={"classes": NUM_CLASSES, "pooling":"avg", "dropout":0.1, "activation":None}
    )

    model_name = model.__class__.__name__

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # hyperparameters
    max_lr = LR
    num_epochs = EPOCHS
    weight_decay = WEIGHT_DECAY

    # loss function
    #criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    criterion1 = smp.losses.FocalLoss(mode='multiclass').to(device)
    criterion2 = smp.losses.DiceLoss(mode='multiclass').to(device)

    # lr adjustment
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs,
    #                                                 steps_per_epoch=len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = GradScaler()
    torch.cuda.empty_cache()

    train_losses, val_losses, val_iou, val_acc, train_iou, train_acc, lrs \
        = [], [], [], [], [], [], []

    min_loss, max_iou = np.inf, 0
    decrease, not_improve = 1, 0
    start_time = time.time()

    m_iou = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=0).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = timestamp_str
    folder_path = Path('./models/') / folder_name
    folder_path.mkdir(exist_ok=True, parents=True)

    for e in range(num_epochs):
        apply_warmup_lr(optimizer, e)
        running_loss, iou_score, acc = 0, 0, 0
        m_iou.reset()
        accuracy.reset()
        model.train()
        train_loop = tqdm.tqdm(train_loader,desc='Train', leave=True)
        for i, data in enumerate(train_loop):
            image, mask = data
            image, mask = image.to(device), mask.to(device).long()

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(image)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                loss = criterion1(output, mask)\
                     + criterion2(output, mask)
            
            scaler.scale(loss).backward()
            
            # 使用 scaler 更新權重
            scaler.step(optimizer)
            
            # 更新 scaler
            scaler.update()

            scheduler.step()

            m_iou.update(output, mask)
            accuracy.update(output, mask)
            running_loss += loss.item()

            if i % 2 ==0:
                train_loop.set_postfix(loss=(running_loss/(i+1)),
                                    mIoU=m_iou.compute().item(),  
                                    acc=accuracy.compute().item(),
                                    ) 
            lrs.append(get_lr(optimizer))
            
        
        epoch_train_loss = running_loss/len(train_loader)
        epoch_train_iou = m_iou.compute()
        epoch_train_acc = accuracy.compute()

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
                    loss = criterion1(output, mask)\
                        + criterion2(output, mask)

                m_iou.update(output, mask)
                accuracy.update(output, mask)
                val_loss += loss.item()

                val_loop.set_postfix(loss=val_loss/(i+1),
                                    mIoU=m_iou.compute().item(),
                                    acc=accuracy.compute().item()
                                    )
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_iou = m_iou.compute()
        epoch_val_acc = accuracy.compute()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        train_iou.append(epoch_train_iou.item())
        train_acc.append(epoch_train_acc.item())
        val_iou.append(epoch_val_iou.item())
        val_acc.append(epoch_val_acc.item())

        if epoch_val_loss < min_loss:
            min_loss = epoch_val_loss
            decrease+=1
            torch.save(model.state_dict(), folder_path / 'best.pt')
            info_file_path = folder_path / "info.txt"
            write_log(info_file_path, timestamp_str, max_lr, num_epochs, weight_decay, \
                min_loss, epoch_val_iou, epoch_val_acc, epoch_train_loss, epoch_train_iou,\
                    epoch_train_acc, e, model_name)
        
        if epoch_val_loss > min_loss:
            not_improve+=1
            print("Not improving:", not_improve)
            if not_improve>=TOL:
                print("Early stop")
                break
        else:
            not_improve = 0
        

        print("Epoch:{}/{}..".format(e+1, num_epochs),
            "Train Loss: {:.3f}..".format(epoch_train_loss),
            "Val Loss: {:.3f}..".format(epoch_val_loss),
            "Train mIoU:{:.3f}..".format(epoch_train_iou),
            "Val mIoU: {:.3f}..".format(epoch_val_iou),
            "Train Acc:{:.3f}..".format(epoch_train_acc),
            "Val Acc:{:.3f}..".format(epoch_val_acc))

    

    print(f"Finish, time taken: {(time.time()-start_time)/60:.2f}")
    plot_loss_lr(train_losses, val_losses, lrs, folder_path)
    plot_metrics(train_iou, val_iou, train_acc, val_acc, folder_path)
    torch.save(model.state_dict(), folder_path / 'last.pt')

if __name__ == '__main__':
    main()