#!/usr/bin/env python3
"""
虹膜分割模型训练脚本
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from models.segmentation import build_segmentation_model
from utils.losses import CombinedSegmentationLoss
from utils.dataset import build_segmentation_datasets


def dice_score(preds, targets, eps=1e-7):
    """计算Dice分数"""
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=[1, 2, 3])
    denom = preds.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()


def iou_score(preds, targets, eps=1e-7):
    """计算IoU分数"""
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=[1, 2, 3])
    union = (preds + targets - preds * targets).sum(dim=[1, 2, 3])
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # 计算指标
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{dice:.4f}',
            'IoU': f'{iou:.4f}'
        })
    
    return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算指标
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)


def save_checkpoint(model, optimizer, epoch, best_metric, save_path):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train Iris Segmentation Model')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--output_dir', type=str, default='outputs/segmentation', help='Output directory')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3', help='Backbone architecture')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建数据集
    print('Building datasets...')
    train_dataset, val_dataset = build_segmentation_datasets(
        args.data_root, 
        img_size=args.img_size, 
        val_split=args.val_split
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # 构建模型
    print('Building model...')
    model = build_segmentation_model(
        num_classes=1,
        pretrained=args.pretrained,
        backbone=args.backbone
    )
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = CombinedSegmentationLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 训练循环
    best_dice = 0.0
    print('Starting training...')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_dice)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_dice, epoch)
        writer.add_scalar('Dice/Val', val_dice, epoch)
        writer.add_scalar('IoU/Train', train_iou, epoch)
        writer.add_scalar('IoU/Val', val_iou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f'New best model saved with Dice: {best_dice:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs, best_dice,
        os.path.join(args.output_dir, 'final_model.pth')
    )
    
    print(f'\nTraining completed! Best Dice: {best_dice:.4f}')
    writer.close()


if __name__ == '__main__':
    main()
