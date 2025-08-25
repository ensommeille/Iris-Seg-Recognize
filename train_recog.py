#!/usr/bin/env python3
"""
虹膜识别模型训练脚本
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

from models.recognition import build_recognition_model
from utils.losses import CombinedRecognitionLoss
from utils.dataset import build_recognition_datasets


def accuracy(outputs, labels):
    """计算分类准确率"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_arcface_loss = 0
    total_triplet_loss = 0
    total_acc = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, embeddings = model(images, labels)
        loss, arcface_loss, triplet_loss = criterion(logits, embeddings, labels)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = accuracy(logits, labels)
        
        total_loss += loss.item()
        total_arcface_loss += arcface_loss.item()
        total_triplet_loss += triplet_loss.item()
        total_acc += acc
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ArcFace': f'{arcface_loss.item():.4f}',
            'Triplet': f'{triplet_loss.item():.4f}',
            'Acc': f'{acc:.4f}'
        })
    
    return (total_loss / len(dataloader), 
            total_arcface_loss / len(dataloader),
            total_triplet_loss / len(dataloader),
            total_acc / len(dataloader))


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_arcface_loss = 0
    total_triplet_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            logits, embeddings = model(images, labels)
            loss, arcface_loss, triplet_loss = criterion(logits, embeddings, labels)
            
            # 计算准确率
            acc = accuracy(logits, labels)
            
            total_loss += loss.item()
            total_arcface_loss += arcface_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_acc += acc
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ArcFace': f'{arcface_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'Acc': f'{acc:.4f}'
            })
    
    return (total_loss / len(dataloader), 
            total_arcface_loss / len(dataloader),
            total_triplet_loss / len(dataloader),
            total_acc / len(dataloader))


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
    parser = argparse.ArgumentParser(description='Train Iris Recognition Model')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--output_dir', type=str, default='outputs/recognition', help='Output directory')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3', help='Backbone architecture')
    parser.add_argument('--embedding_size', type=int, default=512, help='Embedding size')
    parser.add_argument('--arcface_weight', type=float, default=1.0, help='ArcFace loss weight')
    parser.add_argument('--triplet_weight', type=float, default=0.5, help='Triplet loss weight')
    parser.add_argument('--triplet_margin', type=float, default=0.2, help='Triplet loss margin')
    parser.add_argument('--use_mask', action='store_true', help='Use segmentation mask')
    
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
    train_dataset, val_dataset = build_recognition_datasets(
        args.data_root, 
        img_size=args.img_size, 
        val_split=args.val_split,
        use_mask=args.use_mask
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
    print(f'Number of classes: {train_dataset.get_num_classes()}')
    
    # 构建模型
    print('Building model...')
    model = build_recognition_model(
        num_classes=train_dataset.get_num_classes(),
        embedding_size=args.embedding_size,
        pretrained=args.pretrained,
        backbone=args.backbone
    )
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = CombinedRecognitionLoss(
        arcface_weight=args.arcface_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 训练循环
    best_acc = 0.0
    print('Starting training...')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_loss, train_arcface, train_triplet, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_arcface, val_triplet, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('ArcFace/Train', train_arcface, epoch)
        writer.add_scalar('ArcFace/Val', val_arcface, epoch)
        writer.add_scalar('Triplet/Train', train_triplet, epoch)
        writer.add_scalar('Triplet/Val', val_triplet, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train - Loss: {train_loss:.4f}, ArcFace: {train_arcface:.4f}, Triplet: {train_triplet:.4f}, Acc: {train_acc:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, ArcFace: {val_arcface:.4f}, Triplet: {val_triplet:.4f}, Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f'New best model saved with Acc: {best_acc:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs, best_acc,
        os.path.join(args.output_dir, 'final_model.pth')
    )
    
    # 保存人员ID映射
    person_ids = train_dataset.get_person_ids()
    import json
    with open(os.path.join(args.output_dir, 'person_ids.json'), 'w') as f:
        json.dump(person_ids, f, indent=2)
    
    print(f'\nTraining completed! Best Acc: {best_acc:.4f}')
    writer.close()


if __name__ == '__main__':
    main()
