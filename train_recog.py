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


def topk_accuracy(outputs, labels, ks=(1,5)):
    """计算Top-k准确率，返回字典{1:acc1, 5:acc5}（k存在时）"""
    maxk = max(ks)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        if k <= outputs.size(1):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = (correct_k.item() / labels.size(0))
    return res


def train_epoch(model, dataloader, criterion, optimizer, device, classification_enabled=True):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_arcface_loss = 0
    total_triplet_loss = 0
    total_top1 = 0
    total_top5 = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        if classification_enabled:
            logits, embeddings = model(images, labels)
        else:
            embeddings = model(images, labels=None)
            logits = None
        loss, arcface_loss, triplet_loss = criterion(logits, embeddings, labels)
        loss.backward()
        optimizer.step()
        
        # 计算准确率（仅当启用分类时）
        if classification_enabled and (logits is not None):
            accs = topk_accuracy(logits, labels, ks=(1,5))
            acc1 = accs.get(1, 0.0)
            acc5 = accs.get(5, 0.0)
        else:
            acc1, acc5 = 0.0, 0.0
        
        total_loss += loss.item()
        total_arcface_loss += arcface_loss.item()
        total_triplet_loss += triplet_loss.item()
        total_top1 += acc1
        total_top5 += acc5
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ArcFace': f'{arcface_loss.item():.4f}',
            'Triplet': f'{triplet_loss.item():.4f}',
            'Top1': f'{acc1:.6f}' if classification_enabled else '-',
            'Top5': f'{acc5:.6f}' if classification_enabled else '-',
        })
    
    return (total_loss / len(dataloader), 
            total_arcface_loss / len(dataloader),
            total_triplet_loss / len(dataloader),
            (total_top1 / len(dataloader)) if classification_enabled else 0.0,
            (total_top5 / len(dataloader)) if classification_enabled else 0.0)


def validate_epoch(model, dataloader, criterion, device, classification_enabled=True):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_arcface_loss = 0
    total_triplet_loss = 0
    total_top1 = 0
    total_top5 = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            if classification_enabled:
                logits, embeddings = model(images, labels)
            else:
                embeddings = model(images, labels=None)
                logits = None
            loss, arcface_loss, triplet_loss = criterion(logits, embeddings, labels)
            
            # 计算准确率（仅当启用分类时）
            if classification_enabled and (logits is not None):
                accs = topk_accuracy(logits, labels, ks=(1,5))
                acc1 = accs.get(1, 0.0)
                acc5 = accs.get(5, 0.0)
            else:
                acc1, acc5 = 0.0, 0.0
            
            total_loss += loss.item()
            total_arcface_loss += arcface_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_top1 += acc1
            total_top5 += acc5
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ArcFace': f'{arcface_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'Top1': f'{acc1:.6f}' if classification_enabled else '-',
                'Top5': f'{acc5:.6f}' if classification_enabled else '-',
            })
    
    return (total_loss / len(dataloader), 
            total_arcface_loss / len(dataloader),
            total_triplet_loss / len(dataloader),
            (total_top1 / len(dataloader)) if classification_enabled else 0.0,
            (total_top5 / len(dataloader)) if classification_enabled else 0.0)


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
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
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
    # 数据集划分与归一化相关
    parser.add_argument('--split_mode', type=str, default='eye', choices=['eye', 'random'], help='Dataset split mode: eye (L 80/20, R test) or random')
    parser.add_argument('--normalize_iris', action='store_true', default=True, help='Normalize iris to pseudo-polar 64x256 before training')
    parser.add_argument('--norm_H', type=int, default=64, help='Iris normalization height (radial samples)')
    parser.add_argument('--norm_W', type=int, default=256, help='Iris normalization width (angular samples)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training (model+optimizer+epoch)')
    parser.add_argument('--init_model', type=str, default='', help='Init model weights from checkpoint (model only, epoch=0)')
    
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
    if args.split_mode == 'eye':
        from utils.dataset import build_recognition_datasets_eye_split
        train_dataset, val_dataset, test_dataset = build_recognition_datasets_eye_split(
            args.data_root,
            seed=42,
            use_mask=args.use_mask,
            normalize=args.normalize_iris,
            norm_H=args.norm_H,
            norm_W=args.norm_W,
        )
        print(f'Test samples (Right eye): {len(test_dataset)}')
    else:
        train_dataset, val_dataset = build_recognition_datasets(
            args.data_root, 
            img_size=args.img_size, 
            val_split=args.val_split,
            use_mask=args.use_mask,
            normalize=args.normalize_iris,
            norm_H=args.norm_H,
            norm_W=args.norm_W,
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
    print(f'Number of classes (person-eye): {train_dataset.get_num_classes()}')
    
    # 构建模型
    print('Building model...')
    model = build_recognition_model(
        num_classes=train_dataset.get_num_classes(),
        embedding_size=args.embedding_size,
        pretrained=args.pretrained,
        backbone=args.backbone
    )
    model = model.to(device)

    # 根据配置决定是否启用分类分支（ArcFace）
    classification_enabled = (args.arcface_weight > 0.0)
    if not classification_enabled:
        print('Classification branch disabled: training with Triplet loss only.')

    # 损失函数和优化器
    criterion = CombinedRecognitionLoss(
        arcface_weight=args.arcface_weight if classification_enabled else 0.0,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        label_smoothing=0.1
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Extended ArcFace warmup schedule
    # 0–5: margin=0.0, scale=16; 6–10: margin=0.2, scale=20; >=11: margin=0.4, scale=28
    def apply_arcface_warmup(current_epoch: int):
        if current_epoch <= 5:
            model.arcface_head.set_params(scale=16.0, margin=0.0)
        elif current_epoch <= 10:
            model.arcface_head.set_params(scale=20.0, margin=0.2)
        else:
            model.arcface_head.set_params(scale=28.0, margin=0.4)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 恢复/初始化权重
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_acc = ckpt.get('best_metric', 0.0)
        start_epoch = ckpt.get('epoch', -1) + 1
        print(f'Resumed at epoch {start_epoch}, best_acc={best_acc:.6f}')
    elif args.init_model and os.path.isfile(args.init_model):
        print(f'Initializing model weights from: {args.init_model}')
        ckpt = torch.load(args.init_model, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        start_epoch = 0
        best_acc = 0.0
    
    # 训练循环
    print('Starting training...')
    
    # Freeze backbone for first few epochs and ramp triplet weight
    freeze_backbone_epochs = 5
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'\nEpoch {epoch+1}/{start_epoch + args.epochs}')
        apply_arcface_warmup(epoch)
        if epoch < freeze_backbone_epochs:
            for p in model.backbone.features.parameters():
                p.requires_grad = False
        elif epoch == freeze_backbone_epochs:
            for p in model.backbone.features.parameters():
                p.requires_grad = True
        
        # Triplet weight schedule
        if epoch < 5:
            criterion.triplet_weight = 0.1
        elif epoch < 10:
            criterion.triplet_weight = 0.3
        else:
            criterion.triplet_weight = args.triplet_weight
        
        # 训练
        train_loss, train_arcface, train_triplet, train_top1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, classification_enabled=classification_enabled
        )
        
        # 验证
        val_loss, val_arcface, val_triplet, val_top1, val_top5 = validate_epoch(
            model, val_loader, criterion, device, classification_enabled=classification_enabled
        )
        
        # 学习率调度（Triplet-only 时使用 val_triplet 近似替代）
        scheduler_metric = val_top1 if classification_enabled else (-val_triplet)
        scheduler.step(scheduler_metric)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('ArcFace/Train', train_arcface, epoch)
        writer.add_scalar('ArcFace/Val', val_arcface, epoch)
        writer.add_scalar('Triplet/Train', train_triplet, epoch)
        writer.add_scalar('Triplet/Val', val_triplet, epoch)
        if classification_enabled:
            writer.add_scalar('AccuracyTop1/Train', train_top1, epoch)
            writer.add_scalar('AccuracyTop1/Val', val_top1, epoch)
            writer.add_scalar('AccuracyTop5/Train', train_top5, epoch)
            writer.add_scalar('AccuracyTop5/Val', val_top5, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train - Loss: {train_loss:.4f}, ArcFace: {train_arcface:.4f}, Triplet: {train_triplet:.4f}, Top1: {(train_top1 if classification_enabled else 0.0):.6f}, Top5: {(train_top5 if classification_enabled else 0.0):.6f}')
        print(f'Val   - Loss: {val_loss:.4f}, ArcFace: {val_arcface:.4f}, Triplet: {val_triplet:.4f}, Top1: {(val_top1 if classification_enabled else 0.0):.6f}, Top5: {(val_top5 if classification_enabled else 0.0):.6f}')
        
        # 保存最佳模型（Triplet-only 模式用最低验证Triplet损失作为指标）
        if classification_enabled:
            is_better = val_top1 > best_acc
            metric_to_save = val_top1
        else:
            # Lower triplet loss is better
            is_better = (epoch == start_epoch) or (val_triplet < best_acc) or (best_acc == 0.0)
            metric_to_save = -val_triplet
        if is_better:
            best_acc = metric_to_save
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print('New best model saved.')

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
    
    print(f'\nTraining completed! Best Top1 Acc: {best_acc:.6f}')
    writer.close()


if __name__ == '__main__':
    main()
