#!/usr/bin/env python3
"""
虹膜数据库构建脚本
"""
import os
import argparse
import torch
import numpy as np
import cv2
import json
import pickle
from tqdm import tqdm
from glob import glob

from models.segmentation import build_segmentation_model
from models.recognition import build_recognition_model
from utils.dataset import collect_pairs, parse_filename


def preprocess_image(image_path, img_size=256):
    """预处理图像"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f'Cannot read image: {image_path}')
    
    # 处理图像格式
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # 归一化
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return img_tensor


def segment_iris(model, image_tensor, device):
    """分割虹膜"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        mask_logits = model(image_tensor)
        mask = torch.sigmoid(mask_logits)
        mask = (mask > 0.5).float()
    
    return mask.cpu().numpy()[0, 0]


def extract_embedding(model, image_tensor, mask=None, img_size=224):
    """提取特征嵌入"""
    import torch.nn.functional as F
    
    # 如果有掩码，应用掩码
    if mask is not None:
        # 调整掩码大小
        mask_resized = cv2.resize(mask, (img_size, img_size))
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0)
        
        # 应用掩码
        image_tensor = image_tensor * mask_tensor
    
    # 调整图像大小用于识别
    if image_tensor.shape[-1] != img_size:
        image_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Build Iris Database')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--seg_model', type=str, required=True, help='Segmentation model path')
    parser.add_argument('--recog_model', type=str, required=True, help='Recognition model path')
    parser.add_argument('--person_ids', type=str, required=True, help='Person IDs JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Database output path')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # 加载分割模型
    print('Loading segmentation model...')
    seg_model = build_segmentation_model(num_classes=1, pretrained=False)
    checkpoint = torch.load(args.seg_model, map_location=device)
    seg_model.load_state_dict(checkpoint['model_state_dict'])
    seg_model.to(device)
    seg_model.eval()
    
    # 加载识别模型
    print('Loading recognition model...')
    with open(args.person_ids, 'r') as f:
        person_ids = json.load(f)
    
    recog_model = build_recognition_model(
        num_classes=len(person_ids),
        embedding_size=512,
        pretrained=False
    )
    checkpoint = torch.load(args.recog_model, map_location=device)
    recog_model.load_state_dict(checkpoint['model_state_dict'])
    recog_model.to(device)
    recog_model.eval()
    
    # 收集数据
    print('Collecting data...')
    paired_imgs, paired_masks, paired_info = collect_pairs(args.data_root)
    
    # 构建数据库
    print('Building database...')
    database = {}
    
    for i, (img_path, mask_path, info) in enumerate(tqdm(zip(paired_imgs, paired_masks, paired_info), total=len(paired_imgs))):
        person_id = info['person_id']
        if person_id is None:
            continue
        
        try:
            # 预处理图像
            img_tensor = preprocess_image(img_path, img_size=256)
            
            # 分割虹膜
            mask = segment_iris(seg_model, img_tensor, device)
            
            # 提取嵌入
            embedding = extract_embedding(recog_model, img_tensor, mask, img_size=224)
            
            # 存储到数据库
            if person_id not in database:
                database[person_id] = []
            database[person_id].append(embedding)
            
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            continue
    
    # 计算每个人的平均嵌入
    print('Computing average embeddings...')
    for person_id in database:
        embeddings = np.array(database[person_id])
        database[person_id] = np.mean(embeddings, axis=0)
    
    # 保存数据库
    print(f'Saving database to {args.output_path}...')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f'Database built successfully!')
    print(f'Total persons: {len(database)}')
    print(f'Database saved to: {args.output_path}')


if __name__ == '__main__':
    main()
