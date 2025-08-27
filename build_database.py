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


def extract_embedding(model, image_tensor, mask=None, img_size=256, device='cpu'):
    """提取特征嵌入"""
    import torch.nn.functional as F
    
    # 确保图像张量在正确的设备上
    image_tensor = image_tensor.to(device)
    
    # 调整图像大小用于识别
    if image_tensor.shape[-1] != img_size:
        image_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    # 如果有掩码，应用掩码
    if mask is not None:
        # 调整掩码大小
        mask_resized = cv2.resize(mask, (img_size, img_size))
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # 应用掩码
        image_tensor = image_tensor * mask_tensor
    
    with torch.no_grad():
        # 直接调用模型获取嵌入，不传入labels参数
        embedding = model(image_tensor, labels=None)
    
    return embedding.cpu().numpy()[0]


def list_images_recursive(root: str):
    paths = []
    for r, _, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if low.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                paths.append(os.path.join(r, fn))
    return sorted(paths)


def parse_eval_filename(path: str):
    """Parse filename like name + lowercase l/r + image id, e.g., alicel03.jpg -> (alice, 'L')."""
    import re
    base = os.path.basename(path)
    m = re.match(r"^(.+?)([lr])(\d+)\.[^\.]+$", base)
    if not m:
        return None, None
    person = m.group(1)
    eye = m.group(2).upper()
    return person, eye


def main():
    parser = argparse.ArgumentParser(description='Build Iris Database')
    parser.add_argument('--data_root', type=str, help='Data root directory')
    parser.add_argument('--images_dir', type=str, help='Images directory only (e.g., data/eval/images)')
    parser.add_argument('--seg_model', type=str, required=True, help='Segmentation model path')
    parser.add_argument('--recog_model', type=str, required=True, help='Recognition model path')
    parser.add_argument('--person_ids', type=str, help='Person IDs JSON file (optional)')
    parser.add_argument('--output_path', type=str, required=True, help='Database output path')
    parser.add_argument('--templates_per_id', type=int, default=1, help='Number of templates per person-eye (use >1 to enable clustering)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
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
    
    # 确定类别数量
    if args.person_ids and os.path.exists(args.person_ids):
        with open(args.person_ids, 'r') as f:
            person_ids = json.load(f)
        num_classes = len(person_ids)
        print(f'Using {num_classes} classes from person_ids.json')
    else:
        # 如果没有提供person_ids，使用默认值
        num_classes = 2000
        print(f'Using default {num_classes} classes')
    
    recog_model = build_recognition_model(
        num_classes=num_classes,
        embedding_size=512,
        pretrained=False
    )
    checkpoint = torch.load(args.recog_model, map_location=device)
    recog_model.load_state_dict(checkpoint['model_state_dict'])
    recog_model.to(device)
    recog_model.eval()
    
    # 收集数据
    print('Collecting data...')
    items = []  # tuples (img_path, person_id, eye_type)
    if args.images_dir and os.path.isdir(args.images_dir):
        img_files = list_images_recursive(args.images_dir)
        for ip in img_files:
            person, eye = parse_eval_filename(ip)
            if person is None or eye is None:
                continue
            items.append((ip, person, eye))
        print(f'Found {len(items)} images in images_dir')
    else:
        if not args.data_root:
            raise ValueError('Provide --images_dir for eval-only images or --data_root for standard (images+masks) dataset')
        paired_imgs, paired_masks, paired_info = collect_pairs(args.data_root)
        for ip, mp, info in zip(paired_imgs, paired_masks, paired_info):
            if info['person_id'] is None or info.get('eye_type') is None:
                continue
            items.append((ip, info['person_id'], info.get('eye_type')))
        print(f'Found {len(items)} image-mask pairs in data_root')
    
    # 构建数据库
    print('Building database...')
    database = {}
    
    for i, (img_path, person_id, eye_type) in enumerate(tqdm(items)):
        
        try:
            # 预处理图像
            img_tensor = preprocess_image(img_path, img_size=256)
            
            # 分割虹膜（若 eval-only 没有掩码，也统一用分割模型预测）
            mask = segment_iris(seg_model, img_tensor, device)
            
            # 提取嵌入（一致使用 256 尺寸）
            embedding = extract_embedding(recog_model, img_tensor, mask, img_size=256, device=device)
            
            # 存储到数据库（根据人员-眼睛区分）
            combined_id = f"{person_id}_{eye_type}"
            if combined_id not in database:
                database[combined_id] = []
            database[combined_id].append(embedding)
            
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            continue
    
    # 计算每个人的模板：稳健平均或多模板（K-means）
    print('Computing templates per person-eye...')
    K = max(1, int(args.templates_per_id))
    for person_id in list(database.keys()):
        embeddings = np.asarray(database[person_id], dtype=np.float32)  # (N, D)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            database[person_id] = []
            continue
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        emb_norm = embeddings / norms
        # Outlier removal via median prototype
        med = np.median(emb_norm, axis=0)
        med_norm = med / (np.linalg.norm(med) + 1e-12)
        sims = emb_norm @ med_norm
        med_s = float(np.median(sims))
        std_s = float(np.std(sims))
        inlier_mask = np.ones_like(sims, dtype=bool) if std_s == 0.0 else (np.abs(sims - med_s) <= (2.0 * std_s))
        X = emb_norm[inlier_mask]
        if X.shape[0] == 0:
            X = emb_norm
        if K == 1 or X.shape[0] < 2:
            # single robust mean
            center = X.mean(axis=0)
            center = center / (np.linalg.norm(center) + 1e-12)
            database[person_id] = [center.astype(np.float32)]
        else:
            # K-means clustering into K centers
            try:
                from sklearn.cluster import KMeans
                k_eff = min(K, X.shape[0])
                km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
                labels = km.fit_predict(X)
                centers = []
                for c in range(k_eff):
                    cluster = X[labels == c]
                    if cluster.shape[0] == 0:
                        continue
                    ctr = cluster.mean(axis=0)
                    ctr = ctr / (np.linalg.norm(ctr) + 1e-12)
                    centers.append(ctr.astype(np.float32))
                if not centers:
                    ctr = X.mean(axis=0)
                    ctr = ctr / (np.linalg.norm(ctr) + 1e-12)
                    centers = [ctr.astype(np.float32)]
                database[person_id] = centers
            except Exception:
                # fallback to single center
                center = X.mean(axis=0)
                center = center / (np.linalg.norm(center) + 1e-12)
                database[person_id] = [center.astype(np.float32)]
    
    # 保存数据库
    print(f'Saving database to {args.output_path}...')
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f'Database built successfully!')
    print(f'Total persons: {len(database)}')
    print(f'Database saved to: {args.output_path}')


if __name__ == '__main__':
    main()
