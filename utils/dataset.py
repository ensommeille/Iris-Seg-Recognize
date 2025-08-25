"""
数据集定义
支持虹膜分割和识别任务的数据加载
"""
import os
import re
import random
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from collections import defaultdict


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(root):
    """列出指定目录下的所有图像文件"""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files += glob(os.path.join(root, f'*{ext}'))
    return sorted(files)


def parse_filename(filename):
    """解析文件名获取人员ID、眼睛类型和编号"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    pattern = r'^(.+?)([LR])(\d+)$'
    match = re.match(pattern, basename)
    
    if match:
        person_id = match.group(1)
        eye_type = match.group(2)
        number = match.group(3)
        return person_id, eye_type, number
    else:
        return None, None, None


def collect_pairs(data_root):
    """收集图像和掩码的配对"""
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Image directory not found: {img_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')
    
    img_files = list_images(img_dir)
    mask_files = list_images(mask_dir)
    
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
    mask_map = {os.path.splitext(os.path.basename(p))[0].replace('_mask', ''): p for p in mask_files}
    
    paired_imgs = []
    paired_masks = []
    paired_info = []
    
    for name, img_path in img_map.items():
        if name in mask_map:
            paired_imgs.append(img_path)
            paired_masks.append(mask_map[name])
            
            person_id, eye_type, number = parse_filename(img_path)
            paired_info.append({
                'person_id': person_id,
                'eye_type': eye_type,
                'number': number,
                'filename': name
            })
    
    return paired_imgs, paired_masks, paired_info


class IrisSegmentationDataset(Dataset):
    """虹膜分割数据集"""
    def __init__(self, images, masks, img_size=256, augment=False):
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.Normalize(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size), 
                A.Normalize()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # 读取图像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Cannot read image: {img_path}')
        
        # 处理图像格式
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f'Cannot read mask: {mask_path}')
        
        # 处理掩码格式
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype('uint8') * 255
        
        # 数据增强
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        # 转换为tensor
        img = img.transpose(2, 0, 1).astype('float32')
        mask = mask[np.newaxis, :, :].astype('float32') / 255.0
        
        return torch.from_numpy(img), torch.from_numpy(mask)


class IrisRecognitionDataset(Dataset):
    """虹膜识别数据集"""
    def __init__(self, images, masks, info, img_size=224, augment=False, use_mask=True):
        self.images = images
        self.masks = masks
        self.info = info
        self.img_size = img_size
        self.augment = augment
        self.use_mask = use_mask
        
        # 创建人员ID到索引的映射
        self.person_to_indices = defaultdict(list)
        for i, info_item in enumerate(info):
            person_id = info_item['person_id']
            if person_id is not None:
                self.person_to_indices[person_id].append(i)
        
        # 获取所有人员ID
        self.person_ids = list(self.person_to_indices.keys())
        self.person_to_label = {pid: idx for idx, pid in enumerate(self.person_ids)}
        
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.1),
                A.Normalize(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size), 
                A.Normalize()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        info_item = self.info[idx]
        
        # 读取图像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Cannot read image: {img_path}')
        
        # 处理图像格式
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取掩码（如果使用）
        if self.use_mask:
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f'Cannot read mask: {mask_path}')
            
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 127).astype('uint8')
            
            # 应用掩码
            img = img * mask[:, :, np.newaxis]
        
        # 数据增强
        augmented = self.transform(image=img)
        img = augmented['image']
        
        # 转换为tensor
        img = img.transpose(2, 0, 1).astype('float32')
        
        # 获取标签
        person_id = info_item['person_id']
        if person_id in self.person_to_label:
            label = self.person_to_label[person_id]
        else:
            label = 0  # 默认标签
        
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        """获取类别数量"""
        return len(self.person_ids)
    
    def get_person_ids(self):
        """获取所有人员ID"""
        return self.person_ids


def build_segmentation_datasets(data_root, img_size=256, val_split=0.1, seed=42):
    """构建分割数据集"""
    paired_imgs, paired_masks, paired_info = collect_pairs(data_root)
    
    # 随机分割
    random.seed(seed)
    indices = list(range(len(paired_imgs)))
    random.shuffle(indices)
    
    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 构建训练集
    train_imgs = [paired_imgs[i] for i in train_indices]
    train_masks = [paired_masks[i] for i in train_indices]
    train_dataset = IrisSegmentationDataset(train_imgs, train_masks, img_size, augment=True)
    
    # 构建验证集
    val_imgs = [paired_imgs[i] for i in val_indices]
    val_masks = [paired_masks[i] for i in val_indices]
    val_dataset = IrisSegmentationDataset(val_imgs, val_masks, img_size, augment=False)
    
    return train_dataset, val_dataset


def build_recognition_datasets(data_root, img_size=224, val_split=0.1, seed=42, use_mask=True):
    """构建识别数据集"""
    paired_imgs, paired_masks, paired_info = collect_pairs(data_root)
    
    # 按人员ID分组
    person_to_indices = defaultdict(list)
    for i, info_item in enumerate(paired_info):
        person_id = info_item['person_id']
        if person_id is not None:
            person_to_indices[person_id].append(i)
    
    # 确保每个人员都有足够的样本
    valid_persons = {pid: indices for pid, indices in person_to_indices.items() if len(indices) >= 2}
    
    train_indices = []
    val_indices = []
    
    random.seed(seed)
    for person_id, indices in valid_persons.items():
        random.shuffle(indices)
        val_size = max(1, int(len(indices) * val_split))
        train_indices.extend(indices[val_size:])
        val_indices.extend(indices[:val_size])
    
    # 构建训练集
    train_imgs = [paired_imgs[i] for i in train_indices]
    train_masks = [paired_masks[i] for i in train_indices]
    train_info = [paired_info[i] for i in train_indices]
    train_dataset = IrisRecognitionDataset(train_imgs, train_masks, train_info, img_size, augment=True, use_mask=use_mask)
    
    # 构建验证集
    val_imgs = [paired_imgs[i] for i in val_indices]
    val_masks = [paired_masks[i] for i in val_indices]
    val_info = [paired_info[i] for i in val_indices]
    val_dataset = IrisRecognitionDataset(val_imgs, val_masks, val_info, img_size, augment=False, use_mask=use_mask)
    
    return train_dataset, val_dataset
