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
    def __init__(self, images, masks, info, img_size=256, augment=False, use_mask=True, person_to_label=None,
                 normalize=False, norm_H=64, norm_W=256):
        self.images = images
        self.masks = masks
        self.info = info
        self.img_size = img_size
        self.augment = augment
        self.use_mask = use_mask
        self.normalize = normalize
        self.norm_H = norm_H
        self.norm_W = norm_W
        
        # 如果提供了全局的 person_to_label，则直接使用；否则根据本数据构建
        if person_to_label is not None:
            self.person_to_label = dict(person_to_label)
            self.person_ids = list(self.person_to_label.keys())
        else:
            self.person_to_indices = defaultdict(list)
            for i, info_item in enumerate(info):
                person_id = info_item['person_id']
                eye_type = info_item.get('eye_type')
                if person_id is not None and eye_type is not None:
                    combined_id = f"{person_id}_{eye_type}"
                    self.person_to_indices[combined_id].append(i)
            self.person_ids = list(self.person_to_indices.keys())
            self.person_to_label = {pid: idx for idx, pid in enumerate(self.person_ids)}
        
        if self.normalize:
            # 归一化后尺寸固定为(norm_H, norm_W)，不做空间增强，只做像素归一化
            self.transform = A.Compose([
                A.Normalize(),
            ])
        else:
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
        
        if self.normalize:
            # 归一化流程需要mask
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f'Cannot read mask: {mask_path}')
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 127).astype('uint8')
            img = normalize_iris(img, mask, H=self.norm_H, W=self.norm_W)
            # 确保为RGB三通道
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
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
        
        # 数据增强或归一化
        augmented = self.transform(image=img)
        img = augmented['image']
        
        # 如果是归一化路径，保证尺寸为(norm_H, norm_W)
        if self.normalize and (img.shape[0] != self.norm_H or img.shape[1] != self.norm_W):
            img = cv2.resize(img, (self.norm_W, self.norm_H))
        
        # 转换为tensor
        img = img.transpose(2, 0, 1).astype('float32')
        
        # 获取标签（基于人员-眼睛组合）
        person_id = info_item['person_id']
        eye_type = info_item.get('eye_type')
        combined_id = f"{person_id}_{eye_type}" if (person_id is not None and eye_type is not None) else None
        if combined_id in self.person_to_label:
            label = self.person_to_label[combined_id]
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


def build_recognition_datasets(data_root, img_size=256, val_split=0.1, seed=42, use_mask=True,
                               normalize=False, norm_H=64, norm_W=256):
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
    
    # 统一构建全局的 person-eye -> label 映射，确保 train/val 一致
    combined_ids_all = []
    for pid, indices in valid_persons.items():
        # 收集此人的左右眼组合ID
        eyes = set()
        for idx in indices:
            eye = paired_info[idx].get('eye_type')
            if eye is not None:
                eyes.add(eye)
        for eye in sorted(list(eyes)):
            combined_ids_all.append(f"{pid}_{eye}")
    combined_ids_all = sorted(list(set(combined_ids_all)))
    global_person_to_label = {cid: i for i, cid in enumerate(combined_ids_all)}
    
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
    train_dataset = IrisRecognitionDataset(train_imgs, train_masks, train_info, img_size, augment=True, use_mask=use_mask,
                                           person_to_label=global_person_to_label, normalize=normalize, norm_H=norm_H, norm_W=norm_W)
    
    # 构建验证集
    val_imgs = [paired_imgs[i] for i in val_indices]
    val_masks = [paired_masks[i] for i in val_indices]
    val_info = [paired_info[i] for i in val_indices]
    val_dataset = IrisRecognitionDataset(val_imgs, val_masks, val_info, img_size, augment=False, use_mask=use_mask,
                                         person_to_label=global_person_to_label, normalize=normalize, norm_H=norm_H, norm_W=norm_W)
    
    return train_dataset, val_dataset


# ---- Iris normalization helpers ----

def fit_circle(mask):
    """拟合圆形边界，返回圆心和半径"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 用最小外接圆近似
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(x), int(y)), int(radius)


def normalize_iris(img, mask, H=64, W=256):
    """
    虹膜归一化：将笛卡尔坐标的虹膜区域映射到伪极坐标系统 (H x W)
    
    Args:
        img: RGB图像 (H, W, 3)
        mask: 二值掩码 (H, W) 
        H: 目标高度（径向采样点数）
        W: 目标宽度（角度采样点数）
    
    Returns:
        normalized_iris: 归一化后的虹膜图像 (H, W, 3)
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:,:,0]
    else:
        img_gray = img
    
    # 拟合虹膜的内外边界
    # 假设mask是虹膜区域，我们需要估计内外边界
    # 这里简化处理：外边界为mask的轮廓，内边界为mask的腐蚀版本
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=3)
    
    # 拟合外边界圆
    outer_center, outer_radius = fit_circle(mask)
    if outer_center is None:
        # 如果拟合失败，返回原图缩放版本
        return cv2.resize(img, (W, H))
    
    # 拟合内边界圆
    inner_center, inner_radius = fit_circle(inner_mask)
    if inner_center is None:
        inner_center = outer_center
        inner_radius = max(1, outer_radius // 3)  # 默认内半径为外半径的1/3
    
    # 创建归一化图像
    normalized_iris = np.zeros((H, W, 3 if img.ndim == 3 else 1), dtype=np.uint8)
    
    # 角度和径向采样
    theta_step = 2 * np.pi / W
    radius_step = (outer_radius - inner_radius) / H
    
    for i in range(H):  # 径向方向
        for j in range(W):  # 角度方向
            # 计算伪极坐标
            r = inner_radius + i * radius_step
            theta = j * theta_step
            
            # 转换为笛卡尔坐标
            x = int(outer_center[0] + r * np.cos(theta))
            y = int(outer_center[1] + r * np.sin(theta))
            
            # 边界检查
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                if img.ndim == 3:
                    normalized_iris[i, j, :] = img[y, x, :]
                else:
                    normalized_iris[i, j, 0] = img[y, x]
    
    # 如果原图是单通道，返回单通道结果
    if img.ndim == 2:
        normalized_iris = normalized_iris[:, :, 0]
    
    return normalized_iris


def build_recognition_datasets_eye_split(data_root, seed=42, use_mask=True,
                                         normalize=True, norm_H=64, norm_W=256):
    """
    基于眼别的识别数据集划分：
    - 左眼(L)：按80/20划分为train/val
    - 右眼(R)：全部作为test
    返回：train_dataset, val_dataset, test_dataset
    """
    paired_imgs, paired_masks, paired_info = collect_pairs(data_root)

    # 过滤左右眼索引
    left_indices = [i for i, info in enumerate(paired_info) if info.get('eye_type') == 'L']
    right_indices = [i for i, info in enumerate(paired_info) if info.get('eye_type') == 'R']

    # 左眼按人员分组再80/20
    person_to_left = defaultdict(list)
    for i in left_indices:
        pid = paired_info[i]['person_id']
        if pid is not None:
            person_to_left[pid].append(i)

    # 构建全局label映射，仅针对左/右分别使用 person_eye 作为类，便于Triplet采样
    combined_ids_all = []
    for pid, idxs in person_to_left.items():
        if len(idxs) >= 2:
            combined_ids_all.append(f"{pid}_L")
    # 右眼测试也保留映射，便于需要标签的评估
    right_pids = sorted({paired_info[i]['person_id'] for i in right_indices if paired_info[i]['person_id'] is not None})
    for pid in right_pids:
        combined_ids_all.append(f"{pid}_R")
    combined_ids_all = sorted(list(set(combined_ids_all)))
    global_person_to_label = {cid: i for i, cid in enumerate(combined_ids_all)}

    # 左眼train/val划分
    random.seed(seed)
    train_left, val_left = [], []
    for pid, idxs in person_to_left.items():
        if len(idxs) < 2:
            continue
        random.shuffle(idxs)
        val_size = max(1, int(len(idxs) * 0.2))
        val_left.extend(idxs[:val_size])
        train_left.extend(idxs[val_size:])

    # 构建数据集
    def make_dataset(indices, augment):
        imgs = [paired_imgs[i] for i in indices]
        masks = [paired_masks[i] for i in indices]
        infos = [paired_info[i] for i in indices]
        return IrisRecognitionDataset(imgs, masks, infos, img_size=norm_H, augment=augment, use_mask=use_mask,
                                      person_to_label=global_person_to_label, normalize=normalize, norm_H=norm_H, norm_W=norm_W)

    train_dataset = make_dataset(train_left, augment=True)
    val_dataset = make_dataset(val_left, augment=False)
    test_dataset = make_dataset(right_indices, augment=False)

    return train_dataset, val_dataset, test_dataset
