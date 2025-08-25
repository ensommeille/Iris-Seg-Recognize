# dataset.py - supports multiple dataset roots and stronger augmentations using albumentations
import os, random
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset
import albumentations as A

SUPPORTED = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

def list_images(root):
    files = []
    for ext in SUPPORTED:
        files += glob(os.path.join(root, f'*{ext}'))
    files = sorted(files)
    return files

def collect_pairs(root):
    img_dirs = [os.path.join(root, 'train','images'), os.path.join(root,'images')]
    mask_dirs = [os.path.join(root, 'train','masks'), os.path.join(root,'masks')]
    imgs = []
    masks = []
    for d in img_dirs:
        if os.path.isdir(d):
            imgs = list_images(d); break
    for d in mask_dirs:
        if os.path.isdir(d):
            masks = list_images(d); break
    if len(imgs)==0:
        raise FileNotFoundError(f'No image folder in {root}')
    if len(masks)==0:
        raise FileNotFoundError(f'No mask folder in {root}')
    
    # Create a mapping from base name (without extension) to file paths
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
    
    # Find matching pairs based on base names
    paired_imgs = []
    paired_masks = []
    for name, ip in img_map.items():
        if name in mask_map:
            paired_imgs.append(ip)
            paired_masks.append(mask_map[name])
    
    return paired_imgs, paired_masks

class IrisDataset(Dataset):
    def __init__(self, images, masks, img_size=256, augment=False):
        self.images = images; self.masks = masks
        self.img_size = img_size; self.augment = augment
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
            self.transform = A.Compose([A.Resize(img_size,img_size), A.Normalize()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Cannot read {img_path}')
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f'Cannot read {mask_path}')
        if mask.ndim==3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype('uint8')*255
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        img = img.transpose(2,0,1).astype('float32')
        mask = mask[np.newaxis,:,:].astype('float32')/255.0
        return torch.from_numpy(img), torch.from_numpy(mask)

def build_combined_dataset(roots, img_size=256, augment=True, val_split=0.05, seed=42):
    all_datasets = []
    for r in roots:
        imgs, masks = collect_pairs(r)
        ds = IrisDataset(imgs, masks, img_size=img_size, augment=augment)
        all_datasets.append(ds)
    combined = ConcatDataset(all_datasets)
    total = len(combined)
    val_n = max(1, int(total*val_split))
    train_n = total - val_n
    indices = list(range(total))
    random.seed(seed); random.shuffle(indices)
    train_idx = indices[:train_n]; val_idx = indices[train_n:]
    from torch.utils.data import Subset
    train_ds = Subset(combined, train_idx)
    val_ds = Subset(combined, val_idx)
    return train_ds, val_ds, all_datasets
