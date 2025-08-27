#!/usr/bin/env python3
"""
虹膜识别推理脚本
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import time
from tqdm import tqdm
import random
import cv2 as _cv2

from models.segmentation import build_segmentation_model
from models.recognition import build_recognition_model
from utils.dataset import normalize_iris


class IrisInference:
    def __init__(self, seg_model_path, recog_model_path, device='auto', normalize=True, norm_H=64, norm_W=256):
        self.normalize = normalize
        self.norm_H = norm_H
        self.norm_W = norm_W
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载分割模型
        self.seg_model = build_segmentation_model(num_classes=1, pretrained=False)
        checkpoint = torch.load(seg_model_path, map_location=self.device)
        self.seg_model.load_state_dict(checkpoint['model_state_dict'])
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        # 加载识别模型（从权重推断类别数）
        checkpoint = torch.load(recog_model_path, map_location=self.device)
        state = checkpoint['model_state_dict']
        num_classes = None
        for k, v in state.items():
            if k.endswith('arcface_head.weight'):
                num_classes = v.shape[0]
                break
        if num_classes is None:
            raise RuntimeError('Cannot infer num_classes from recognition checkpoint')
        self.recog_model = build_recognition_model(
            num_classes=num_classes,
            embedding_size=512,
            pretrained=False
        )
        self.recog_model.load_state_dict(state)
        self.recog_model.to(self.device)
        self.recog_model.eval()
        self.database = {}
        
        # 最近一次分割的掩码（原图分辨率），用于可视化
        self._last_mask_orig = None

    def save_segmentation(self, image_path, mask_bin, out_mask_path=None, out_overlay_path=None, alpha=0.5):
        try:
            img_bgr = _cv2.imread(image_path)
            if img_bgr is None:
                return
            h, w = img_bgr.shape[:2]
            mask_uint8 = (mask_bin * 255.0).astype('uint8') if mask_bin.max() <= 1.0 else mask_bin.astype('uint8')
            mask_resized = _cv2.resize(mask_uint8, (w, h), interpolation=_cv2.INTER_NEAREST)
            if out_mask_path:
                _cv2.imwrite(out_mask_path, mask_resized)
            if out_overlay_path:
                overlay = img_bgr.copy()
                # 创建绿色掩码
                green_mask = np.zeros_like(img_bgr)
                green_mask[:, :] = (0, 255, 0)  # BGR格式的绿色
                # 在掩码区域应用绿色覆盖
                mask_3ch = np.stack([mask_resized, mask_resized, mask_resized], axis=2) > 127
                overlay[mask_3ch] = _cv2.addWeighted(img_bgr, 1 - alpha, green_mask, alpha, 0)[mask_3ch]
                _cv2.imwrite(out_overlay_path, overlay)
        except Exception as e:
            print(f"Error in save_segmentation: {e}")
            pass
    
    def preprocess_image(self, image_path, img_size=256):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.normalize:
            # 在缩放图上执行分割得到mask，再将mask放缩回原图尺寸后做归一化展开
            img_for_seg = cv2.resize(img, (img_size, img_size))
            img_norm = img_for_seg.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
            mask_small = self.segment_iris(img_tensor)
            # 将mask映射回原图尺寸，并缓存用于可视化
            mask_orig = (cv2.resize((mask_small * 255).astype('uint8'), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127).astype('uint8')
            self._last_mask_orig = mask_orig
            # 归一化为伪极坐标图
            norm_img = normalize_iris(img, mask_orig, H=self.norm_H, W=self.norm_W)
            if norm_img.ndim == 2:
                norm_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
            img_normalized = norm_img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
            return img_tensor
        else:
            img_resized = cv2.resize(img, (img_size, img_size))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
            return img_tensor
    
    def segment_iris(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            mask_logits = self.seg_model(image_tensor)
            mask = torch.sigmoid(mask_logits)
            mask = (mask > 0.5).float()
        
        return mask.cpu().numpy()[0, 0]
    
    def extract_embedding(self, image_tensor, mask=None, img_size=256):
        # 当preprocess_image已执行归一化时，直接送入模型
        if not self.normalize and (mask is not None):
            mask_resized = cv2.resize(mask, (img_size, img_size))
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor * mask_tensor
        
        if image_tensor.shape[-1] != img_size:
            image_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            embedding = self.recog_model(image_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def query(self, image_path, top_k=5, threshold=0.5, save_masks_dir=None):
        img_tensor = self.preprocess_image(image_path, img_size=256)
        mask = None
        if not self.normalize:
            mask = self.segment_iris(img_tensor)
        embedding = self.extract_embedding(img_tensor, mask, img_size=(self.norm_W if self.normalize else 256))
        
        # 可选地保存掩码与叠加图
        if save_masks_dir:
            os.makedirs(save_masks_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            if self.normalize and self._last_mask_orig is not None:
                self.save_segmentation(
                    image_path,
                    self._last_mask_orig,
                    out_mask_path=os.path.join(save_masks_dir, f"{base}_mask.png"),
                    out_overlay_path=os.path.join(save_masks_dir, f"{base}_overlay.jpg"),
                    alpha=0.5,
                )
            elif mask is not None:
                self.save_segmentation(
                    image_path,
                    mask,
                    out_mask_path=os.path.join(save_masks_dir, f"{base}_mask.png"),
                    out_overlay_path=os.path.join(save_masks_dir, f"{base}_overlay.jpg"),
                    alpha=0.5,
                )
        
        similarities = []
        for person_id, db_value in self.database.items():
            # db_value can be a single vector or a list of template vectors
            if isinstance(db_value, list):
                sims = [cosine_similarity([embedding], [tpl])[0,0] for tpl in db_value]
                similarity = float(max(sims)) if sims else -1.0
            else:
                similarity = float(cosine_similarity([embedding], [db_value])[0, 0])
            similarities.append((person_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for person_id, similarity in similarities[:top_k]:
            results.append({
                'person_id': person_id,
                'similarity': float(similarity),
                'matched': similarity >= threshold
            })
        
        return results

    def evaluate_directory(self, image_dir, top_k=5, threshold=0.5, results_csv=None, sample_n=100, seed=42, save_masks_dir=None):
        """Batch query over a directory and compute metrics.
        Assumes filenames like S5999L00.jpg, ground-truth combined id = S5999_L.
        """
        # Accept lowercase/uppercase L/R and any extension
        pattern = re.compile(r"^(.+?)([LRlr])(\d+)\.[^.]+$")
        image_files = []
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):  # simple gather
            # os.listdir then filter to avoid glob on Windows non-ASCII issues
            pass
        # Use os.walk to be robust
        for root, _, files in os.walk(image_dir):
            for fname in files:
                low = fname.lower()
                if low.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, fname))

        # Optional random sampling
        random.seed(seed)
        if len(image_files) > sample_n:
            image_files = random.sample(image_files, sample_n)
        n = len(image_files)
        if n == 0:
            print(f'No images found in {image_dir}')
            return {
                'num_images': 0,
                'top1_acc': 0.0,
                'top5_acc': 0.0,
                'avg_top1_similarity': 0.0,
                'avg_gt_similarity': 0.0,
            }

        correct_top1 = 0
        correct_top5 = 0
        sum_top1_sim = 0.0
        sum_gt_sim = 0.0

        rows = []

        start_time = time.time()
        for image_path in tqdm(image_files, desc='Evaluating'):
            fname = os.path.basename(image_path)
            m = pattern.match(fname)
            gt_id = None
            if m:
                person_id = m.group(1)
                eye = m.group(2).upper()
                gt_id = f"{person_id}_{eye}"

            # One forward path to get embedding; compute similarities once
            img_tensor = self.preprocess_image(image_path, img_size=256)
            mask = None
            if not self.normalize:
                mask = self.segment_iris(img_tensor)
            if save_masks_dir:
                os.makedirs(save_masks_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(image_path))[0]
                if self.normalize and self._last_mask_orig is not None:
                    self.save_segmentation(
                        image_path,
                        self._last_mask_orig,
                        out_mask_path=os.path.join(save_masks_dir, f"{base}_mask.png"),
                        out_overlay_path=os.path.join(save_masks_dir, f"{base}_overlay.jpg"),
                        alpha=0.5,
                    )
                elif mask is not None:
                    self.save_segmentation(
                        image_path,
                        mask,
                        out_mask_path=os.path.join(save_masks_dir, f"{base}_mask.png"),
                        out_overlay_path=os.path.join(save_masks_dir, f"{base}_overlay.jpg"),
                        alpha=0.5,
                    )
            embedding = self.extract_embedding(img_tensor, mask, img_size=(self.norm_W if self.normalize else 256))

            similarities = []
            for pid_db, db_value in self.database.items():
                if isinstance(db_value, list):
                    sims = [cosine_similarity([embedding], [tpl])[0,0] for tpl in db_value]
                    simv = float(max(sims)) if sims else -1.0
                else:
                    simv = float(cosine_similarity([embedding], [db_value])[0,0])
                similarities.append((pid_db, simv))
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = [{'person_id': p, 'similarity': s, 'matched': (s >= threshold)} for p, s in similarities[:top_k]]
            top1 = results[0] if results else {'person_id': None, 'similarity': 0.0}

            # compute gt similarity from same embedding
            if gt_id is not None and gt_id in self.database:
                db_value = self.database[gt_id]
                if isinstance(db_value, list):
                    sims = [cosine_similarity([embedding], [tpl])[0,0] for tpl in db_value]
                    gt_sim = float(max(sims)) if sims else float('nan')
                else:
                    gt_sim = float(cosine_similarity([embedding], [db_value])[0,0])
            else:
                gt_sim = float('nan')

            is_top1 = (gt_id is not None and top1['person_id'] == gt_id)
            is_top5 = False
            if gt_id is not None:
                for r in results:
                    if r['person_id'] == gt_id:
                        is_top5 = True
                        break

            correct_top1 += 1 if is_top1 else 0
            correct_top5 += 1 if is_top5 else 0
            sum_top1_sim += float(top1['similarity']) if results else 0.0
            sum_gt_sim += gt_sim if gt_sim == gt_sim else 0.0  # skip NaN

            rows.append({
                'image': image_path,
                'gt_id': gt_id if gt_id is not None else '',
                'top1_id': top1['person_id'] if results else '',
                'top1_similarity': float(top1['similarity']) if results else 0.0,
                'correct_top1': int(is_top1),
                'correct_top5': int(is_top5),
                'gt_similarity': gt_sim if gt_sim == gt_sim else ''
            })

        elapsed = time.time() - start_time if time.time() - start_time > 0 else 1e-12
        fps = n / elapsed
        top1_acc = correct_top1 / n
        top5_acc = correct_top5 / n
        avg_top1_similarity = sum_top1_sim / n
        # compute avg over available gt sims
        valid_gt_sims = [r['gt_similarity'] for r in rows if isinstance(r['gt_similarity'], float)]
        avg_gt_similarity = float(sum(valid_gt_sims) / len(valid_gt_sims)) if valid_gt_sims else 0.0

        if results_csv:
            os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
            with open(results_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        summary = {
            'num_images': n,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'avg_top1_similarity': avg_top1_similarity,
            'avg_gt_similarity': avg_gt_similarity,
            'elapsed_sec': elapsed,
            'fps': fps,
        }
        print('Evaluation summary:')
        print(summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description='Iris Recognition Inference')
    parser.add_argument('--seg_model', type=str, required=True)
    parser.add_argument('--recog_model', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--query_image', type=str)
    parser.add_argument('--query_dir', type=str)
    parser.add_argument('--results_csv', type=str)
    parser.add_argument('--sample_n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_masks_dir', type=str, help='Directory to save predicted masks and overlays (optional)')
    # 归一化开关与尺寸
    parser.add_argument('--normalize_iris', action='store_true', default=True, help='Normalize iris to pseudo-polar before embedding')
    parser.add_argument('--norm_H', type=int, default=64, help='Iris normalization height (radial samples)')
    parser.add_argument('--norm_W', type=int, default=256, help='Iris normalization width (angular samples)')
    
    args = parser.parse_args()
    
    inference = IrisInference(
        seg_model_path=args.seg_model,
        recog_model_path=args.recog_model,
        normalize=args.normalize_iris,
        norm_H=args.norm_H,
        norm_W=args.norm_W,
    )
    
    # 加载数据库
    with open(args.database_path, 'rb') as f:
        inference.database = pickle.load(f)
    
    # 查询
    if args.query_image:
        results = inference.query(args.query_image, args.top_k, args.threshold, save_masks_dir=args.save_masks_dir)
        print('\nQuery Results:')
        for i, result in enumerate(results):
            status = '✓' if result['matched'] else '✗'
            print(f'{i+1}. {status} {result["person_id"]}: {result["similarity"]:.4f}')
    elif args.query_dir:
        _ = inference.evaluate_directory(
            args.query_dir,
            top_k=args.top_k,
            threshold=args.threshold,
            results_csv=args.results_csv,
            sample_n=args.sample_n,
            seed=args.seed,
            save_masks_dir=args.save_masks_dir,
        )
    else:
        print('Please provide --query_image or --query_dir')


if __name__ == '__main__':
    main()
