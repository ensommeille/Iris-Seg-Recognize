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

from models.segmentation import build_segmentation_model
from models.recognition import build_recognition_model


class IrisInference:
    def __init__(self, seg_model_path, recog_model_path, device='auto'):
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
    
    def preprocess_image(self, image_path, img_size=256):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        if mask is not None:
            mask_resized = cv2.resize(mask, (img_size, img_size))
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor * mask_tensor
        
        if image_tensor.shape[-1] != img_size:
            image_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            embedding = self.recog_model(image_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def query(self, image_path, top_k=5, threshold=0.5):
        img_tensor = self.preprocess_image(image_path, img_size=256)
        mask = self.segment_iris(img_tensor)
        embedding = self.extract_embedding(img_tensor, mask, img_size=256)
        
        similarities = []
        for person_id, db_embedding in self.database.items():
            similarity = cosine_similarity([embedding], [db_embedding])[0, 0]
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

    def evaluate_directory(self, image_dir, top_k=5, threshold=0.5, results_csv=None, sample_n=100, seed=42):
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

            results = self.query(image_path, top_k=top_k, threshold=threshold)
            # results is list of dicts [{'person_id': id, 'similarity': sim, 'matched': bool}, ...]
            top1 = results[0] if results else {'person_id': None, 'similarity': 0.0}

            # compute gt similarity directly if gt in database
            if gt_id is not None and gt_id in self.database:
                # reuse embedding computed within query by recomputing once
                # To avoid recomputing embedding twice, we redo minimal steps here
                img_tensor = self.preprocess_image(image_path, img_size=256)
                mask = self.segment_iris(img_tensor)
                embedding = self.extract_embedding(img_tensor, mask, img_size=256)
                gt_sim = float(cosine_similarity([embedding], [self.database[gt_id]])[0,0])
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
    
    args = parser.parse_args()
    
    inference = IrisInference(
        seg_model_path=args.seg_model,
        recog_model_path=args.recog_model
    )
    
    # 加载数据库
    with open(args.database_path, 'rb') as f:
        inference.database = pickle.load(f)
    
    # 查询
    if args.query_image:
        results = inference.query(args.query_image, args.top_k, args.threshold)
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
        )
    else:
        print('Please provide --query_image or --query_dir')


if __name__ == '__main__':
    main()
