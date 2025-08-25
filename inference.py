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

from models.segmentation import build_segmentation_model
from models.recognition import build_recognition_model


class IrisInference:
    def __init__(self, seg_model_path, recog_model_path, person_ids_path, device='auto'):
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
        
        # 加载识别模型
        with open(person_ids_path, 'r') as f:
            person_ids = json.load(f)
        
        self.recog_model = build_recognition_model(
            num_classes=len(person_ids),
            embedding_size=512,
            pretrained=False
        )
        checkpoint = torch.load(recog_model_path, map_location=self.device)
        self.recog_model.load_state_dict(checkpoint['model_state_dict'])
        self.recog_model.to(self.device)
        self.recog_model.eval()
        
        self.person_ids = person_ids
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
    
    def extract_embedding(self, image_tensor, mask=None, img_size=224):
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
        embedding = self.extract_embedding(img_tensor, mask, img_size=224)
        
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


def main():
    parser = argparse.ArgumentParser(description='Iris Recognition Inference')
    parser.add_argument('--seg_model', type=str, required=True)
    parser.add_argument('--recog_model', type=str, required=True)
    parser.add_argument('--person_ids', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--query_image', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    inference = IrisInference(
        seg_model_path=args.seg_model,
        recog_model_path=args.recog_model,
        person_ids_path=args.person_ids
    )
    
    # 加载数据库
    with open(args.database_path, 'rb') as f:
        inference.database = pickle.load(f)
    
    # 查询
    results = inference.query(args.query_image, args.top_k, args.threshold)
    
    print('\nQuery Results:')
    for i, result in enumerate(results):
        status = '✓' if result['matched'] else '✗'
        print(f'{i+1}. {status} {result["person_id"]}: {result["similarity"]:.4f}')


if __name__ == '__main__':
    main()
