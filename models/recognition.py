"""
虹膜识别模型定义
MobileNetV3 backbone + ArcFace head + Triplet Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class ArcMarginProduct(nn.Module):
    """
    ArcFace margin product layer
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # normalize features and weights
        input = F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # cos(theta)
        cos_theta = F.linear(input, weight)
        cos_theta = cos_theta.clamp(-1, 1)
        
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta - self.margin, cos_theta)
        else:
            sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
            cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)
        
        # multiply by scale
        output = cos_theta_m * self.scale
        
        # one-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # apply margin only to positive pairs
        output = torch.where(one_hot == 1, output, cos_theta * self.scale)
        
        return output


class IrisRecognitionModel(nn.Module):
    """
    虹膜识别模型
    MobileNetV3 backbone + embedding head + ArcFace classification head
    """
    def __init__(self, num_classes, embedding_size=512, pretrained=True, backbone='mobilenet_v3'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.backbone_name = backbone
        
        # Backbone
        if backbone == 'mobilenet_v3':
            try:
                self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
                # Remove classifier
                self.backbone.classifier = nn.Identity()
                backbone_features = 960
            except Exception:
                # Fallback to MobileNetV2
                self.backbone = models.mobilenet_v2(pretrained=pretrained)
                self.backbone.classifier = nn.Identity()
                backbone_features = 1280
        else:
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            backbone_features = 1280
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_features, embedding_size * 2),
            nn.BatchNorm1d(embedding_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size, affine=False)  # No affine for embedding normalization
        )
        
        # ArcFace head
        self.arcface_head = ArcMarginProduct(embedding_size, num_classes, scale=30.0, margin=0.5)
        
    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone.features(x)
        
        # Get embeddings
        embeddings = self.embedding_head(features)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        if self.training and labels is not None:
            # Training mode with ArcFace
            logits = self.arcface_head(embeddings_norm, labels)
            return logits, embeddings_norm
        else:
            # Inference mode - return embeddings only
            return embeddings_norm


def build_recognition_model(num_classes, embedding_size=512, pretrained=True, backbone='mobilenet_v3'):
    """构建识别模型"""
    return IrisRecognitionModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        pretrained=pretrained,
        backbone=backbone
    )
