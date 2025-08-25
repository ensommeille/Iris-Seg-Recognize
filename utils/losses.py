"""
损失函数定义
包含分割和识别任务所需的各种损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(preds, targets, eps=1e-7):
    """Dice Loss for segmentation"""
    preds = preds.float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=[1, 2, 3])
    denom = preds.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
    loss = 1.0 - ((2 * inter + eps) / (denom + eps))
    return loss.mean()


def focal_loss(preds, targets, alpha=0.25, gamma=2.0, eps=1e-7):
    """Focal Loss for segmentation"""
    preds = torch.sigmoid(preds)
    targets = targets.float()
    
    # BCE loss
    bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
    
    # Focal term
    pt = preds * targets + (1 - preds) * (1 - targets)
    focal_term = (1 - pt) ** gamma
    
    # Alpha term
    alpha_term = alpha * targets + (1 - alpha) * (1 - targets)
    
    focal_loss = alpha_term * focal_term * bce_loss
    return focal_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """组合分割损失函数：BCE + Dice + Focal"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets)
        
        # Dice Loss
        dice_loss_val = dice_loss(torch.sigmoid(preds), targets)
        
        # Focal Loss
        focal_loss_val = focal_loss(preds, targets)
        
        # Combined loss
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss_val + 
                     self.focal_weight * focal_loss_val)
        
        return total_loss


class TripletLoss(nn.Module):
    """Triplet Loss with batch-hard mining"""
    def __init__(self, margin=0.2, distance='cosine'):
        super().__init__()
        self.margin = margin
        self.distance = distance
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (N, D) tensor of embeddings
            labels: (N,) tensor of labels
        """
        if self.distance == 'cosine':
            # Normalize embeddings for cosine distance
            embeddings = F.normalize(embeddings, dim=1)
            dist_matrix = 1 - torch.mm(embeddings, embeddings.t())
        else:
            # Euclidean distance
            dist_matrix = torch.cdist(embeddings, embeddings)
        
        # Get positive and negative masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = (labels != labels.t()).float()
        
        # Find hardest positive and negative for each anchor
        hardest_positive = (dist_matrix * positive_mask).max(dim=1)[0]
        hardest_negative = (dist_matrix + 1e6 * positive_mask).min(dim=1)[0]
        
        # Triplet loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
        
        # Only consider valid triplets (where we have both positive and negative)
        valid_mask = (hardest_positive > 0) & (hardest_negative < 1e6)
        if valid_mask.sum() > 0:
            triplet_loss = triplet_loss[valid_mask].mean()
        else:
            triplet_loss = triplet_loss.mean()
            
        return triplet_loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss"""
    def __init__(self, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
    def forward(self, logits, labels):
        """Standard cross entropy loss for ArcFace logits"""
        return F.cross_entropy(logits, labels)


class CombinedRecognitionLoss(nn.Module):
    """组合识别损失函数：ArcFace + Triplet"""
    def __init__(self, arcface_weight=1.0, triplet_weight=0.5, triplet_margin=0.2):
        super().__init__()
        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.arcface_loss = ArcFaceLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
    def forward(self, logits, embeddings, labels):
        """
        Args:
            logits: ArcFace logits
            embeddings: normalized embeddings
            labels: ground truth labels
        """
        arcface_loss = self.arcface_loss(logits, labels)
        triplet_loss = self.triplet_loss(embeddings, labels)
        
        total_loss = (self.arcface_weight * arcface_loss + 
                     self.triplet_weight * triplet_loss)
        
        return total_loss, arcface_loss, triplet_loss
