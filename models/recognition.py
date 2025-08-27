"""
虹膜识别模型定义
MobileNetV3 backbone + ArcFace head + Triplet Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V2_Weights
except Exception:
    MobileNet_V3_Large__weights = None
    MobileNet_V2_Weights = None
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
        self._update_cached_trig()

    def _update_cached_trig(self):
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def set_params(self, scale: float | None = None, margin: float | None = None):
        """Dynamically update scale and/or margin during training."""
        if scale is not None:
            self.scale = float(scale)
        if margin is not None:
            self.margin = float(margin)
            self._update_cached_trig()

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
                if 'MobileNet_V3_Large_Weights' in globals() and MobileNet_V3_Large_Weights is not None:
                    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                    self.backbone = models.mobilenet_v3_large(weights=weights)
                else:
                    self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
                # Remove classifier
                self.backbone.classifier = nn.Identity()
                backbone_features = 960
            except Exception:
                # Fallback to MobileNetV2
                if 'MobileNet_V2_Weights' in globals() and MobileNet_V2_Weights is not None:
                    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                    self.backbone = models.mobilenet_v2(weights=weights)
                else:
                    self.backbone = models.mobilenet_v2(pretrained=pretrained)
                self.backbone.classifier = nn.Identity()
                backbone_features = 1280
        elif backbone == 'complex_irisnet':
            # Lightweight complex-valued backbone
            self.backbone = ComplexIrisNetWrapper(base_channels=32)
            backbone_features = self.backbone.out_channels
        else:
            if 'MobileNet_V2_Weights' in globals() and MobileNet_V2_Weights is not None:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.mobilenet_v2(weights=weights)
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
        if self.backbone_name == 'complex_irisnet':
            features = self.backbone.features(x)
        else:
            features = self.backbone.features(x)
        
        # Get embeddings
        embeddings = self.embedding_head(features)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # If labels are provided, always compute ArcFace logits (works for train and eval)
        if labels is not None:
            logits = self.arcface_head(embeddings_norm, labels)
            return logits, embeddings_norm
        
        # Otherwise return embeddings only
        return embeddings_norm


def build_recognition_model(num_classes, embedding_size=512, pretrained=True, backbone='mobilenet_v3'):
    """构建识别模型"""
    return IrisRecognitionModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        pretrained=pretrained,
        backbone=backbone
    )


# ------------------------------
# Complex-valued primitives
# ------------------------------
class ComplexConv2d(nn.Module):
    """Complex 2D convolution implemented with two real convolutions.
    Input/Output channel convention: [real, imag] concatenated along C.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        # Real and Imag convs
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        c = x.shape[1] // 2
        xr, xi = x[:, :c], x[:, c:]
        yr = self.conv_re(xr) - self.conv_im(xi)
        yi = self.conv_re(xi) + self.conv_im(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Apply BN to real and imag parts separately."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        c = x.shape[1] // 2
        xr, xi = x[:, :c], x[:, c:]
        xr = self.bn_r(xr)
        xi = self.bn_i(xi)
        return torch.cat([xr, xi], dim=1)


class ComplexReLU(nn.Module):
    """ReLU applied to real and imag parts independently."""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        c = x.shape[1] // 2
        xr, xi = x[:, :c], x[:, c:]
        xr = self.relu(xr)
        xi = self.relu(xi)
        return torch.cat([xr, xi], dim=1)


class ComplexConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            ComplexBatchNorm2d(out_ch),
            ComplexReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ComplexIrisNetBackbone(nn.Module):
    """A lightweight complex-valued CNN backbone with downsampling stages.
    Input is real RGB (B,3,H,W). We convert it to complex by concatenating a zero imaginary part.
    The network returns complex features as concatenated real/imag channels (2*C).
    """
    def __init__(self, base_channels=32):
        super().__init__()
        # After converting to complex, input channels become 2*3=6, but our ComplexConv2d expects in_ch as real channels count.
        # We keep a helper that expands input to complex at runtime; first layer therefore uses in_ch=3 and internally handles split.
        # To keep shapes consistent, we manually build complex tensors before feeding blocks.
        self.c1 = ComplexConvBNReLU(3, base_channels, k=3, s=2, p=1)      # -> 2*base
        self.c2 = ComplexConvBNReLU(base_channels, base_channels*2, k=3, s=2, p=1)  # -> 4*base
        self.c3 = ComplexConvBNReLU(base_channels*2, base_channels*4, k=3, s=2, p=1) # -> 8*base
        self.c4 = ComplexConvBNReLU(base_channels*4, base_channels*8, k=3, s=1, p=1) # -> 16*base
        self.out_complex_channels = base_channels * 8  # real channels count (imag the same)

    def _to_complex(self, x):
        # x: (B, C, H, W) real-valued -> concat imag zeros
        zeros = torch.zeros_like(x)
        return torch.cat([x, zeros], dim=1)

    def forward(self, x):
        # Represent as complex (concat real/imag)
        x = self._to_complex(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        # Return as concatenated real/imag: channels = 2*out_complex_channels
        return x


class ComplexIrisNetWrapper(nn.Module):
    """Wrapper to expose a .features module so it matches MobileNet interface."""
    def __init__(self, base_channels=32):
        super().__init__()
        self.features = ComplexIrisNetBackbone(base_channels=base_channels)
        # Expose out channels (real+imag concatenated)
        self.out_channels = self.features.out_complex_channels * 2

    def forward(self, x):
        return self.features(x)
