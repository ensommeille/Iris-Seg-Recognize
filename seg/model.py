# model.py - 优化版 MobileNetV3/2 encoder + UNet-style decoder
# 特性：
#  - MobileNetV3 优先（不可用回退到 MobileNetV2）
#  - Depthwise separable convs, CBAM, ASPP
#  - Deep supervision（侧输出融合）
#  - 可选 boundary head（辅助边界预测）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depth(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(in_planes // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_planes // reduction, 1), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_ = torch.cat([avg, mx], dim=1)
        x_ = self.conv(x_)
        return self.sigmoid(x_)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ASPP (lightweight)
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,6,12)):
        super().__init__()
        mid = out_ch // 2 if out_ch >= 64 else out_ch
        self.branches = nn.ModuleList()
        # 1x1
        self.branches.append(ConvBNReLU(in_ch, mid, kernel=1, padding=0))
        # dilated convs
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True)
            ))
        # image pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = ConvBNReLU(in_ch, mid, kernel=1, padding=0)
        self.project = ConvBNReLU(mid * (2 + len(dilations)), out_ch, kernel=1, padding=0)

    def forward(self, x):
        outs = []
        for b in self.branches:
            outs.append(b(x))
        p = self.pool(x)
        p = self.pool_conv(p)
        p = F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=False)
        outs.append(p)
        out = torch.cat(outs, dim=1)
        out = self.project(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = DepthwiseSeparableConv(in_ch + skip_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.att = CBAM(out_ch) if use_cbam else nn.Identity()
    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.att(x)
        return x

class MobileUNetOptimized(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, base_channels=32, use_boundary=False, backbone='mobilenet_v3'):
        super().__init__()
        self.use_boundary = use_boundary
        self.backbone_name = backbone
        # Try MobileNetV3 first, otherwise fallback to V2
        try_mbv3 = (backbone == 'mobilenet_v3')
        used_mb = None
        if try_mbv3:
            try:
                mb = models.mobilenet_v3_large(pretrained=pretrained)
                features = mb.features
                self.initial = nn.Sequential(features[0])
                self.enc1 = nn.Sequential(*features[1:3])
                self.enc2 = nn.Sequential(*features[3:6])
                self.enc3 = nn.Sequential(*features[6:12])
                self.enc4 = nn.Sequential(*features[12:])
                bottleneck_ch = 960
                used_mb = 'v3'
            except Exception:
                used_mb = None

        if used_mb is None:
            mb = models.mobilenet_v2(pretrained=pretrained)
            features = mb.features
            self.initial = features[0]
            self.enc1 = nn.Sequential(*features[1:3])
            self.enc2 = nn.Sequential(*features[3:6])
            self.enc3 = nn.Sequential(*features[6:13])
            self.enc4 = nn.Sequential(*features[13:])
            bottleneck_ch = 1280

        # ASPP
        self.aspp = ASPP(bottleneck_ch, base_channels * 8, dilations=(1,6,12))

        # project encoder features
        self.reduce3 = ConvBNReLU(self._get_channels(self.enc3), base_channels * 4, kernel=1, padding=0)
        self.reduce2 = ConvBNReLU(self._get_channels(self.enc2), base_channels * 2, kernel=1, padding=0)
        self.reduce1 = ConvBNReLU(self._get_channels(self.enc1), base_channels, kernel=1, padding=0)

        # decoder
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        # side outputs (deep supervision)
        self.side3 = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
        self.side1 = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # final
        self.final_conv = nn.Sequential(
            ConvBNReLU(base_channels, base_channels),
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )

        # optional boundary head
        if self.use_boundary:
            self.boundary_head = nn.Sequential(
                ConvBNReLU(base_channels, max(base_channels//2, 8)),
                nn.Conv2d(max(base_channels//2, 8), 1, kernel_size=1)
            )

    def _get_channels(self, module):
        try:
            last = list(module.children())[-1]
            if hasattr(last, 'out_channels'):
                return last.out_channels
        except Exception:
            pass
        return 96

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        r4 = self.aspp(x4)
        r3 = self.reduce3(x3)
        r2 = self.reduce2(x2)
        r1 = self.reduce1(x1)

        d3 = self.up3(r4, r3)
        d2 = self.up2(d3, r2)
        d1 = self.up1(d2, r1)

        s3 = F.interpolate(self.side3(d3), size=(H, W), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.side2(d2), size=(H, W), mode='bilinear', align_corners=False)
        s1 = F.interpolate(self.side1(d1), size=(H, W), mode='bilinear', align_corners=False)

        out_main = self.final_conv(d1)
        out_main_up = F.interpolate(out_main, size=(H, W), mode='bilinear', align_corners=False)

        out = (out_main_up + s1 + s2 + s3) / 4.0  # logits fused

        if self.use_boundary:
            boundary = self.boundary_head(d1)
            boundary = F.interpolate(boundary, size=(H, W), mode='bilinear', align_corners=False)
            return out, boundary

        return out

def build_model(num_classes=1, pretrained=True, base_channels=32, use_boundary=False, backbone='mobilenet_v3'):
    return MobileUNetOptimized(num_classes=num_classes, pretrained=pretrained, base_channels=base_channels,
                               use_boundary=use_boundary, backbone=backbone)

