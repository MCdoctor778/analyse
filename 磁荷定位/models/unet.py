"""
U-Net模型实现 - 用于图像分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import BaseModel


class DoubleConv(nn.Module):
    """双卷积块"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 使用双线性插值或转置卷积进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 输入可能不是2的幂次方大小，需要进行裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 连接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(BaseModel):
    """
    U-Net模型实现
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 1,
                 features: List[int] = [64, 128, 256, 512],
                 bilinear: bool = True):
        """
        初始化U-Net模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（类别数）
            features: 特征通道数列表
            bilinear: 是否使用双线性插值进行上采样
        """
        super().__init__()
        
        self.config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'features': features,
            'bilinear': bilinear
        }
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 下采样路径
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # 瓶颈层
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # 上采样路径
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # 输出层
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels, height, width]
            
        Returns:
            输出张量 [batch_size, out_channels, height, width]
        """
        # 下采样路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 上采样路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出层
        logits = self.outc(x)
        
        return logits
    
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        预测分割掩码
        
        Args:
            x: 输入图像
            threshold: 二值化阈值
            
        Returns:
            二值化分割掩码
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            
            if self.out_channels > 1:
                # 多类别分割
                return torch.argmax(logits, dim=1)
            else:
                # 二分类分割
                return (torch.sigmoid(logits) > threshold).float()