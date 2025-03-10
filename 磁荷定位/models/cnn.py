"""
CNN模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import BaseModel


class CNNModel(BaseModel):
    """
    基础CNN模型
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 10,
                 hidden_channels: List[int] = [16, 32, 64, 128],
                 kernel_size: int = 3,
                 dropout_rate: float = 0.2):
        """
        初始化CNN模型
        
        Args:
            in_channels: 输入通道数
            num_classes: 分类数量
            hidden_channels: 隐藏层通道数列表
            kernel_size: 卷积核大小
            dropout_rate: Dropout比率
        """
        super().__init__()
        
        self.config = {
            'in_channels': in_channels,
            'num_classes': num_classes,
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'dropout_rate': dropout_rate
        }
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层卷积
        self.conv_layers.append(nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=kernel_size//2))
        
        # 添加剩余卷积层
        for i in range(len(hidden_channels)-1):
            self.conv_layers.append(
                nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size, padding=kernel_size//2)
            )
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层 - 假设输入图像为28x28
        # 经过len(hidden_channels)次池化后，特征图大小为28/(2^len(hidden_channels))
        feature_size = 28 // (2 ** len(hidden_channels))
        self.fc1 = nn.Linear(hidden_channels[-1] * feature_size * feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels, height, width]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 应用卷积层
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测函数
        
        Args:
            x: 输入张量
            
        Returns:
            预测类别
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted