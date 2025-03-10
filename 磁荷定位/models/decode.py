"""
DECODE模型 - 基于深度学习的单分子定位模型
参考: https://www.nature.com/articles/s41592-020-0869-y
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import BaseModel
from .unet import UNet, DoubleConv, Down, Up, OutConv


class TemporalBlock(nn.Module):
    """时间上下文处理模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 
                             kernel_size=(kernel_size, 1, 1),
                             padding=(kernel_size//2, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x形状: [B, C, T, H, W]
        return self.relu(self.bn(self.conv(x)))


class DECODEModel(BaseModel):
    """
    DECODE模型实现 - 用于高密度三维单分子定位
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 frame_window: int = 5,
                 param_channels: int = 6,  # 坐标(x,y,z), 不确定性, 强度, 背景
                 features: List[int] = [64, 128, 256, 512],
                 bilinear: bool = True):
        """
        初始化DECODE模型
        
        Args:
            in_channels: 输入通道数
            frame_window: 时间窗口大小
            param_channels: 输出参数通道数
            features: 特征通道数列表
            bilinear: 是否使用双线性插值进行上采样
        """
        super().__init__()
        
        self.config = {
            'in_channels': in_channels,
            'frame_window': frame_window,
            'param_channels': param_channels,
            'features': features,
            'bilinear': bilinear
        }
        
        # 单帧U-Net
        self.frame_unet = UNet(
            in_channels=in_channels,
            out_channels=features[0],
            features=features,
            bilinear=bilinear
        )
        
        # 时间上下文处理模块
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(features[0], features[0], kernel_size=3),
            TemporalBlock(features[0], features[0], kernel_size=3)
        ])
        
        # 参数预测U-Net
        self.param_unet = UNet(
            in_channels=features[0],
            out_channels=param_channels,
            features=features,
            bilinear=bilinear
        )
        
        # 检测概率预测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, frame_window, height, width]
            
        Returns:
            包含各项预测的字典
        """
        batch_size, frame_count, height, width = x.shape
        
        # 处理每一帧
        frame_features = []
        for i in range(frame_count):
            # [B, 1, H, W]
            frame = x[:, i:i+1, :, :]
            # [B, C, H, W]
            feat = self.frame_unet(frame)
            frame_features.append(feat)
        
        # [B, C, T, H, W]
        temporal_features = torch.stack(frame_features, dim=2)
        
        # 时间上下文处理
        for temporal_block in self.temporal_blocks:
            temporal_features = temporal_block(temporal_features)
        
        # 取中心帧特征
        center_features = temporal_features[:, :, frame_count//2, :, :]
        
        # 参数预测
        params = self.param_unet(center_features)
        
        # 检测概率预测
        detection_prob = self.detection_head(center_features)
        
        # 分离各个参数
        x_coord = params[:, 0:1, :, :]  # x坐标
        y_coord = params[:, 1:2, :, :]  # y坐标
        z_coord = params[:, 2:3, :, :]  # z坐标
        uncertainty = params[:, 3:4, :, :]  # 不确定性
        intensity = params[:, 4:5, :, :]  # 发射强度
        background = params[:, 5:6, :, :]  # 背景
        
        return {
            'detection_prob': detection_prob,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'z_coord': z_coord,
            'uncertainty': uncertainty,
            'intensity': intensity,
            'background': background
        }
    
    def predict_emitters(self, 
                         x: torch.Tensor, 
                         detection_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        预测发射体位置和参数
        
        Args:
            x: 输入图像序列 [batch_size, frame_window, height, width]
            detection_threshold: 检测概率阈值
            
        Returns:
            包含发射体位置和参数的字典
        """
        self.eval()
        with torch.no_grad():
            # 获取预测结果
            results = self(x)
            
            # 应用阈值
            detection_mask = results['detection_prob'] > detection_threshold
            
            # 提取发射体坐标和参数
            emitters = {}
            for key, value in results.items():
                if key != 'detection_prob':
                    # 只保留检测到的发射体的参数
                    emitters[key] = value[detection_mask]
            
            # 添加检测概率
            emitters['detection_prob'] = results['detection_prob'][detection_mask]
            
            return emitters 