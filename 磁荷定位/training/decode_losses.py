"""
DECODE损失函数 - 用于训练DECODE模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class DECODELoss(nn.Module):
    """
    DECODE综合损失函数
    """
    
    def __init__(self, 
                 detection_weight: float = 1.0,
                 coord_weight: float = 1.0,
                 z_weight: float = 0.5,
                 intensity_weight: float = 0.2,
                 background_weight: float = 0.1,
                 uncertainty_weight: float = 0.1):
        """
        初始化DECODE损失函数
        
        Args:
            detection_weight: 检测损失权重
            coord_weight: 坐标损失权重
            z_weight: z坐标损失权重
            intensity_weight: 强度损失权重
            background_weight: 背景损失权重
            uncertainty_weight: 不确定性损失权重
        """
        super().__init__()
        self.detection_weight = detection_weight
        self.coord_weight = coord_weight
        self.z_weight = z_weight
        self.intensity_weight = intensity_weight
        self.background_weight = background_weight
        self.uncertainty_weight = uncertainty_weight
        
        # 损失函数
        self.detection_loss = nn.BCELoss(reduction='none')
        self.coord_loss = nn.SmoothL1Loss(reduction='none')
        self.z_loss = nn.SmoothL1Loss(reduction='none')
        self.intensity_loss = nn.SmoothL1Loss(reduction='none')
        self.background_loss = nn.MSELoss(reduction='mean')
        self.uncertainty_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算DECODE损失
        
        Args:
            predictions: 预测结果字典
            targets: 目标值字典
            
        Returns:
            损失字典
        """
        # 检测掩码
        detection_mask = targets['detection_map'] > 0.5
        
        # 检测损失
        detection_loss = self.detection_loss(
            predictions['detection_prob'],
            targets['detection_map']
        )
        
        # 均衡检测损失（正负样本加权）
        pos_weight = (targets['detection_map'] == 0).float().sum() / (targets['detection_map'] > 0).float().sum()
        weighted_detection_loss = torch.where(
            targets['detection_map'] > 0.5,
            detection_loss * pos_weight,
            detection_loss
        )
        total_detection_loss = weighted_detection_loss.mean()
        
        # 只在检测点计算坐标损失
        if detection_mask.sum() > 0:
            # 坐标损失
            x_loss = self.coord_loss(
                predictions['x_coord'][detection_mask],
                targets['x_coord'][detection_mask]
            ).mean()
            
            y_loss = self.coord_loss(
                predictions['y_coord'][detection_mask],
                targets['y_coord'][detection_mask]
            ).mean()
            
            total_coord_loss = x_loss + y_loss
            
            # z坐标损失
            z_loss = self.z_loss(
                predictions['z_coord'][detection_mask],
                targets['z_coord'][detection_mask]
            ).mean()
            
            # 强度损失
            intensity_loss = self.intensity_loss(
                predictions['intensity'][detection_mask],
                targets['intensity'][detection_mask]
            ).mean()
            
            # 不确定性损失
            uncertainty_loss = self.uncertainty_loss(
                predictions['uncertainty'][detection_mask],
                targets['uncertainty'][detection_mask]
            ).mean()
        else:
            total_coord_loss = torch.tensor(0.0, device=predictions['x_coord'].device)
            z_loss = torch.tensor(0.0, device=predictions['x_coord'].device)
            intensity_loss = torch.tensor(0.0, device=predictions['x_coord'].device)
            uncertainty_loss = torch.tensor(0.0, device=predictions['x_coord'].device)
        
        # 背景损失
        background_loss = self.background_loss(
            predictions['background'],
            targets['background']
        )
        
        # 总损失
        total_loss = (
            self.detection_weight * total_detection_loss +
            self.coord_weight * total_coord_loss +
            self.z_weight * z_loss +
            self.intensity_weight * intensity_loss +
            self.background_weight * background_loss +
            self.uncertainty_weight * uncertainty_loss
        )
        
        # 返回损失字典
        return {
            'loss': total_loss,
            'detection_loss': total_detection_loss,
            'coord_loss': total_coord_loss,
            'z_loss': z_loss,
            'intensity_loss': intensity_loss,
            'background_loss': background_loss,
            'uncertainty_loss': uncertainty_loss
        } 