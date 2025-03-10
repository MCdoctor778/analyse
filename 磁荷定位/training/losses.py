"""
损失函数模块 - 提供各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Callable


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡问题
    
    参考: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 'none', 'mean' 或 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 预测值 [B, C, ...]
            targets: 目标值 [B, ...]
            
        Returns:
            损失值
        """
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - 用于图像分割
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        初始化Dice Loss
        
        Args:
            smooth: 平滑因子，防止分母为0
            reduction: 'none', 'mean' 或 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        Args:
            inputs: 预测值 [B, C, ...]
            targets: 目标值 [B, C, ...]
            
        Returns:
            损失值
        """
        # 应用sigmoid
        inputs = torch.sigmoid(inputs)
        
        # 展平输入和目标
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 计算Dice Loss
        dice_loss = 1.0 - dice
        
        # 应用reduction
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss * inputs.size(0)
        else:
            return dice_loss.unsqueeze(0).repeat(inputs.size(0))


class CombinedLoss(nn.Module):
    """
    组合损失函数 - 组合多个损失函数
    """
    
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float]):
        """
        初始化组合损失函数
        
        Args:
            losses: 损失函数字典
            weights: 权重字典
        """
        super().__init__()
        self.losses = losses
        self.weights = weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            inputs: 预测值
            targets: 目标值
            
        Returns:
            组合损失值
        """
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
        
        return total_loss


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    获取损失函数
    
    Args:
        name: 损失函数名称
        **kwargs: 损失函数参数
        
    Returns:
        损失函数
    """
    if name == 'mse' or name == 'MSELoss':
        return nn.MSELoss(**kwargs)
    elif name == 'l1' or name == 'L1Loss':
        return nn.L1Loss(**kwargs)
    elif name == 'ce' or name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'bce' or name == 'BCELoss':
        return nn.BCELoss(**kwargs)
    elif name == 'bce_with_logits' or name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'nll' or name == 'NLLLoss':
        return nn.NLLLoss(**kwargs)
    elif name == 'kl_div' or name == 'KLDivLoss':
        return nn.KLDivLoss(**kwargs)
    elif name == 'smooth_l1' or name == 'SmoothL1Loss':
        return nn.SmoothL1Loss(**kwargs)
    elif name == 'focal' or name == 'FocalLoss':
        return FocalLoss(**kwargs)
    elif name == 'dice' or name == 'DiceLoss':
        return DiceLoss(**kwargs)
    else:
        raise ValueError(f"未知的损失函数: {name}")