"""
评估指标模块 - 提供各种评估指标函数
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算准确率
    
    Args:
        outputs: 预测值 [batch_size, num_classes]
        targets: 目标值 [batch_size]
        
    Returns:
        准确率
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).float()
    return correct.mean()


def precision(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    """
    计算精确率
    
    Args:
        outputs: 预测值 [batch_size, num_classes]
        targets: 目标值 [batch_size]
        average: 平均方式，'macro'或'micro'
        
    Returns:
        精确率
    """
    _, predicted = torch.max(outputs, 1)
    
    # 获取类别数
    num_classes = outputs.size(1)
    
    # 计算每个类别的精确率
    precisions = []
    for c in range(num_classes):
        true_positives = ((predicted == c) & (targets == c)).float().sum()
        predicted_positives = (predicted == c).float().sum()
        
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
        else:
            precisions.append(torch.tensor(0.0, device=outputs.device))
    
    # 计算平均精确率
    if average == 'macro':
        return torch.stack(precisions).mean()
    elif average == 'micro':
        true_positives = (predicted == targets).float().sum()
        return true_positives / targets.size(0)
    else:
        raise ValueError(f"未知的平均方式: {average}")


def recall(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    """
    计算召回率
    
    Args:
        outputs: 预测值 [batch_size, num_classes]
        targets: 目标值 [batch_size]
        average: 平均方式，'macro'或'micro'
        
    Returns:
        召回率
    """
    _, predicted = torch.max(outputs, 1)
    
    # 获取类别数
    num_classes = outputs.size(1)
    
    # 计算每个类别的召回率
    recalls = []
    for c in range(num_classes):
        true_positives = ((predicted == c) & (targets == c)).float().sum()
        actual_positives = (targets == c).float().sum()
        
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
        else:
            recalls.append(torch.tensor(0.0, device=outputs.device))
    
    # 计算平均召回率
    if average == 'macro':
        return torch.stack(recalls).mean()
    elif average == 'micro':
        true_positives = (predicted == targets).float().sum()
        return true_positives / targets.size(0)
    else:
        raise ValueError(f"未知的平均方式: {average}")


def f1_score(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    """
    计算F1分数
    
    Args:
        outputs: 预测值 [batch_size, num_classes]
        targets: 目标值 [batch_size]
        average: 平均方式，'macro'或'micro'
        
    Returns:
        F1分数
    """
    p = precision(outputs, targets, average)
    r = recall(outputs, targets, average)
    
    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return torch.tensor(0.0, device=outputs.device)


def iou_score(outputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算IoU分数（用于分割任务）
    
    Args:
        outputs: 预测值 [batch_size, num_classes, height, width]
        targets: 目标值 [batch_size, num_classes, height, width]
        smooth: 平滑因子
        
    Returns:
        IoU分数
    """
    # 应用sigmoid或softmax
    if outputs.size(1) > 1:
        outputs = torch.softmax(outputs, dim=1)
    else:
        outputs = torch.sigmoid(outputs)
    
    # 二值化预测值
    outputs = (outputs > 0.5).float()
    
    # 计算交集和并集
    intersection = (outputs * targets).sum(dim=(2, 3))
    union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # 返回平均IoU
    return iou.mean()


def dice_coefficient(outputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算Dice系数（用于分割任务）
    
    Args:
        outputs: 预测值 [batch_size, num_classes, height, width]
        targets: 目标值 [batch_size, num_classes, height, width]
        smooth: 平滑因子
        
    Returns:
        Dice系数
    """
    # 应用sigmoid或softmax
    if outputs.size(1) > 1:
        outputs = torch.softmax(outputs, dim=1)
    else:
        outputs = torch.sigmoid(outputs)
    
    # 二值化预测值
    outputs = (outputs > 0.5).float()
    
    # 计算交集
    intersection = (outputs * targets).sum(dim=(2, 3))
    
    # 计算Dice系数
    dice = (2.0 * intersection + smooth) / (outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    
    # 返回平均Dice系数
    return dice.mean()


def confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    计算混淆矩阵
    
    Args:
        outputs: 预测值 [batch_size, num_classes]
        targets: 目标值 [batch_size]
        num_classes: 类别数
        
    Returns:
        混淆矩阵 [num_classes, num_classes]
    """
    _, predicted = torch.max(outputs, 1)
    
    # 初始化混淆矩阵
    conf_matrix = torch.zeros(num_classes, num_classes, device=outputs.device)
    
    # 填充混淆矩阵
    for t, p in zip(targets.view(-1), predicted.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    
    return conf_matrix