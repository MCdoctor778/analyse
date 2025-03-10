"""
杂项工具函数 - 包括随机种子设置和设备获取等
"""

import random
import numpy as np
import torch
import os
from typing import Optional, Union, List, Tuple


def seed_everything(seed: int) -> None:
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    获取设备
    
    Args:
        device_id: GPU设备ID，如果为None则使用CPU
        
    Returns:
        设备
    """
    if device_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{device_id}')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将PyTorch张量转换为NumPy数组
    
    Args:
        tensor: PyTorch张量
        
    Returns:
        NumPy数组
    """
    return tensor.detach().cpu().numpy()


def to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    将NumPy数组转换为PyTorch张量
    
    Args:
        array: NumPy数组
        device: 设备
        
    Returns:
        PyTorch张量
    """
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    独热编码
    
    Args:
        labels: 标签张量 [batch_size]
        num_classes: 类别数
        
    Returns:
        独热编码张量 [batch_size, num_classes]
    """
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(
        1, labels.unsqueeze(1), 1
    )


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取优化器的学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """
    设置优化器的学习率
    
    Args:
        optimizer: 优化器
        lr: 学习率
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_memory_usage() -> Tuple[float, float]:
    """
    获取当前GPU和CPU内存使用情况
    
    Returns:
        (GPU内存使用量(MB), CPU内存使用量(MB))
    """
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    try:
        import psutil
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        cpu_memory = 0
    
    return gpu_memory, cpu_memory