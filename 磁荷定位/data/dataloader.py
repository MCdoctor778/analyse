"""
数据加载器模块 - 提供数据加载器工厂函数
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否将数据固定在内存中
        drop_last: 是否丢弃最后一个不完整的批次
        collate_fn: 整理函数
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    分割数据集为训练集、验证集和测试集
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (训练集, 验证集, 测试集)元组
    """
    # 检查比例和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 计算每个集合的大小
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    collate_fn: Optional[Callable] = None
) -> Dict[str, DataLoader]:
    """
    创建训练集、验证集和测试集的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_workers: 工作进程数
        pin_memory: 是否将数据固定在内存中
        seed: 随机种子
        collate_fn: 整理函数
        
    Returns:
        包含'train', 'val', 'test'数据加载器的字典
    """
    # 分割数据集
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )
    
    # 创建数据加载器
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }