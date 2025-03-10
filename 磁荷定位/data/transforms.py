"""
数据转换模块 - 提供各种数据转换函数
"""

import torch
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


def get_transforms(task: str, **kwargs) -> Callable:
    """
    获取数据转换函数
    
    Args:
        task: 任务类型，如'classification', 'segmentation', 'detection'等
        **kwargs: 其他参数
        
    Returns:
        数据转换函数
    """
    if task == 'classification':
        return get_classification_transforms(**kwargs)
    elif task == 'segmentation':
        return get_segmentation_transforms(**kwargs)
    elif task == 'detection':
        return get_detection_transforms(**kwargs)
    else:
        raise ValueError(f"未知的任务类型: {task}")


def get_classification_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment: bool = True
) -> Dict[str, Callable]:
    """
    获取图像分类任务的数据转换函数
    
    Args:
        input_size: 输入大小
        mean: 均值
        std: 标准差
        augment: 是否进行数据增强
        
    Returns:
        包含'train'和'val'转换函数的字典
    """
    # 验证转换
    val_transforms = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    if not augment:
        return {'train': val_transforms, 'val': val_transforms}
    
    # 训练转换（带数据增强）
    train_transforms = T.Compose([
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    return {'train': train_transforms, 'val': val_transforms}


def get_segmentation_transforms(
    input_size: Tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment: bool = True
) -> Dict[str, Callable]:
    """
    获取图像分割任务的数据转换函数
    
    Args:
        input_size: 输入大小
        mean: 均值
        std: 标准差
        augment: 是否进行数据增强
        
    Returns:
        包含'train'和'val'转换函数的字典
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # 验证转换
        val_transforms = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        if not augment:
            return {'train': val_transforms, 'val': val_transforms}
        
        # 训练转换（带数据增强）
        train_transforms = A.Compose([
            A.RandomResizedCrop(height=input_size[0], width=input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        return {'train': train_transforms, 'val': val_transforms}
    
    except ImportError:
        print("警告: albumentations库未安装，使用torchvision.transforms代替")
        
        # 验证转换
        val_transforms = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        if not augment:
            return {'train': val_transforms, 'val': val_transforms}
        
        # 训练转换（带数据增强）
        train_transforms = T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return {'train': train_transforms, 'val': val_transforms}


def get_detection_transforms(
    input_size: Tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment: bool = True
) -> Dict[str, Callable]:
    """
    获取目标检测任务的数据转换函数
    
    Args:
        input_size: 输入大小
        mean: 均值
        std: 标准差
        augment: 是否进行数据增强
        
    Returns:
        包含'train'和'val'转换函数的字典
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # 验证转换
        val_transforms = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        if not augment:
            return {'train': val_transforms, 'val': val_transforms}
        
        # 训练转换（带数据增强）
        train_transforms = A.Compose([
            A.RandomResizedCrop(height=input_size[0], width=input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        return {'train': train_transforms, 'val': val_transforms}
    
    except ImportError:
        print("警告: albumentations库未安装，目标检测任务需要安装albumentations")
        
        # 返回简单的转换函数
        simple_transform = lambda x: x
        return {'train': simple_transform, 'val': simple_transform}