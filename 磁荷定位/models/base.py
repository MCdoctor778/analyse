"""
基础模型类 - 所有模型的父类
"""

import torch
import torch.nn as nn
import yaml
import json
import os
from typing import Dict, Any, Optional, Union, Tuple


class BaseModel(nn.Module):
    """
    所有模型的基类，提供通用功能如保存、加载等
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    def forward(self, x):
        """
        前向传播方法，需要被子类重写
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型和元数据
        
        Args:
            path: 保存路径
            metadata: 额外的元数据
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'metadata': metadata or {}
        }
        
        # 保存模型
        torch.save(save_dict, path)
        
        # 保存配置（如果有）
        if hasattr(self, 'config'):
            config_path = os.path.splitext(path)[0] + '.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> Tuple['BaseModel', Dict[str, Any]]:
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 加载模型的设备
            
        Returns:
            加载的模型和元数据
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 加载保存的数据
        checkpoint = torch.load(path, map_location=device)
        
        # 检查模型类型
        model_name = checkpoint.get('model_name', cls.__name__)
        
        # 尝试加载配置
        config_path = os.path.splitext(path)[0] + '.yaml'
        config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # 创建模型实例
        if config is not None:
            model = cls(**config)
        else:
            model = cls()
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint.get('metadata', {})
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取模型参数数量
        
        Returns:
            包含总参数和可训练参数数量的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }
    
    def summary(self) -> str:
        """
        返回模型摘要信息
        
        Returns:
            模型摘要字符串
        """
        param_counts = self.get_parameter_count()
        
        summary_str = f"模型: {self.model_name}\n"
        summary_str += f"总参数: {param_counts['total']:,}\n"
        summary_str += f"可训练参数: {param_counts['trainable']:,}\n"
        
        return summary_str