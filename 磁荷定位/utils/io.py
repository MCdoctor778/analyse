"""
IO工具函数 - 用于处理文件读写
"""

import yaml
import json
import os
import pickle
import torch
from typing import Any, Dict, List, Optional, Union


def load_yaml(path: str) -> Dict:
    """
    加载YAML文件
    
    Args:
        path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: str, **kwargs) -> None:
    """
    保存数据到YAML文件
    
    Args:
        data: 要保存的数据
        path: 保存路径
        **kwargs: 其他参数传递给yaml.dump
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, **kwargs)


def load_json(path: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, path: str, **kwargs) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        path: 保存路径
        **kwargs: 其他参数传递给json.dump
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, **kwargs)


def load_pickle(path: str) -> Any:
    """
    加载Pickle文件
    
    Args:
        path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str) -> None:
    """
    保存数据到Pickle文件
    
    Args:
        data: 要保存的数据
        path: 保存路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_model(path: str, model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    加载模型
    
    Args:
        path: 模型路径
        model: 模型实例
        device: 设备
        
    Returns:
        加载的模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_dict = torch.load(path, map_location=device)
    
    # 处理不同格式的保存文件
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model


def save_model(model: torch.nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
               epoch: Optional[int] = None, **kwargs) -> None:
    """
    保存模型
    
    Args:
        model: 要保存的模型
        path: 保存路径
        optimizer: 优化器
        epoch: 当前轮数
        **kwargs: 其他要保存的数据
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    save_dict.update(kwargs)
    
    torch.save(save_dict, path)


def list_files(directory: str, extension: Optional[str] = None, recursive: bool = False) -> List[str]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        extension: 文件扩展名
        recursive: 是否递归搜索
        
    Returns:
        文件路径列表
    """
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extension is None or filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f))]
        if extension is not None:
            files = [f for f in files if f.endswith(extension)]
        return files