"""
数据集类 - 包括基础数据集、图像数据集和文本数据集
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from PIL import Image
import glob


class BaseDataset(Dataset):
    """
    基础数据集类
    """
    
    def __init__(self, 
                 data: Union[List, np.ndarray, pd.DataFrame],
                 transform: Optional[Callable] = None):
        """
        初始化基础数据集
        
        Args:
            data: 数据
            transform: 转换函数
        """
        self.data = data
        self.transform = transform
    
    def __len__(self) -> int:
        """
        返回数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项
        """
        item = self.data[idx]
        
        if self.transform is not None:
            item = self.transform(item)
        
        return item


class ImageDataset(BaseDataset):
    """
    图像数据集类
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: Optional[List[Any]] = None,
                 transform: Optional[Callable] = None):
        """
        初始化图像数据集
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 转换函数
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """
        返回数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            (图像, 标签)元组
        """
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform is not None:
            image = self.transform(image)
        
        # 返回图像和标签
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image
    
    @classmethod
    def from_directory(cls, 
                       directory: str, 
                       transform: Optional[Callable] = None,
                       extensions: List[str] = ['jpg', 'jpeg', 'png'],
                       recursive: bool = True) -> 'ImageDataset':
        """
        从目录创建图像数据集
        
        Args:
            directory: 图像目录
            transform: 转换函数
            extensions: 图像扩展名列表
            recursive: 是否递归搜索
            
        Returns:
            图像数据集
        """
        # 获取所有图像路径
        image_paths = []
        labels = []
        
        # 获取所有子目录（类别）
        class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        if class_dirs:
            # 有子目录，按类别组织
            for class_idx, class_dir in enumerate(sorted(class_dirs)):
                class_path = os.path.join(directory, class_dir)
                
                # 获取该类别的所有图像
                for ext in extensions:
                    if recursive:
                        pattern = os.path.join(class_path, f'**/*.{ext}')
                        paths = glob.glob(pattern, recursive=True)
                    else:
                        pattern = os.path.join(class_path, f'*.{ext}')
                        paths = glob.glob(pattern)
                    
                    image_paths.extend(paths)
                    labels.extend([class_idx] * len(paths))
        else:
            # 没有子目录，直接获取所有图像
            for ext in extensions:
                if recursive:
                    pattern = os.path.join(directory, f'**/*.{ext}')
                    paths = glob.glob(pattern, recursive=True)
                else:
                    pattern = os.path.join(directory, f'*.{ext}')
                    paths = glob.glob(pattern)
                
                image_paths.extend(paths)
            
            # 没有标签
            labels = None
        
        return cls(image_paths, labels, transform)


class TextDataset(BaseDataset):
    """
    文本数据集类
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: Optional[List[Any]] = None,
                 tokenizer: Optional[Callable] = None,
                 max_length: Optional[int] = None):
        """
        初始化文本数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """
        返回数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[Union[str, Dict[str, torch.Tensor]], Optional[Any]]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            (文本, 标签)元组
        """
        text = self.texts[idx]
        
        # 应用分词器
        if self.tokenizer is not None:
            if self.max_length is not None:
                encoded = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            else:
                encoded = self.tokenizer(text, return_tensors='pt')
            
            # 移除批次维度
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            text = encoded
        
        # 返回文本和标签
        if self.labels is not None:
            return text, self.labels[idx]
        else:
            return text
    
    @classmethod
    def from_csv(cls, 
                 csv_path: str, 
                 text_column: str,
                 label_column: Optional[str] = None,
                 tokenizer: Optional[Callable] = None,
                 max_length: Optional[int] = None) -> 'TextDataset':
        """
        从CSV文件创建文本数据集
        
        Args:
            csv_path: CSV文件路径
            text_column: 文本列名
            label_column: 标签列名
            tokenizer: 分词器
            max_length: 最大长度
            
        Returns:
            文本数据集
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 获取文本和标签
        texts = df[text_column].tolist()
        labels = df[label_column].tolist() if label_column is not None else None
        
        return cls(texts, labels, tokenizer, max_length)
    
    @classmethod
    def from_json(cls, 
                  json_path: str, 
                  text_key: str,
                  label_key: Optional[str] = None,
                  tokenizer: Optional[Callable] = None,
                  max_length: Optional[int] = None) -> 'TextDataset':
        """
        从JSON文件创建文本数据集
        
        Args:
            json_path: JSON文件路径
            text_key: 文本键名
            label_key: 标签键名
            tokenizer: 分词器
            max_length: 最大长度
            
        Returns:
            文本数据集
        """
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取文本和标签
        texts = [item[text_key] for item in data]
        labels = [item[label_key] for item in data] if label_key is not None else None
        
        return cls(texts, labels, tokenizer, max_length)