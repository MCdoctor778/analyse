"""
配置类 - 用于管理模型和训练配置
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union


class Config:
    """
    配置类，用于管理模型和训练配置
    """
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        初始化配置类
        
        Args:
            config: 配置字典或配置文件路径
        """
        self._config = {}
        
        if config is not None:
            if isinstance(config, dict):
                self._config = config
            elif isinstance(config, str):
                self.load(config)
    
    def __getitem__(self, key: str) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键
            
        Returns:
            配置项值
        """
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置项键
            value: 配置项值
        """
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        检查配置项是否存在
        
        Args:
            key: 配置项键
            
        Returns:
            配置项是否存在
        """
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，如果不存在则返回默认值
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            配置项值
        """
        return self._config.get(key, default)
    
    def update(self, config: Dict) -> None:
        """
        更新配置
        
        Args:
            config: 配置字典
        """
        self._config.update(config)
    
    def load(self, path: str) -> None:
        """
        从文件加载配置
        
        Args:
            path: 配置文件路径
        """
        _, ext = os.path.splitext(path)
        
        if ext.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                self._config.update(yaml.safe_load(f))
        elif ext.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self._config.update(json.load(f))
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    
    def save(self, path: str) -> None:
        """
        保存配置到文件
        
        Args:
            path: 配置文件路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        _, ext = os.path.splitext(path)
        
        if ext.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return self._config.copy()
    
    def __str__(self) -> str:
        """
        转换为字符串
        
        Returns:
            配置字符串
        """
        return str(self._config)
    
    def __repr__(self) -> str:
        """
        转换为字符串表示
        
        Returns:
            配置字符串表示
        """
        return repr(self._config)