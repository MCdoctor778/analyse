"""
DeepLearningFramework - 基于DECODE架构的深度学习框架

这个框架提供了一套完整的深度学习工具，包括模型定义、训练、评估和可视化。
"""

import torch
import numpy as np
import os
import sys

__version__ = '0.1.0'
__author__ = 'wangqi'

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保所有子模块可以被导入
sys.path.append(_ROOT_DIR)

# 导入主要模块
from . import models
from . import training
from . import evaluation
from . import utils
from . import visualization
from . import data

# 检查CUDA可用性
_CUDA_AVAILABLE = torch.cuda.is_available()

def cuda_available() -> bool:
    """
    返回CUDA是否可用
    
    Returns:
        bool: CUDA可用性状态
    """
    return _CUDA_AVAILABLE