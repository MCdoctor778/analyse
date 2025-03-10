"""
评估器类 - 用于模型评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import logging

from models.base import BaseModel
from .metrics import accuracy, precision, recall, f1_score, confusion_matrix


class Evaluator:
    """
    模型评估器
    """
    
    def __init__(self, 
                 model: BaseModel,
                 device: Optional[torch.device] = None,
                 metrics: Optional[Dict[str, Callable]] = None):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 评估设备
            metrics: 评估指标字典
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics or {
            'accuracy': accuracy
        }
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def evaluate(self, 
                 data_loader: DataLoader, 
                 loss_fn: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            loss_fn: 损失函数
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        # 初始化结果
        results = {name: 0.0 for name in self.metrics.keys()}
        if loss_fn is not None:
            results['loss'] = 0.0
        
        # 初始化混淆矩阵（如果需要）
        conf_matrix = None
        
        # 进度条
        pbar = tqdm(data_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                inputs, targets = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)
                    results['loss'] += loss.item()
                
                # 计算指标
                for name, metric_fn in self.metrics.items():
                    results[name] += metric_fn(outputs, targets).item()
                
                # 更新混淆矩阵（如果需要）
                if 'confusion_matrix' in self.metrics:
                    if conf_matrix is None:
                        num_classes = outputs.size(1)
                        conf_matrix = torch.zeros(num_classes, num_classes, device=self.device)
                    
                    batch_conf_matrix = confusion_matrix(outputs, targets, num_classes)
                    conf_matrix += batch_conf_matrix
        
        # 计算平均值
        for name in results.keys():
            results[name] /= len(data_loader)
        
        # 添加混淆矩阵（如果有）
        if conf_matrix is not None:
            results['confusion_matrix'] = conf_matrix.cpu().numpy()
        
        return results
    
    def _prepare_batch(self, batch: Union[Tuple, List, Dict, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备批次数据
        
        Args:
            batch: 批次数据
            
        Returns:
            输入和目标张量
        """
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            return inputs, targets
        elif isinstance(batch, dict) and 'inputs' in batch and 'targets' in batch:
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)
            return inputs, targets
        else:
            raise ValueError("不支持的批次格式。请提供(inputs, targets)元组或包含'inputs'和'targets'键的字典。")
    
    def predict(self, 
                data_loader: DataLoader, 
                return_targets: bool = False) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        使用模型进行预测
        
        Args:
            data_loader: 数据加载器
            return_targets: 是否返回目标值
            
        Returns:
            预测结果列表，如果return_targets为True，则返回(预测结果列表, 目标值列表)
        """
        self.model.eval()
        
        # 初始化结果
        predictions = []
        targets_list = [] if return_targets else None
        
        # 进度条
        pbar = tqdm(data_loader, desc="Predicting")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    inputs = inputs.to(self.device)
                    if return_targets:
                        targets = targets.to(self.device)
                        targets_list.append(targets.cpu())
                elif isinstance(batch, dict) and 'inputs' in batch:
                    inputs = batch['inputs'].to(self.device)
                    if return_targets and 'targets' in batch:
                        targets = batch['targets'].to(self.device)
                        targets_list.append(targets.cpu())
                else:
                    inputs = batch.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 保存预测结果
                predictions.append(outputs.cpu())
        
        # 合并预测结果
        predictions = torch.cat(predictions, dim=0)
        
        if return_targets:
            targets_list = torch.cat(targets_list, dim=0)
            return predictions, targets_list
        else:
            return predictions
    
    def evaluate_and_print(self, 
                           data_loader: DataLoader, 
                           loss_fn: Optional[nn.Module] = None,
                           dataset_name: str = "Test") -> Dict[str, float]:
        """
        评估模型并打印结果
        
        Args:
            data_loader: 数据加载器
            loss_fn: 损失函数
            dataset_name: 数据集名称
            
        Returns:
            评估结果字典
        """
        # 开始计时
        start_time = time.time()
        
        # 评估模型
        results = self.evaluate(data_loader, loss_fn)
        
        # 计算耗时
        eval_time = time.time() - start_time
        
        # 打印结果
        self.logger.info(f"{dataset_name}集评估结果 ({eval_time:.2f}s):")
        
        for name, value in results.items():
            if name != 'confusion_matrix':
                self.logger.info(f"  {name}: {value:.4f}")
        
        # 如果有混淆矩阵，打印它
        if 'confusion_matrix' in results:
            self.logger.info(f"  混淆矩阵:\n{results['confusion_matrix']}")
        
        return results