"""
训练器类 - 用于模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import logging

from models.base import BaseModel
from .callbacks import Callback


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self,
                 model: BaseModel,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 callbacks: Optional[List[Callback]] = None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            loss_fn: 损失函数
            optimizer: 优化器
            device: 训练设备
            callbacks: 回调函数列表
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.callbacks = callbacks or []
        
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
        
        # 训练状态
        self.train_state = {
            'epoch': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_metrics': {},
            'val_metrics': {},
            'best_val_loss': float('inf'),
            'early_stop': False
        }
    
    def train_epoch(self, train_loader: DataLoader, metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            metrics: 评估指标字典
            
        Returns:
            训练结果字典
        """
        self.model.train()
        total_loss = 0.0
        metrics_results = {name: 0.0 for name in metrics.keys()} if metrics else {}
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {self.train_state['epoch']+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            inputs, targets = self._prepare_batch(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新损失
            total_loss += loss.item()
            
            # 计算指标
            if metrics:
                with torch.no_grad():
                    for name, metric_fn in metrics.items():
                        metrics_results[name] += metric_fn(outputs, targets).item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                **{name: value / (batch_idx + 1) for name, value in metrics_results.items()}
            })
            
            # 调用回调函数
            for callback in self.callbacks:
                callback.on_batch_end(self.train_state, batch_idx, {
                    'loss': loss.item(),
                    **{name: value / (batch_idx + 1) for name, value in metrics_results.items()}
                })
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {name: value / len(train_loader) for name, value in metrics_results.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader: DataLoader, metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            metrics: 评估指标字典
            
        Returns:
            验证结果字典
        """
        self.model.eval()
        total_loss = 0.0
        metrics_results = {name: 0.0 for name in metrics.keys()} if metrics else {}
        
        # 进度条
        pbar = tqdm(val_loader, desc=f"Epoch {self.train_state['epoch']+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                inputs, targets = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # 更新损失
                total_loss += loss.item()
                
                # 计算指标
                if metrics:
                    for name, metric_fn in metrics.items():
                        metrics_results[name] += metric_fn(outputs, targets).item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    **{name: value / (batch_idx + 1) for name, value in metrics_results.items()}
                })
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {name: value / len(val_loader) for name, value in metrics_results.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
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
    
    def fit(self, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            epochs: int = 10,
            metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            metrics: 评估指标字典
            
        Returns:
            训练历史记录
        """
        # 初始化历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            **(({f'train_{name}': [] for name in metrics.keys()} if metrics else {})),
            **(({f'val_{name}': [] for name in metrics.keys()} if metrics else {})),
            'best_val_loss': float('inf'),
            'early_stop': False
        }
        
        # 调用回调函数
        for callback in self.callbacks:
            callback.on_train_begin(self.train_state)
        
        # 训练循环
        for epoch in range(epochs):
            self.train_state['epoch'] = epoch
            
            # 调用回调函数
            for callback in self.callbacks:
                callback.on_epoch_begin(self.train_state)
            
            # 训练一个epoch
            start_time = time.time()
            train_results = self.train_epoch(train_loader, metrics)
            
            # 更新训练状态
            self.train_state['train_loss'] = train_results['loss']
            if metrics:
                self.train_state['train_metrics'] = {name: train_results[name] for name in metrics.keys()}
            
            # 验证
            if val_loader is not None:
                val_results = self.validate(val_loader, metrics)
                
                # 更新训练状态
                self.train_state['val_loss'] = val_results['loss']
                if metrics:
                    self.train_state['val_metrics'] = {name: val_results[name] for name in metrics.keys()}
                
                # 检查是否是最佳模型
                if val_results['loss'] < self.train_state['best_val_loss']:
                    self.train_state['best_val_loss'] = val_results['loss']
            
            # 计算epoch耗时
            epoch_time = time.time() - start_time
            
            # 更新历史记录
            history['train_loss'].append(train_results['loss'])
            if val_loader is not None:
                history['val_loss'].append(val_results['loss'])
            
            if metrics:
                for name in metrics.keys():
                    history[f'train_{name}'].append(train_results[name])
                    if val_loader is not None:
                        history[f'val_{name}'].append(val_results[name])
            
            # 打印epoch结果
            log_msg = f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_results['loss']:.4f}"
            if val_loader is not None:
                log_msg += f" - val_loss: {val_results['loss']:.4f}"
            
            if metrics:
                for name in metrics.keys():
                    log_msg += f" - train_{name}: {train_results[name]:.4f}"
                    if val_loader is not None:
                        log_msg += f" - val_{name}: {val_results[name]:.4f}"
            
            self.logger.info(log_msg)
            
            # 调用回调函数
            for callback in self.callbacks:
                callback.on_epoch_end(self.train_state)
            
            # 检查是否提前停止
            if self.train_state['early_stop']:
                self.logger.info(f"提前停止训练，在epoch {epoch+1}")
                break
        
        # 调用回调函数
        for callback in self.callbacks:
            callback.on_train_end(self.train_state)
        
        return history
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存训练器状态
        
        Args:
            path: 保存路径
            metadata: 额外的元数据
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备保存数据
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_state': self.train_state,
            'metadata': metadata or {}
        }
        
        # 保存
        torch.save(save_dict, path)
        self.logger.info(f"训练器状态已保存到 {path}")
    
    @classmethod
    def load(cls, 
             path: str, 
             model: BaseModel,
             loss_fn: nn.Module,
             device: Optional[torch.device] = None,
             callbacks: Optional[List[Callback]] = None) -> 'Trainer':
        """
        加载训练器状态
        
        Args:
            path: 加载路径
            model: 模型
            loss_fn: 损失函数
            device: 设备
            callbacks: 回调函数列表
            
        Returns:
            加载的训练器
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 加载保存的数据
        checkpoint = torch.load(path, map_location=device)
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # 创建优化器（需要先将模型参数移动到设备上）
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 创建训练器
        trainer = cls(model, loss_fn, optimizer, device, callbacks)
        
        # 加载训练状态
        trainer.train_state = checkpoint['train_state']
        
        return trainer, checkpoint.get('metadata', {})