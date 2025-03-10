"""
训练回调函数 - 用于在训练过程中执行特定操作
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from torch.utils.tensorboard import SummaryWriter


class Callback:
    """回调函数基类"""
    
    def on_train_begin(self, train_state: Dict[str, Any]) -> None:
        """
        训练开始时调用
        
        Args:
            train_state: 训练状态
        """
        pass
    
    def on_train_end(self, train_state: Dict[str, Any]) -> None:
        """
        训练结束时调用
        
        Args:
            train_state: 训练状态
        """
        pass
    
    def on_epoch_begin(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch开始时调用
        
        Args:
            train_state: 训练状态
        """
        pass
    
    def on_epoch_end(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch结束时调用
        
        Args:
            train_state: 训练状态
        """
        pass
    
    def on_batch_begin(self, train_state: Dict[str, Any], batch_idx: int) -> None:
        """
        每个batch开始时调用
        
        Args:
            train_state: 训练状态
            batch_idx: batch索引
        """
        pass
    
    def on_batch_end(self, train_state: Dict[str, Any], batch_idx: int, logs: Dict[str, float]) -> None:
        """
        每个batch结束时调用
        
        Args:
            train_state: 训练状态
            batch_idx: batch索引
            logs: 日志信息
        """
        pass


class EarlyStopping(Callback):
    """早停回调函数"""
    
    def __init__(self, 
                 monitor: str = 'val_loss', 
                 min_delta: float = 0.0, 
                 patience: int = 10,
                 verbose: bool = True,
                 mode: str = 'min'):
        """
        初始化早停回调函数
        
        Args:
            monitor: 监控的指标
            min_delta: 最小变化阈值
            patience: 容忍的epoch数
            verbose: 是否打印信息
            mode: 'min'或'max'，指标是越小越好还是越大越好
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.wait = 0
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch结束时检查是否应该早停
        
        Args:
            train_state: 训练状态
        """
        # 获取当前指标值
        current = None
        if self.monitor == 'val_loss':
            current = train_state['val_loss']
        elif self.monitor == 'train_loss':
            current = train_state['train_loss']
        elif self.monitor.startswith('val_') and self.monitor[4:] in train_state['val_metrics']:
            current = train_state['val_metrics'][self.monitor[4:]]
        elif self.monitor.startswith('train_') and self.monitor[6:] in train_state['train_metrics']:
            current = train_state['train_metrics'][self.monitor[6:]]
        
        if current is None:
            self.logger.warning(f"早停回调函数无法监控指标 {self.monitor}，因为它不存在于训练状态中")
            return
        
        # 检查是否有改进
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        
        if improved:
            # 有改进，重置等待计数
            self.best = current
            self.wait = 0
        else:
            # 没有改进，增加等待计数
            self.wait += 1
            if self.wait >= self.patience:
                # 达到容忍限制，触发早停
                train_state['early_stop'] = True
                if self.verbose:
                    self.logger.info(f"早停触发，{self.monitor} 在 {self.patience} 个epoch内没有改进")


class ModelCheckpoint(Callback):
    """模型检查点回调函数"""
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_loss',
                 verbose: bool = True,
                 save_best_only: bool = True,
                 mode: str = 'min',
                 save_weights_only: bool = False):
        """
        初始化模型检查点回调函数
        
        Args:
            filepath: 保存路径
            monitor: 监控的指标
            verbose: 是否打印信息
            save_best_only: 是否只保存最佳模型
            mode: 'min'或'max'，指标是越小越好还是越大越好
            save_weights_only: 是否只保存权重
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.logger = logging.getLogger(__name__)
        
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch结束时保存模型
        
        Args:
            train_state: 训练状态
        """
        # 获取当前指标值
        current = None
        if self.monitor == 'val_loss':
            current = train_state['val_loss']
        elif self.monitor == 'train_loss':
            current = train_state['train_loss']
        elif self.monitor.startswith('val_') and self.monitor[4:] in train_state['val_metrics']:
            current = train_state['val_metrics'][self.monitor[4:]]
        elif self.monitor.startswith('train_') and self.monitor[6:] in train_state['train_metrics']:
            current = train_state['train_metrics'][self.monitor[6:]]
        
        if current is None:
            self.logger.warning(f"模型检查点回调函数无法监控指标 {self.monitor}，因为它不存在于训练状态中")
            return
        
        # 检查是否应该保存模型
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best
            
            if improved:
                if self.verbose:
                    self.logger.info(f"Epoch {train_state['epoch']+1}: {self.monitor} 改进从 {self.best:.4f} 到 {current:.4f}，保存模型到 {self.filepath}")
                self.best = current
                self._save_model(train_state)
        else:
            # 每个epoch都保存
            epoch_filepath = self.filepath.format(epoch=train_state['epoch']+1, **{self.monitor: current})
            if self.verbose:
                self.logger.info(f"Epoch {train_state['epoch']+1}: 保存模型到 {epoch_filepath}")
            self._save_model(train_state, epoch_filepath)
    
    def _save_model(self, train_state, filepath: Optional[str] = None) -> None:
        """
        保存模型
        
        Args:
            train_state: 训练状态
            filepath: 保存路径
        """
        from models.base import BaseModel  # 使用绝对导入
        
        if filepath is None:
            filepath = self.filepath
        
        # 添加调试信息
        print(f"训练状态类型: {type(train_state)}")
        if isinstance(train_state, dict):
            print(f"训练状态键: {train_state.keys()}")
        else:
            print(f"训练状态属性: {dir(train_state)}")
        
        # 获取模型 - 同时支持字典和对象
        model = None
        if isinstance(train_state, dict):
            model = train_state.get('model')
        else:
            model = getattr(train_state, 'model', None)
        
        if model is None:
            self.logger.warning("无法保存模型，因为训练状态中没有模型")
            return
        
        # 保存模型
        if self.save_weights_only:
            torch.save(model.state_dict(), filepath)
        else:
            if isinstance(model, BaseModel):
                # 获取元数据 - 同时支持字典和对象
                metadata = {}
                if isinstance(train_state, dict):
                    metadata = {
                        'epoch': train_state.get('epoch', 0),
                        'train_loss': train_state.get('train_loss', 0),
                        'val_loss': train_state.get('val_loss'),
                        'train_metrics': train_state.get('train_metrics', {}),
                        'val_metrics': train_state.get('val_metrics', {})
                    }
                else:
                    metadata = {
                        'epoch': getattr(train_state, 'epoch', 0),
                        'train_loss': getattr(train_state, 'train_loss', 0),
                        'val_loss': getattr(train_state, 'val_loss', None),
                        'train_metrics': getattr(train_state, 'train_metrics', {}),
                        'val_metrics': getattr(train_state, 'val_metrics', {})
                    }
                
                model.save(filepath, metadata=metadata)
            else:
                # 创建保存字典 - 同时支持字典和对象
                save_dict = {'model_state_dict': model.state_dict()}
                
                if isinstance(train_state, dict):
                    save_dict.update({
                        'epoch': train_state.get('epoch', 0),
                        'train_loss': train_state.get('train_loss', 0),
                        'val_loss': train_state.get('val_loss'),
                        'train_metrics': train_state.get('train_metrics', {}),
                        'val_metrics': train_state.get('val_metrics', {})
                    })
                else:
                    save_dict.update({
                        'epoch': getattr(train_state, 'epoch', 0),
                        'train_loss': getattr(train_state, 'train_loss', 0),
                        'val_loss': getattr(train_state, 'val_loss', None),
                        'train_metrics': getattr(train_state, 'train_metrics', {}),
                        'val_metrics': getattr(train_state, 'val_metrics', {})
                    })
                
                torch.save(save_dict, filepath)


class LearningRateScheduler(Callback):
    """学习率调度器回调函数"""
    
    def __init__(self, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 monitor: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化学习率调度器回调函数
        
        Args:
            scheduler: PyTorch学习率调度器
            monitor: 监控的指标（对于ReduceLROnPlateau）
            verbose: 是否打印信息
        """
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch结束时调整学习率
        
        Args:
            train_state: 训练状态
        """
        # 获取当前指标值（对于ReduceLROnPlateau）
        current = None
        if self.monitor is not None:
            if self.monitor == 'val_loss':
                current = train_state['val_loss']
            elif self.monitor == 'train_loss':
                current = train_state['train_loss']
            elif self.monitor.startswith('val_') and self.monitor[4:] in train_state['val_metrics']:
                current = train_state['val_metrics'][self.monitor[4:]]
            elif self.monitor.startswith('train_') and self.monitor[6:] in train_state['train_metrics']:
                current = train_state['train_metrics'][self.monitor[6:]]
        
        # 调整学习率
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if current is None:
                self.logger.warning(f"学习率调度器无法监控指标 {self.monitor}，因为它不存在于训练状态中")
                return
            self.scheduler.step(current)
        else:
            self.scheduler.step()
        
        # 打印新的学习率
        if self.verbose:
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                self.logger.info(f"Epoch {train_state['epoch']+1}: 学习率设置为 {param_group['lr']:.6f} (参数组 {i})")


class TensorBoardLogger(Callback):
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: 日志目录
        """
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(self, train_state: Dict[str, Any]) -> None:
        """
        训练开始时初始化SummaryWriter
        
        Args:
            train_state: 训练状态
        """
        self.writer = SummaryWriter(self.log_dir)
    
    def on_train_end(self, train_state: Dict[str, Any]) -> None:
        """
        训练结束时关闭SummaryWriter
        
        Args:
            train_state: 训练状态
        """
        if self.writer is not None:
            self.writer.close()
    
    def on_epoch_end(self, train_state: Dict[str, Any]) -> None:
        """
        每个epoch结束时记录指标
        
        Args:
            train_state: 训练状态
        """
        if self.writer is None:
            return
        
        epoch = train_state['epoch']
        
        # 记录损失
        self.writer.add_scalar('Loss/train', train_state['train_loss'], epoch)
        if 'val_loss' in train_state:
            self.writer.add_scalar('Loss/val', train_state['val_loss'], epoch)
        
        # 记录指标
        for name, value in train_state.get('train_metrics', {}).items():
            self.writer.add_scalar(f'Metrics/{name}/train', value, epoch)
        
        for name, value in train_state.get('val_metrics', {}).items():
            self.writer.add_scalar(f'Metrics/{name}/val', value, epoch)
        
        # 记录学习率
        if 'optimizer' in train_state:
            for i, param_group in enumerate(train_state['optimizer'].param_groups):
                self.writer.add_scalar(f'LearningRate/group{i}', param_group['lr'], epoch)