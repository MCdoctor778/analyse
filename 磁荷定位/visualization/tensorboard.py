"""
TensorBoard可视化器 - 用于在TensorBoard中可视化模型训练过程
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class TensorboardVisualizer:
    """
    TensorBoard可视化器
    """
    
    def __init__(self, log_dir: str):
        """
        初始化TensorBoard可视化器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
    
    def close(self):
        """关闭SummaryWriter"""
        self.writer.close()
    
    def add_scalar(self, tag: str, value: float, step: int):
        """
        添加标量
        
        Args:
            tag: 标签
            value: 值
            step: 步骤
        """
        self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        添加多个标量
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-值字典
            step: 步骤
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        添加直方图
        
        Args:
            tag: 标签
            values: 值
            step: 步骤
        """
        self.writer.add_histogram(tag, values, step)
    
    def add_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """
        添加图像
        
        Args:
            tag: 标签
            img_tensor: 图像张量 [C, H, W]
            step: 步骤
        """
        self.writer.add_image(tag, img_tensor, step)
    
    def add_images(self, tag: str, img_tensor: torch.Tensor, step: int):
        """
        添加多个图像
        
        Args:
            tag: 标签
            img_tensor: 图像张量 [N, C, H, W]
            step: 步骤
        """
        self.writer.add_images(tag, img_tensor, step)
    
    def add_figure(self, tag: str, figure: plt.Figure, step: int):
        """
        添加matplotlib图表
        
        Args:
            tag: 标签
            figure: matplotlib图表
            step: 步骤
        """
        self.writer.add_figure(tag, figure, step)
    
    def add_graph(self, model: nn.Module, input_to_model: torch.Tensor):
        """
        添加模型图
        
        Args:
            model: 模型
            input_to_model: 模型输入
        """
        self.writer.add_graph(model, input_to_model)
    
    def add_pr_curve(self, tag: str, labels: torch.Tensor, predictions: torch.Tensor, step: int):
        """
        添加PR曲线
        
        Args:
            tag: 标签
            labels: 真实标签
            predictions: 预测概率
            step: 步骤
        """
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def add_embedding(self, 
                      mat: torch.Tensor, 
                      metadata: Optional[List[str]] = None,
                      label_img: Optional[torch.Tensor] = None,
                      tag: str = 'default'):
        """
        添加嵌入
        
        Args:
            mat: 嵌入矩阵
            metadata: 元数据
            label_img: 标签图像
            tag: 标签
        """
        self.writer.add_embedding(mat, metadata, label_img, tag)
    
    def add_confusion_matrix(self, 
                             tag: str, 
                             cm: np.ndarray, 
                             class_names: List[str],
                             step: int,
                             normalize: bool = False):
        """
        添加混淆矩阵
        
        Args:
            tag: 标签
            cm: 混淆矩阵
            class_names: 类别名称
            step: 步骤
            normalize: 是否归一化
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 添加图表
        self.add_figure(tag, plt.gcf(), step)
    
    def add_text(self, tag: str, text_string: str, step: int):
        """
        添加文本
        
        Args:
            tag: 标签
            text_string: 文本
            step: 步骤
        """
        self.writer.add_text(tag, text_string, step)
    
    def add_image_with_boxes(self, 
                             tag: str, 
                             img_tensor: torch.Tensor, 
                             box_tensor: torch.Tensor,
                             step: int,
                             labels: Optional[List[str]] = None):
        """
        添加带边界框的图像
        
        Args:
            tag: 标签
            img_tensor: 图像张量 [C, H, W]
            box_tensor: 边界框张量 [N, 4]，格式为[x1, y1, x2, y2]
            step: 步骤
            labels: 标签列表
        """
        self.writer.add_image_with_boxes(tag, img_tensor, box_tensor, step, labels)
    
    def add_mesh(self, 
                 tag: str, 
                 vertices: torch.Tensor, 
                 colors: Optional[torch.Tensor] = None,
                 faces: Optional[torch.Tensor] = None,
                 step: int = 0):
        """
        添加3D网格
        
        Args:
            tag: 标签
            vertices: 顶点张量 [N, 3]
            colors: 颜色张量 [N, 3]
            faces: 面张量 [F, 3]
            step: 步骤
        """
        self.writer.add_mesh(tag, vertices, colors, faces, step)
    
    def add_hparams(self, 
                    hparam_dict: Dict[str, Union[bool, str, float, int]],
                    metric_dict: Dict[str, float]):
        """
        添加超参数
        
        Args:
            hparam_dict: 超参数字典
            metric_dict: 指标字典
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def log_model_gradients(self, model: nn.Module, step: int, prefix: str = 'gradients/'):
        """
        记录模型梯度
        
        Args:
            model: 模型
            step: 步骤
            prefix: 前缀
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.add_histogram(f"{prefix}{name}", param.grad, step)
    
    def log_model_weights(self, model: nn.Module, step: int, prefix: str = 'weights/'):
        """
        记录模型权重
        
        Args:
            model: 模型
            step: 步骤
            prefix: 前缀
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.add_histogram(f"{prefix}{name}", param, step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """
        记录学习率
        
        Args:
            optimizer: 优化器
            step: 步骤
        """
        for i, param_group in enumerate(optimizer.param_groups):
            self.add_scalar(f"learning_rate/group{i}", param_group['lr'], step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = 'metrics/'):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤
            prefix: 前缀
        """
        for name, value in metrics.items():
            self.add_scalar(f"{prefix}{name}", value, step)
    
    def log_batch_predictions(self, 
                              tag: str,
                              images: torch.Tensor,
                              true_labels: torch.Tensor,
                              pred_labels: torch.Tensor,
                              step: int,
                              class_names: Optional[List[str]] = None,
                              max_images: int = 8):
        """
        记录批次预测结果
        
        Args:
            tag: 标签
            images: 图像张量 [N, C, H, W]
            true_labels: 真实标签 [N]
            pred_labels: 预测标签 [N]
            step: 步骤
            class_names: 类别名称
            max_images: 最大图像数量
        """
        # 限制图像数量
        n = min(images.size(0), max_images)
        
        # 创建图表
        fig, axes = plt.subplots(1, n, figsize=(15, 3))
        
        for i in range(n):
            ax = axes[i] if n > 1 else axes
            
            # 转置图像通道
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            
            # 归一化图像
            img = (img - img.min()) / (img.max() - img.min())
            
            # 显示图像
            ax.imshow(img)
            
            # 设置标题
            true_label = true_labels[i].item()
            pred_label = pred_labels[i].item()
            
            if class_names is not None:
                true_name = class_names[true_label]
                pred_name = class_names[pred_label]
                title = f"真实: {true_name}\n预测: {pred_name}"
            else:
                title = f"真实: {true_label}\n预测: {pred_label}"
            
            # 设置颜色
            color = 'green' if true_label == pred_label else 'red'
            
            ax.set_title(title, color=color)
            ax.axis('off')
        
        plt.tight_layout()
        
        # 添加图表
        self.add_figure(tag, fig, step)
        plt.close(fig)
    
    def log_segmentation_predictions(self,
                                     tag: str,
                                     images: torch.Tensor,
                                     true_masks: torch.Tensor,
                                     pred_masks: torch.Tensor,
                                     step: int,
                                     max_images: int = 4,
                                     alpha: float = 0.5):
        """
        记录分割预测结果
        
        Args:
            tag: 标签
            images: 图像张量 [N, C, H, W]
            true_masks: 真实掩码 [N, C, H, W]
            pred_masks: 预测掩码 [N, C, H, W]
            step: 步骤
            max_images: 最大图像数量
            alpha: 掩码透明度
        """
        # 限制图像数量
        n = min(images.size(0), max_images)
        
        # 创建图表
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        
        for i in range(n):
            # 转置图像通道
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            
            # 归一化图像
            img = (img - img.min()) / (img.max() - img.min())
            
            # 获取掩码
            true_mask = true_masks[i].cpu().numpy()
            pred_mask = pred_masks[i].cpu().numpy()
            
            if true_mask.shape[0] == 1:
                true_mask = true_mask[0]
            else:
                true_mask = true_mask.transpose(1, 2, 0)
            
            if pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
            else:
                pred_mask = pred_mask.transpose(1, 2, 0)
            
            # 显示图像和掩码
            if n > 1:
                # 原始图像
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('原始图像')
                axes[i, 0].axis('off')
                
                # 真实掩码
                axes[i, 1].imshow(img)
                axes[i, 1].imshow(true_mask, alpha=alpha, cmap='jet')
                axes[i, 1].set_title('真实掩码')
                axes[i, 1].axis('off')
                
                # 预测掩码
                axes[i, 2].imshow(img)
                axes[i, 2].imshow(pred_mask, alpha=alpha, cmap='jet')
                axes[i, 2].set_title('预测掩码')
                axes[i, 2].axis('off')
            else:
                # 原始图像
                axes[0].imshow(img)
                axes[0].set_title('原始图像')
                axes[0].axis('off')
                
                # 真实掩码
                axes[1].imshow(img)
                axes[1].imshow(true_mask, alpha=alpha, cmap='jet')
                axes[1].set_title('真实掩码')
                axes[1].axis('off')
                
                # 预测掩码
                axes[2].imshow(img)
                axes[2].imshow(pred_mask, alpha=alpha, cmap='jet')
                axes[2].set_title('预测掩码')
                axes[2].axis('off')
        
        plt.tight_layout()
        
        # 添加图表
        self.add_figure(tag, fig, step)
        plt.close(fig)