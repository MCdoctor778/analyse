"""
绘图模块 - 提供各种绘图函数
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = 'Loss Curves',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        matplotlib图表对象
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='验证损失')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_metrics(
    metrics_dict: Dict[str, List[float]],
    title: str = 'Metrics',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制指标曲线
    
    Args:
        metrics_dict: 指标字典，键为指标名称，值为指标值列表
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        matplotlib图表对象
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(list(metrics_dict.values())[0]) + 1)
    
    for name, values in metrics_dict.items():
        plt.plot(epochs, values, '-', label=name)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 是否归一化
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        save_path: 保存路径
        
    Returns:
        matplotlib图表对象
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算混淆矩阵
    cm = sk_confusion_matrix(y_true, y_pred)
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_predictions(
    images: torch.Tensor,
    true_labels: Optional[torch.Tensor] = None,
    pred_labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    num_images: int = 5,
    figsize: Tuple[int, int] = (15, 3),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制预测结果
    
    Args:
        images: 图像张量 [N, C, H, W]
        true_labels: 真实标签 [N]
        pred_labels: 预测标签 [N]
        class_names: 类别名称列表
        num_images: 要显示的图像数量
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        matplotlib图表对象
    """
    # 转换为numpy数组
    images = images.cpu().numpy()
    if true_labels is not None:
        true_labels = true_labels.cpu().numpy()
    if pred_labels is not None:
        pred_labels = pred_labels.cpu().numpy()
    
    # 限制图像数量
    num_images = min(num_images, len(images))
    
    # 创建图表
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        
        # 转置图像通道
        img = np.transpose(images[i], (1, 2, 0))
        
        # 归一化图像
        img = (img - img.min()) / (img.max() - img.min())
        
        # 显示图像
        ax.imshow(img)
        
        # 设置标题
        title = ""
        if true_labels is not None and class_names is not None:
            title += f"真实: {class_names[true_labels[i]]}\n"
        elif true_labels is not None:
            title += f"真实: {true_labels[i]}\n"
        
        if pred_labels is not None and class_names is not None:
            title += f"预测: {class_names[pred_labels[i]]}"
        elif pred_labels is not None:
            title += f"预测: {pred_labels[i]}"
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_segmentation_results(
    images: torch.Tensor,
    true_masks: Optional[torch.Tensor] = None,
    pred_masks: Optional[torch.Tensor] = None,
    num_images: int = 3,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制分割结果
    
    Args:
        images: 图像张量 [N, C, H, W]
        true_masks: 真实掩码 [N, C, H, W]
        pred_masks: 预测掩码 [N, C, H, W]
        num_images: 要显示的图像数量
        figsize: 图表大小
        alpha: 掩码透明度
        save_path: 保存路径
        
    Returns:
        matplotlib图表对象
    """
    # 转换为numpy数组
    images = images.cpu().numpy()
    if true_masks is not None:
        true_masks = true_masks.cpu().numpy()
    if pred_masks is not None:
        pred_masks = pred_masks.cpu().numpy()
    
    # 限制图像数量
    num_images = min(num_images, len(images))
    
    # 创建图表
    fig, axes = plt.subplots(num_images, 3, figsize=figsize)
    
    for i in range(num_images):
        # 转置图像通道
        img = np.transpose(images[i], (1, 2, 0))
        
        # 归一化图像
        img = (img - img.min()) / (img.max() - img.min())
        
        # 显示原始图像
        if num_images > 1:
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('原始图像')
            axes[i, 0].axis('off')
            
            # 显示真实掩码
            if true_masks is not None:
                axes[i, 1].imshow(img)
                mask = np.transpose(true_masks[i], (1, 2, 0))
                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                axes[i, 1].imshow(mask, alpha=alpha, cmap='jet')
                axes[i, 1].set_title('真实掩码')
                axes[i, 1].axis('off')
            
            # 显示预测掩码
            if pred_masks is not None:
                axes[i, 2].imshow(img)
                mask = np.transpose(pred_masks[i], (1, 2, 0))
                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                axes[i, 2].imshow(mask, alpha=alpha, cmap='jet')
                axes[i, 2].set_title('预测掩码')
                axes[i, 2].axis('off')
        else:
            axes[0].imshow(img)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 显示真实掩码
            if true_masks is not None:
                axes[1].imshow(img)
                mask = np.transpose(true_masks[i], (1, 2, 0))
                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                axes[1].imshow(mask, alpha=alpha, cmap='jet')
                axes[1].set_title('真实掩码')
                axes[1].axis('off')
            
            # 显示预测掩码
            if pred_masks is not None:
                axes[2].imshow(img)
                mask = np.transpose(pred_masks[i], (1, 2, 0))
                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                axes[2].imshow(mask, alpha=alpha, cmap='jet')
                axes[2].set_title('预测掩码')
                axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig