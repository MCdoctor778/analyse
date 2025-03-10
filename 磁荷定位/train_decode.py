"""
DECODE模型训练脚本 - 用于训练基于深度学习的单分子定位模型
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入我们的框架
from models import DECODEModel
from data import EmitterSimulator, GaussianPSF
from training import Trainer, DECODELoss, EarlyStopping, ModelCheckpoint, TensorBoardLogger
from utils import seed_everything, get_device, setup_logger, Config, Timer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DECODE模型训练')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=int, default=None, help='GPU设备ID')
    parser.add_argument('--output_dir', type=str, default='./output/decode', help='输出目录')
    parser.add_argument('--img_size', type=int, default=64, help='图像大小')
    parser.add_argument('--frame_window', type=int, default=5, help='帧窗口大小')
    parser.add_argument('--train_samples', type=int, default=1000, help='训练样本数量')
    parser.add_argument('--val_samples', type=int, default=100, help='验证样本数量')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(轮数)')
    return parser.parse_args()


class DECODEDataset(torch.utils.data.Dataset):
    """DECODE数据集包装器"""
    
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], {k: v[idx] for k, v in self.targets.items()}


class DECODETrainer:
    """DECODE模型训练器"""
    
    def __init__(self, model, loss_fn, optimizer, device, logger=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_detection_loss': [],
            'val_detection_loss': [],
            'train_coord_loss': [],
            'val_coord_loss': []
        }
    
    def train_epoch(self, train_loader):
        """训练一个轮次"""
        self.model.train()
        total_loss = 0
        total_detection_loss = 0
        total_coord_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc="训练")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            loss_dict = self.loss_fn(outputs, targets)
            loss = loss_dict['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 参数更新
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_detection_loss += loss_dict['detection_loss'].item()
            total_coord_loss += loss_dict['coord_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_detection_loss = total_detection_loss / len(train_loader)
        avg_coord_loss = total_coord_loss / len(train_loader)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_detection_loss'].append(avg_detection_loss)
        self.history['train_coord_loss'].append(avg_coord_loss)
        
        return avg_loss, avg_detection_loss, avg_coord_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_detection_loss = 0
        total_coord_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss_dict = self.loss_fn(outputs, targets)
                loss = loss_dict['loss']
                
                # 累计损失
                total_loss += loss.item()
                total_detection_loss += loss_dict['detection_loss'].item()
                total_coord_loss += loss_dict['coord_loss'].item()
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        avg_detection_loss = total_detection_loss / len(val_loader)
        avg_coord_loss = total_coord_loss / len(val_loader)
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_detection_loss'].append(avg_detection_loss)
        self.history['val_coord_loss'].append(avg_coord_loss)
        
        return avg_loss, avg_detection_loss, avg_coord_loss
    
    def fit(self, train_loader, val_loader, epochs, output_dir, save_interval=10):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练轮次
            train_loss, train_detection_loss, train_coord_loss = self.train_epoch(train_loader)
            self.logger.info(f"训练损失: {train_loss:.4f}, 检测损失: {train_detection_loss:.4f}, 坐标损失: {train_coord_loss:.4f}")
            
            # 验证
            val_loss, val_detection_loss, val_coord_loss = self.validate(val_loader)
            self.logger.info(f"验证损失: {val_loss:.4f}, 检测损失: {val_detection_loss:.4f}, 坐标损失: {val_coord_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(output_dir, 'best_model.pth'))
                self.logger.info(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
            
            # 定期保存模型
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
                self.logger.info(f"保存模型checkpoint: epoch_{epoch+1}")
                
                # 可视化结果
                self.visualize_results(val_loader, os.path.join(output_dir, f'vis_epoch_{epoch+1}.png'))
        
        # 保存训练历史
        self.plot_history(os.path.join(output_dir, 'training_history.png'))
        
        return self.history
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
    
    def plot_history(self, save_path):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))
        
        # 总损失
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('总损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 检测损失
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_detection_loss'], label='训练检测损失')
        plt.plot(self.history['val_detection_loss'], label='验证检测损失')
        plt.title('检测损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 坐标损失
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_coord_loss'], label='训练坐标损失')
        plt.plot(self.history['val_coord_loss'], label='验证坐标损失')
        plt.title('坐标损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_results(self, val_loader, save_path, num_samples=2):
        """可视化验证结果"""
        self.model.eval()
        
        # 获取一批验证数据
        inputs, targets = next(iter(val_loader))
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # 创建可视化图
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        for i in range(num_samples):
            if num_samples == 1:
                row_axes = axes
            else:
                row_axes = axes[i]
            
            # 中心帧
            center_frame = inputs.shape[1] // 2
            
            # 输入图像
            row_axes[0].imshow(inputs[i, center_frame].cpu().numpy(), cmap='gray')
            row_axes[0].set_title('输入图像')
            
            # 真实检测图
            row_axes[1].imshow(targets['detection_map'][i, 0].cpu().numpy(), cmap='jet')
            row_axes[1].set_title('真实检测图')
            
            # 预测检测图
            row_axes[2].imshow(outputs['detection_prob'][i, 0].cpu().numpy(), cmap='jet')
            row_axes[2].set_title('预测检测图')
            
            # 真实位置
            # 创建真实位置的散点图
            detection_mask = targets['detection_map'][i, 0].cpu().numpy() > 0.5
            y_indices, x_indices = np.where(detection_mask)
            
            if len(y_indices) > 0:
                x_coords = x_indices + targets['x_coord'][i, 0].cpu().numpy()[detection_mask]
                y_coords = y_indices + targets['y_coord'][i, 0].cpu().numpy()[detection_mask]
                
                row_axes[3].imshow(inputs[i, center_frame].cpu().numpy(), cmap='gray')
                row_axes[3].scatter(x_coords, y_coords, c='r', s=20, marker='x')
                row_axes[3].set_title('真实位置')
            else:
                row_axes[3].imshow(inputs[i, center_frame].cpu().numpy(), cmap='gray')
                row_axes[3].set_title('真实位置 (无)')
            
            # 预测位置
            # 创建预测位置的散点图
            pred_detection_mask = outputs['detection_prob'][i, 0].cpu().numpy() > 0.5
            pred_y_indices, pred_x_indices = np.where(pred_detection_mask)
            
            if len(pred_y_indices) > 0:
                pred_x_coords = pred_x_indices + outputs['x_coord'][i, 0].cpu().numpy()[pred_detection_mask]
                pred_y_coords = pred_y_indices + outputs['y_coord'][i, 0].cpu().numpy()[pred_detection_mask]
                
                row_axes[4].imshow(inputs[i, center_frame].cpu().numpy(), cmap='gray')
                row_axes[4].scatter(pred_x_coords, pred_y_coords, c='g', s=20, marker='+')
                row_axes[4].set_title('预测位置')
            else:
                row_axes[4].imshow(inputs[i, center_frame].cpu().numpy(), cmap='gray')
                row_axes[4].set_title('预测位置 (无)')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def custom_collate_fn(batch):
    """自定义整理函数"""
    inputs = torch.stack([item[0] for item in batch])
    targets = {k: torch.stack([item[1][k] for item in batch]) for k in batch[0][1].keys()}
    return inputs, targets


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('decode_training', log_file=os.path.join(args.output_dir, 'train.log'))
    logger.info(f'参数: {args}')
    
    # 设置随机种子
    seed_everything(args.seed)
    logger.info(f'设置随机种子: {args.seed}')
    
    # 获取设备
    device = get_device(args.device)
    logger.info(f'使用设备: {device}')
    
    # 创建计时器
    timer = Timer()
    
    # 创建PSF模拟器
    logger.info('创建PSF模拟器...')
    psf_model = GaussianPSF()
    
    # 创建发射体模拟器
    logger.info('创建发射体模拟器...')
    emitter_simulator = EmitterSimulator(
        img_size=(args.img_size, args.img_size),
        psf_model=psf_model
    )
    
    # 生成训练数据
    logger.info('生成训练数据...')
    timer.start()
    train_inputs, train_targets = emitter_simulator.generate_training_data(
        num_samples=args.train_samples,
        num_frames=args.frame_window,
        device=device
    )
    logger.info(f'生成训练数据完成，耗时: {timer.stop():.2f}秒')
    
    # 生成验证数据
    logger.info('生成验证数据...')
    timer.start()
    val_inputs, val_targets = emitter_simulator.generate_training_data(
        num_samples=args.val_samples,
        num_frames=args.frame_window,
        device=device
    )
    logger.info(f'生成验证数据完成，耗时: {timer.stop():.2f}秒')
    
    # 创建数据集
    train_dataset = DECODEDataset(train_inputs, train_targets)
    val_dataset = DECODEDataset(val_inputs, val_targets)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # 创建模型
    logger.info('创建DECODE模型...')
    model = DECODEModel(
        in_channels=1,
        frame_window=args.frame_window,
        param_channels=6,
        features=[64, 128, 256, 512],
        bilinear=True
    ).to(device)
    
    # 输出模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型总参数: {total_params:,}')
    logger.info(f'可训练参数: {trainable_params:,}')
    
    # 创建损失函数
    loss_fn = DECODELoss(
        detection_weight=1.0,
        coord_weight=1.0,
        z_weight=0.5,
        intensity_weight=0.2,
        background_weight=0.1,
        uncertainty_weight=0.1
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建训练器
    trainer = DECODETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        logger=logger
    )
    
    # 训练模型
    logger.info('开始训练...')
    timer.start()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        output_dir=args.output_dir,
        save_interval=args.save_interval
    )
    training_time = timer.stop()
    logger.info(f'训练完成，总耗时: {training_time:.2f}秒')
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    trainer.save_model(final_model_path)
    logger.info(f'最终模型已保存到: {final_model_path}')
    
    logger.info('训练完成!')


if __name__ == '__main__':
    main() 