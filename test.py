"""
MNIST分类示例 - 展示如何使用我们的深度学习框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的框架
from models import CNNModel
from training import Trainer, get_loss_function, EarlyStopping, ModelCheckpoint, TensorBoardLogger
from evaluation import Evaluator, accuracy, precision, recall, f1_score
from visualization import plot_loss_curves, plot_metrics, plot_confusion_matrix, plot_predictions
from utils import seed_everything, get_device, setup_logger, Config, Timer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST分类示例')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=int, default=None, help='GPU设备ID')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('mnist_example', log_file=os.path.join(args.output_dir, 'train.log'))
    logger.info(f'参数: {args}')
    
    # 设置随机种子
    seed_everything(args.seed)
    logger.info(f'设置随机种子: {args.seed}')
    
    # 获取设备
    device = get_device(args.device)
    logger.info(f'使用设备: {device}')
    
    # 创建计时器
    timer = Timer()
    
    # 准备数据
    logger.info('准备数据...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 创建模型
    logger.info('创建模型...')
    model = CNNModel(in_channels=1, num_classes=10, hidden_channels=[32, 64])
    logger.info(model.summary())
    
    # 定义损失函数和优化器
    loss_fn = get_loss_function('ce')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'best_model.pth'),
            monitor='val_loss',
            save_best_only=True
        ),
        TensorBoardLogger(os.path.join(args.output_dir, 'logs'))
    ]
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, device, callbacks)
    
    # 定义评估指标
    metrics = {
        'accuracy': accuracy,
        'precision': lambda x, y: precision(x, y, average='macro'),
        'recall': lambda x, y: recall(x, y, average='macro'),
        'f1': lambda x, y: f1_score(x, y, average='macro')
    }
    
    # 训练模型
    logger.info('开始训练...')
    timer.start()
    history = trainer.fit(train_loader, test_loader, epochs=args.epochs, metrics=metrics)
    training_time = timer.stop()
    logger.info(f'训练完成，耗时: {training_time:.2f}秒')
    
    # 评估模型
    logger.info('评估模型...')
    evaluator = Evaluator(model, device, metrics)
    results = evaluator.evaluate_and_print(test_loader, loss_fn)
    
    # 可视化结果
    logger.info('可视化结果...')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        title='损失曲线',
        save_path=os.path.join(args.output_dir, 'loss_curves.png')
    )
    
    # 绘制指标曲线
    metrics_dict = {
        '训练准确率': history['train_accuracy'],
        '验证准确率': history['val_accuracy'],
        '训练F1分数': history['train_f1'],
        '验证F1分数': history['val_f1']
    }
    plot_metrics(
        metrics_dict,
        title='评估指标',
        save_path=os.path.join(args.output_dir, 'metrics.png')
    )
    
    # 获取一些测试样本进行预测
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 绘制预测结果
    plot_predictions(
        images[:10],
        labels[:10],
        predicted[:10].cpu(),
        class_names=[str(i) for i in range(10)],
        num_images=10,
        figsize=(20, 4),
        save_path=os.path.join(args.output_dir, 'predictions.png')
    )
    
    # 计算混淆矩阵
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=[str(i) for i in range(10)],
        normalize=True,
        title='混淆矩阵',
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    logger.info(f'所有结果已保存到: {args.output_dir}')


if __name__ == '__main__':
    main()