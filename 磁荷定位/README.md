# 深度学习框架

基于DECODE架构重构的深度学习框架，提供了一套完整的深度学习工具，包括模型定义、训练、评估和可视化。

## 项目结构

```
deep-learning-framework/
├── models/                  # 模型定义
│   ├── base.py              # 基础模型类
│   ├── cnn.py               # CNN模型
│   ├── unet.py              # U-Net模型
│   ├── transformer.py       # Transformer模型
│   └── __init__.py
├── training/                # 训练模块
│   ├── trainer.py           # 训练器
│   ├── callbacks.py         # 回调函数
│   ├── losses.py            # 损失函数
│   └── __init__.py
├── data/                    # 数据模块
│   ├── dataset.py           # 数据集类
│   ├── transforms.py        # 数据转换
│   ├── dataloader.py        # 数据加载器
│   └── __init__.py
├── evaluation/              # 评估模块
│   ├── metrics.py           # 评估指标
│   ├── evaluator.py         # 评估器
│   └── __init__.py
├── visualization/           # 可视化模块
│   ├── plots.py             # 绘图函数
│   ├── tensorboard.py       # TensorBoard可视化器
│   └── __init__.py
└── __init__.py              # 主模块
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/deep-learning-framework.git
cd deep-learning-framework

# 安装依赖
pip install -r requirements.txt
```

## 使用示例

### 1. 图像分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 导入框架
from deep_learning_framework import models
from deep_learning_framework.training import Trainer, get_loss_function
from deep_learning_framework.data import get_transforms, get_dataloader
from deep_learning_framework.evaluation import Evaluator
from deep_learning_framework.visualization import plot_loss_curves

# 准备数据
transform = get_transforms('classification', input_size=(32, 32))
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform['train'])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform['val'])

train_loader = get_dataloader(train_dataset, batch_size=64)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

# 创建模型
model = models.CNNModel(in_channels=3, num_classes=10)

# 定义损失函数和优化器
loss_fn = get_loss_function('ce')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建训练器
trainer = Trainer(model, loss_fn, optimizer)

# 训练模型
history = trainer.fit(train_loader, test_loader, epochs=10)

# 评估模型
evaluator = Evaluator(model)
results = evaluator.evaluate_and_print(test_loader, loss_fn)

# 可视化结果
plot_loss_curves(history['train_loss'], history['val_loss'])
```

### 2. 图像分割

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 导入框架
from deep_learning_framework import models
from deep_learning_framework.training import Trainer, get_loss_function
from deep_learning_framework.data import ImageDataset, get_transforms, get_dataloader
from deep_learning_framework.evaluation import Evaluator, dice_coefficient
from deep_learning_framework.visualization import plot_segmentation_results

# 准备数据
transform = get_transforms('segmentation', input_size=(512, 512))
train_dataset = ImageDataset(image_paths=train_images, labels=train_masks, transform=transform['train'])
val_dataset = ImageDataset(image_paths=val_images, labels=val_masks, transform=transform['val'])

train_loader = get_dataloader(train_dataset, batch_size=8)
val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False)

# 创建模型
model = models.UNet(in_channels=3, out_channels=1)

# 定义损失函数和优化器
loss_fn = get_loss_function('dice')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建训练器
trainer = Trainer(model, loss_fn, optimizer)

# 训练模型
history = trainer.fit(train_loader, val_loader, epochs=50, metrics={'dice': dice_coefficient})

# 评估模型
evaluator = Evaluator(model, metrics={'dice': dice_coefficient})
results = evaluator.evaluate_and_print(val_loader, loss_fn)

# 可视化结果
images, masks = next(iter(val_loader))
with torch.no_grad():
    pred_masks = model(images)
plot_segmentation_results(images, masks, pred_masks)
```

### 3. 序列处理

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 导入框架
from deep_learning_framework import models
from deep_learning_framework.training import Trainer, get_loss_function
from deep_learning_framework.data import TextDataset, get_dataloader
from deep_learning_framework.evaluation import Evaluator, accuracy
from deep_learning_framework.visualization import plot_metrics

# 准备数据
train_dataset = TextDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_length=128)
val_dataset = TextDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer, max_length=128)

train_loader = get_dataloader(train_dataset, batch_size=16)
val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)

# 创建模型
model = models.TransformerModel(vocab_size=30000, d_model=512, nhead=8)

# 定义损失函数和优化器
loss_fn = get_loss_function('ce')
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 创建训练器
trainer = Trainer(model, loss_fn, optimizer)

# 训练模型
history = trainer.fit(train_loader, val_loader, epochs=20, metrics={'accuracy': accuracy})

# 评估模型
evaluator = Evaluator(model, metrics={'accuracy': accuracy})
results = evaluator.evaluate_and_print(val_loader, loss_fn)

# 可视化结果
plot_metrics({
    'Train Accuracy': history['train_accuracy'],
    'Val Accuracy': history['val_accuracy']
})
```

## 自定义模型

您可以通过继承`BaseModel`类来创建自定义模型：

```python
from deep_learning_framework.models import BaseModel
import torch.nn as nn

class MyModel(BaseModel):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

## 自定义数据集

您可以通过继承`BaseDataset`类来创建自定义数据集：

```python
from deep_learning_framework.data import BaseDataset
import torch

class MyDataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        self.data = load_data(data_path)  # 自定义加载函数
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item
```

## 贡献

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。