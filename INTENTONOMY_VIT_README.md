# Intentonomy ViT 分类

使用 Vision Transformer (ViT) 对 Intentonomy 数据集进行多标签分类的 Lightning-Hydra 实现。

## 文件结构

- `src/data/intentonomy_datamodule.py`: Intentonomy 数据模块，负责加载图像和标注
- `src/data/components/cutout.py`: Cutout 数据增强实现
- `src/models/components/vit.py`: Vision Transformer 模型组件
- `src/models/components/aslloss.py`: Asymmetric Loss 损失函数实现
- `src/models/intentonomy_vit_module.py`: Lightning 模块，包含训练/验证/测试逻辑
- `configs/data/intentonomy.yaml`: 数据配置
- `configs/model/intentonomy_vit.yaml`: 模型配置

## 使用方法

### 训练

在 `lightning-hydra` 目录下运行：

```bash
python src/train.py data=intentonomy model=intentonomy_vit
```

### 自定义配置

可以通过命令行参数覆盖配置：

```bash
# 修改批次大小
python src/train.py data=intentonomy model=intentonomy_vit data.batch_size=64

# 修改学习率
python src/train.py data=intentonomy model=intentonomy_vit model.optimizer.lr=0.001

# 修改图像大小
python src/train.py data=intentonomy model=intentonomy_vit data.image_size=384

# 修改损失函数参数
python src/train.py data=intentonomy model=intentonomy_vit \
    model.criterion.gamma_neg=5 \
    model.criterion.gamma_pos=1 \
    model.criterion.clip=0.1
```

### 数据路径

默认数据路径配置在 `configs/data/intentonomy.yaml` 中：
- 图像目录: `../Intentonomy/data/images/low`
- 标注目录: `../Intentonomy/data/annotation`

如果数据路径不同，可以通过命令行修改：

```bash
python src/train.py data=intentonomy model=intentonomy_vit \
    data.image_dir=/path/to/images \
    data.annotation_dir=/path/to/annotations
```

## 模型说明

- **架构**: Vision Transformer (ViT-B/16)
- **预训练**: 使用 ImageNet 预训练权重
- **任务**: 多标签分类（28个类别）
- **数据增强**:
  - **RandAugment**: 随机增强，`num_ops=2`, `magnitude=9`
  - **Cutout**: 随机遮挡，`cutout_factor=0.5`
- **损失函数**: AsymmetricLossOptimized (ASL)
  - `gamma_neg=2`: 负样本的focal loss权重
  - `gamma_pos=0`: 正样本的focal loss权重
  - `clip=0.05`: 负样本概率裁剪
  - `eps=1e-5`: 数值稳定性参数
- **评估指标**: Accuracy, F1-Score, Precision, Recall

## 数据格式

标注文件为 COCO 格式的 JSON 文件：
- `intentonomy_train2020.json`: 训练集标注
- `intentonomy_val2020.json`: 验证集标注
- `intentonomy_test2020.json`: 测试集标注

每个图像可以有多个标签（多标签分类）。

