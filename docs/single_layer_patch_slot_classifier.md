# PatchSlotClassifier 实现文档（干净版本）

## 概述

实现了一种极简的意图分类方法：从指定CLIP层提取patch tokens，通过纯slot attention（无intent conditioning），将所有slots和cls token normalize后flatten并concat，通过单个MLP生成所有意图的logits。

## 设计原则

### 为什么选择干净版本？
1. **简化架构**: 移除intent conditioning的复杂性
2. **减少依赖**: 不需要intent queries，更独立
3. **更好的基线**: 作为slot-based方法的最小可行版本
4. **易于调试**: 更少的组件意味着更容易定位问题
5. **规范化处理**: 添加normalization保证稳定性

### 与Intent-Conditioned版本的对比

| 特性 | Intent-Conditioned | 干净版本 |
|------|-------------------|---------|
| Intent Queries | ✅ 需要 | ❌ 不需要 |
| Intent Conditioning | ✅ 使用 | ❌ 不使用 |
| Temperature Scaling | ✅ 可选 | ❌ 移除 |
| Normalization | ❌ 无 | ✅ tokens & cls |
| 复杂度 | 高 | 低 ✅ |
| 适用场景 | 需要intent先验 | 通用baseline |

## 实现细节

### 核心模块: `SlotAttention`

纯slot attention实现，无intent conditioning：

```python
class SlotAttention(nn.Module):
    """Pure Slot Attention without intent conditioning."""
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D] -> slots: [B, K, D]"""
```

**关键公式**：
```
Q = to_q(slots)
K = to_k(tokens)
V = to_v(tokens)

attn = softmax(Q @ K^T / sqrt(d))
slots = slots + attn @ V  # 迭代更新
```

### 核心模块: `PatchSlotClassifier`

```python
class PatchSlotClassifier(nn.Module):
    """
    Minimal Slot Experiment
    patch tokens -> slot attention -> flatten -> concat CLS -> MLP
    """
```

**完整流程**：

```python
def forward(self, tokens, cls_embed, return_slots=False):
    # 1. Normalization（保持与baseline一致）
    tokens = F.normalize(tokens, dim=-1)     # [B, N, D]
    cls_embed = F.normalize(cls_embed, dim=-1)  # [B, D]
    
    # 2. Slot Attention
    slots = self.slot_attention(tokens)      # [B, K, D]
    
    # 3. Flatten & Concat
    slots_flat = slots.reshape(bsz, -1)      # [B, K*D]
    fused = torch.cat([cls_embed, slots_flat], dim=-1)  # [B, (1+K)*D]
    
    # 4. MLP分类
    logits = self.mlp_head(fused)            # [B, num_intents]
    
    return logits, (slots if return_slots else None)
```

### MLP架构

```python
mlp_input_dim = (1 + num_slots) * dim  # CLS + K slots
self.mlp_head = nn.Sequential(
    nn.Linear(mlp_input_dim, mlp_input_dim // 2),
    nn.GELU(),
    nn.Linear(mlp_input_dim // 2, num_intents),
)
```

默认配置 (K=4, D=768):
- 输入维度: (1+4) × 768 = 3840
- 隐藏层: 1920
- 输出: 28 (intents)

## 使用方法

### 1. 代码中使用

```python
from src.models.intentonomy_clip_vit_slot_module import CLIPIntentSlotModel

model = CLIPIntentSlotModel(
    num_classes=28,
    clip_model_name="ViT-B/32",
    selected_layers=[12],  # 最后一层
    use_cls_token=True,    # 必须启用！
    num_slots=4,
    slot_iters=3,
    use_single_layer_patch_slot=True,  # 启用干净版本
)

# Forward pass（不需要intent queries！）
images = torch.randn(2, 3, 224, 224)
logits, slots = model(images, return_slots=True)
```

### 2. 配置文件

使用 `configs/experiment/intentonomy_clip_vit_slot_single_layer.yaml`:

```yaml
model:
  clip_model_name: "ViT-B/32"
  selected_layers: [12]
  use_cls_token: true  # 必须！
  num_slots: 4
  use_single_layer_patch_slot: true
```

### 3. 训练

```bash
python src/train.py experiment=intentonomy_clip_vit_slot_single_layer
```

## 关键特性

### 1. Normalization

**为什么添加normalization？**
- 保持与baseline方法一致
- 提高训练稳定性
- 避免梯度爆炸/消失

```python
tokens = F.normalize(tokens, dim=-1)      # 每个token归一化到单位球面
cls_embed = F.normalize(cls_embed, dim=-1)  # cls也归一化
```

### 2. 简化的Slot Attention

**移除的内容**：
- ❌ Intent conditioning (`q_slot + q_intent`)
- ❌ 复杂的query生成机制
- ❌ Temperature scaling

**保留的内容**：
- ✅ 可学习的slot初始化 (`slot_mu`, `slot_sigma`)
- ✅ 迭代refinement机制
- ✅ LayerNorm + FFN结构

### 3. 统一的MLP分类头

不同于per-intent的独立头，使用单个共享MLP：

**优势**：
- 参数共享，泛化性更好
- 自动学习intents之间的关系
- 推理速度快（一次前向传播）

**劣势**：
- 可能损失一些per-intent的特异性

## 参数分析

假设 B=32, N=196 (14×14), D=768, K=4, I=28:

### SlotAttention 参数
```
slot_mu:     1 × 4 × 768 = 3,072
slot_sigma:  1 × 4 × 768 = 3,072
to_q:        768 × 768 = 589,824
to_k:        768 × 768 = 589,824
to_v:        768 × 768 = 589,824
FFN:         768 → 1536 → 768 ≈ 2.4M
Total:       ≈ 4.2M
```

### MLP Head 参数
```
Input:  (1+4) × 768 = 3840
Hidden: 3840 → 1920 = 7.4M
Output: 1920 → 28 = 54K
Total:  ≈ 7.5M
```

### 总参数量
```
SlotAttention: 4.2M
MLP Head:      7.5M
Total:         11.7M (可训练参数)
```

**与原方法对比**：
- Per-intent方法: 28次slot attention调用 + 28个独立头
- 干净版本: 1次slot attention + 1个共享MLP
- **推理速度提升: ~28x**

## 使用第24层 (ViT-L/14)

要使用第24层的patch tokens:

```yaml
model:
  clip_model_name: "ViT-L/14"  # 24层ViT-L
  selected_layers: [24]
  use_single_layer_patch_slot: true
```

**模型对比**:
| 模型 | 层数 | 隐藏维度 | 支持第24层 |
|------|------|----------|-----------|
| ViT-B/32 | 12 | 768 | ❌ |
| ViT-B/16 | 12 | 768 | ❌ |
| **ViT-L/14** | **24** | **1024** | **✅** |

## 测试验证

运行单元测试:

```bash
python test_single_layer_unit.py
```

测试内容:
1. ✅ SlotAttention（纯版本）
2. ✅ PatchSlotClassifier结构
3. ✅ Forward Pass
4. ✅ 梯度流
5. ✅ Normalization效果

## 常见问题

### Q: 为什么不需要intent queries？
A: 干净版本让slots自由学习representations，不强制使用intent先验。MLP分类头会自动学习如何将slots映射到intents。

### Q: Normalization会影响性能吗？
A: 通常会提高稳定性。我们normalize到单位球面，让模型专注于方向而非magnitude。

### Q: 能否不用cls token？
A: 不建议。cls token提供全局信息，对分类很重要。必须设置`use_cls_token=True`。

### Q: 如何调参？
A: 关键超参数：
- `num_slots`: 4, 8, 16（更多slots可能捕获更多信息）
- `slot_iters`: 2, 3, 5（更多迭代可能更精细）
- `mlp_hidden`: 当前是输入维度的一半，可调整

### Q: 与baseline性能对比？
A: 需要在实际数据上验证。理论上：
- **速度**: 显著更快（无per-intent循环）
- **精度**: 可能略有不同，取决于任务

### Q: 如何可视化slots？
A: 
```python
logits, slots = model(images, return_slots=True)
print(f"Slots shape: {slots.shape}")  # [B, 4, D]

# 可以分析每个slot关注的tokens
# 或者使用attention weights可视化
```

## 代码示例

### 完整训练示例

```python
import torch
from src.models.intentonomy_clip_vit_slot_module import CLIPIntentSlotModel

# 创建模型
model = CLIPIntentSlotModel(
    num_classes=28,
    clip_model_name="ViT-B/32",
    selected_layers=[12],
    use_cls_token=True,
    num_slots=4,
    slot_iters=3,
    use_single_layer_patch_slot=True,
)

# 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for images, labels in dataloader:
    logits, _ = model(images, return_slots=False)
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Slot可视化示例

```python
# 获取slots
logits, slots = model(images, return_slots=True)  # slots: [B, 4, D]

# 分析每个slot的activation
slot_norms = torch.norm(slots, dim=-1)  # [B, 4]
print(f"Slot norms: {slot_norms[0]}")  # 查看第一张图的4个slots

# 可以进一步分析哪些intents与哪些slots相关
```

## 关键修改文件

1. **主文件**: [src/models/intentonomy_clip_vit_slot_module.py](../src/models/intentonomy_clip_vit_slot_module.py)
   - 新增 `SlotAttention` 类（纯版本）
   - 新增 `PatchSlotClassifier` 类
   - 修改 `IntentClassifier` 添加路由

2. **配置文件**: [configs/experiment/intentonomy_clip_vit_slot_single_layer.yaml](../configs/experiment/intentonomy_clip_vit_slot_single_layer.yaml)

3. **测试文件**: [test_single_layer_unit.py](../test_single_layer_unit.py)

## 下一步

1. ✅ 基础实现完成
2. ✅ 单元测试通过
3. ⏳ 在Intentonomy数据集上训练
4. ⏳ 与baseline对比性能
5. ⏳ 优化超参数
6. ⏳ Slot可解释性分析

## 实现亮点

✨ **极简设计**: 移除所有不必要的复杂性
✨ **稳定训练**: 添加normalization保证稳定性
✨ **高效推理**: 单次前向传播，无循环
✨ **易于扩展**: 清晰的模块化结构
✨ **向后兼容**: 通过参数开关控制

## 结论

`PatchSlotClassifier` 提供了一个干净、高效的slot-based意图识别方案。通过移除intent conditioning和temperature scaling等复杂机制，我们获得了：

1. **更简单的架构** - 易于理解和调试
2. **更快的推理** - 单次slot attention而非28次
3. **更好的基线** - 作为slot方法的最小可行版本

适合作为实验起点，后续可根据实际效果决定是否需要添加更复杂的机制。
