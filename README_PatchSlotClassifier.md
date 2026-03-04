# PatchSlotClassifier - 干净版本实现

> 极简的Slot-based意图分类器：**patch tokens → slot attention → flatten → concat CLS → MLP**

## 🚀 快速开始

### 使用方法

```python
from src.models.intentonomy_clip_vit_slot_module import CLIPIntentSlotModel

# 创建模型
model = CLIPIntentSlotModel(
    num_classes=28,
    clip_model_name="ViT-B/32",
    selected_layers=[12],
    use_cls_token=True,
    num_slots=4,
    use_single_layer_patch_slot=True,  # 启用干净版本
)

# 推理
logits, slots = model(images, return_slots=True)
```

### 训练

```bash
python src/train.py experiment=intentonomy_clip_vit_slot_single_layer
```

## ⚡ 核心特性

| 特性 | 说明 |
|------|------|
| 🧹 **极简设计** | 移除intent conditioning等复杂机制 |
| 🚄 **高效推理** | 单次slot attention，无per-intent循环 |
| 📊 **稳定训练** | 添加normalization保证稳定性 |
| 🔧 **易于调试** | 清晰的模块化结构 |
| ♻️ **向后兼容** | 通过参数开关控制 |

## 📦 架构

```
Input Image (224×224)
    ↓
CLIP Backbone (指定层)
    ↓
Patch Tokens [B, N, D]  ←─ normalize
    ↓                       ↓
SlotAttention           CLS Token [B, D]
    ↓                       ↓
Slots [B, K, D]            │
    ↓                       │
Flatten [B, K*D] ────concat──→ [B, (1+K)*D]
                            ↓
                        MLP Head
                            ↓
                    Logits [B, num_intents]
```

## 📝 文件说明

| 文件 | 说明 |
|------|------|
| `src/models/intentonomy_clip_vit_slot_module.py` | 主实现 |
| `configs/experiment/intentonomy_clip_vit_slot_single_layer.yaml` | 配置文件 |
| `test_single_layer_unit.py` | 单元测试 |
| `demo_patch_slot_classifier.py` | 交互式演示 |
| `docs/single_layer_patch_slot_classifier.md` | 详细文档 |

## 🧪 测试

```bash
# 运行单元测试
python test_single_layer_unit.py

# 运行交互式演示
python demo_patch_slot_classifier.py
```

## 🔑 关键改进

### 与Intent-Conditioned版本对比

| 项目 | Intent-Conditioned | 干净版本 |
|------|-------------------|---------|
| Intent Queries | ✅ 需要 | ❌ 不需要 |
| Conditioning | ✅ 复杂 | ❌ 简化 |
| Normalization | ❌ 无 | ✅ 有 |
| Temperature | ✅ 可选 | ❌ 移除 |
| 推理速度 | 慢 (28次循环) | 快 (1次) ✅ |

### 参数分析

假设 D=768, K=4, I=28:

- **MLP输入**: (1+4) × 768 = 3,840
- **MLP隐藏**: 3,840 → 1,920
- **总参数**: ~11.7M

推理速度提升：**~28x** (无per-intent循环)

## 🎯 使用第24层

```yaml
model:
  clip_model_name: "ViT-L/14"  # 24层，D=1024
  selected_layers: [24]
  use_single_layer_patch_slot: true
```

| 模型 | 层数 | 支持第24层 |
|------|------|-----------|
| ViT-B/32 | 12 | ❌ |
| **ViT-L/14** | **24** | **✅** |

## 💡 关键代码

### SlotAttention（纯版本）

```python
class SlotAttention(nn.Module):
    """Pure Slot Attention without intent conditioning."""
    
    def forward(self, tokens):
        # 迭代refinement
        for _ in range(self.iters):
            q = self.to_q(slots)
            attn = softmax(q @ k.T / sqrt(d))
            slots = slots + attn @ v
        return slots
```

### PatchSlotClassifier

```python
class PatchSlotClassifier(nn.Module):
    def forward(self, tokens, cls_embed):
        # Normalize
        tokens = F.normalize(tokens, dim=-1)
        cls_embed = F.normalize(cls_embed, dim=-1)
        
        # Slot attention
        slots = self.slot_attention(tokens)
        
        # Flatten & concat
        slots_flat = slots.reshape(bsz, -1)
        fused = torch.cat([cls_embed, slots_flat], dim=-1)
        
        # MLP
        logits = self.mlp_head(fused)
        return logits
```

## 📖 详细文档

查看 [docs/single_layer_patch_slot_classifier.md](docs/single_layer_patch_slot_classifier.md) 了解：
- 设计原则
- 完整实现细节
- 参数分析
- FAQ
- 可视化示例

## ✅ 验证状态

- ✅ 代码语法验证通过
- ✅ 无编译错误
- ✅ 单元测试就绪
- ✅ 向后兼容性保持
- ✅ 完整文档

## 🚦 下一步

1. ⏳ 在Intentonomy数据集上训练
2. ⏳ 与baseline对比性能（F1, mAP）
3. ⏳ Slot可解释性分析
4. ⏳ 超参数优化

## 📞 支持

遇到问题？
1. 查看 [详细文档](docs/single_layer_patch_slot_classifier.md)
2. 运行 `demo_patch_slot_classifier.py` 了解工作原理
3. 检查单元测试 `test_single_layer_unit.py`

---

**实现日期**: 2026-03-03  
**版本**: 1.0 (干净版本)
