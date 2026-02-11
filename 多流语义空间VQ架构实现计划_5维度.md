# 多流语义空间VQ架构实现计划（5维度版本）

## 架构概览

```
CLIP ViT (Frozen) → Patch Tokens [B, 256, 1024]
    ↓
    ├─→ SemanticSpatialVQ_coco (COCO anchors) → z_coco [B, 256, 1024]
    ├─→ SemanticSpatialVQ_places (Places anchors) → z_places [B, 256, 1024]
    ├─→ SemanticSpatialVQ_emotion (Emotion anchors) → z_emotion [B, 256, 1024]
    ├─→ SemanticSpatialVQ_ava (AVA anchors) → z_ava [B, 256, 1024]
    └─→ SemanticSpatialVQ_actions (Stanford 40 Actions anchors) → z_actions [B, 256, 1024]
    ↓
Sum Fusion: z_fused = z_coco + z_places + z_emotion + z_ava + z_actions [B, 256, 1024]
    ↓
Global Pooling (mean) → [B, 1024]
    ↓
Classifier → [B, 28]
```

## 实施步骤

### 1. 创建语义锚点生成脚本

**文件**: `lightning-hydra/scripts/generate_semantic_anchors.py`

- 定义5个数据集的类别列表：
  - **COCO**: 80个类别（person, bicycle, car, ...）
  - **Places365**: 365个场景类别（airport_terminal, bedroom, forest, ...）
  - **Emotion**: 7个基本情绪，每个扩展同义词（Happiness → ["joy", "delight", "cheerful", "happy"]）
  - **AVA**: 14个风格属性（Complementary Colors, Duotones, High Dynamic Range, ...）
  - **Stanford 40 Actions**: 40个人类动作类别（playing instrument, reading, cooking, riding bike, ...）
- 使用CLIP Text Encoder提取特征，prompt格式：
  - **COCO/Places/Emotion/AVA**: `"a photo of {name}"`
  - **Stanford 40 Actions**: `"a photo of a person {action}"`（使用进行时，如 "a photo of a person playing instrument"）

**Codebook大小和扩充策略**：
- **Object (COCO)**: Codebook大小 = 256
  - 80个类别，每类约3个code（通过加噪声复制实现）
  - 扩充方法：对每个类别的原始embedding添加小噪声，生成多个变体
- **Action (Stanford 40 Actions)**: Codebook大小 = 256
  - 40个动作类别，直接使用或通过加噪声扩充到256
- **Scene (Places365)**: Codebook大小 = 512
  - 365个场景类别，直接使用或通过加噪声扩充到512
- **Style (AVA)**: Codebook大小 = 256
  - 14个风格属性，每类约18个code（通过加噪声复制实现）
  - 扩充方法：对每个类别的原始embedding添加小噪声，生成多个变体
- **Emotion**: Codebook大小 = 128
  - 7个基本情绪，通过扩充同义词库（Synonyms Expansion）实现
  - 扩充方法：为每个情绪类别添加多个同义词，如 Happiness → ["joy", "delight", "cheerful", "happy", "pleased", ...]

- 保存为 `semantic_anchors.pth`，格式：`{"coco": tensor[256, 1024], "places": tensor[512, 1024], "emotion": tensor[128, 1024], "ava": tensor[256, 1024], "actions": tensor[256, 1024]}`

**扩充策略实现细节**：
- **加噪声复制（Noise Augmentation）**（用于COCO和AVA）：
  ```python
  # 伪代码示例
  base_embedding = clip_model.encode_text(class_name)  # [1, 1024]
  num_codes_per_class = target_size // num_classes  # COCO: 3, AVA: 18
  augmented = []
  for i in range(num_codes_per_class):
      noise = torch.randn_like(base_embedding) * noise_scale  # 小噪声
      augmented.append(base_embedding + noise)
  ```
- **同义词扩展（Synonyms Expansion）**（用于Emotion）：
  ```python
  # 为每个情绪类别定义同义词列表
  emotion_synonyms = {
      "Happiness": ["joy", "delight", "cheerful", "happy", "pleased", "glad", ...],
      "Sadness": ["sorrow", "grief", "melancholy", "unhappy", ...],
      # ... 其他情绪
  }
  # 对每个同义词生成embedding，直到达到128个code
  ```

**Stanford 40 Actions类别列表（示例）**：
- playing instrument, reading, cooking, riding bike, riding horse, running, walking, jumping, phoning, using computer, taking photo, playing with pet, brushing teeth, brushing hair, writing, drinking, eating, cooking, text messaging, watching TV, etc.

**Prompt格式说明**：
- **COCO**: `"a photo of person"`, `"a photo of bicycle"`, `"a photo of car"`, ...
- **Places365**: `"a photo of airport_terminal"`, `"a photo of bedroom"`, `"a photo of forest"`, ...
- **Emotion**: `"a photo of joy"`, `"a photo of delight"`, `"a photo of cheerful"`, ...
- **AVA**: `"a photo of Complementary Colors"`, `"a photo of Duotones"`, ...
- **Stanford 40 Actions**: `"a photo of a person playing instrument"`, `"a photo of a person reading"`, `"a photo of a person cooking"`, ...（注意：使用进行时形式）

### 2. 实现SemanticSpatialVQ组件

**文件**: `lightning-hydra/src/models/components/semantic_spatial_vq.py`

- 继承或参考现有 `VectorQuantizer`，但使用语义锚点初始化
- 关键特性：
  - 输入：`[B, N_patches, D]` 空间特征图
  - 对每个patch位置独立量化（保留空间维度）
  - 使用Cosine相似度计算距离（更适合CLIP特征空间）
  - Codebook用锚点初始化，支持冻结/可学习模式
  - 返回量化特征和编码索引（用于可视化）
- **Codebook大小配置**：
  - 支持不同大小的codebook（256, 512, 128等）
  - 初始化时从预生成的锚点embedding加载

### 3. 创建新的Multi-Stream模块

**文件**: `lightning-hydra/src/models/intentonomy_clip_vit_multistream_module.py`

- 替换现有的 `IntentonomyClipViTCodebookModule`
- 架构组件：
  - **Backbone**: CLIP ViT（冻结），提取patch tokens `[B, 256, 1024]`
  - **5个SemanticSpatialVQ**: 分别对应COCO、Places、Emotion、AVA、Actions
  - **Fusion**: Sum融合5个VQ输出（保留空间对应关系）
  - **Global Pooling**: Mean pooling得到全局特征
  - **Classifier**: 线性层输出28类logits
- Loss计算：
  - ASL Loss（分类损失）
  - VQ Commitment Loss（5个VQ的损失求和）
  - 语义一致性Loss（可选，防止codebook漂移）

### 4. 实现语义一致性Loss

在 `IntentonomyClipViTMultiStreamModule` 中添加：

- 如果 `freeze_codebook=True`，则不需要额外loss
- 如果codebook可学习，添加正则项：`L_anchor = ||Codebook_i - Initial_Anchors_i||_2^2`（对5个VQ分别计算）

### 5. 创建配置文件

**文件**: `lightning-hydra/configs/model/intentonomy_clip_vit_multistream.yaml`
- 参考 `intentonomy_clip_vit_codebook.yaml` 的结构
- 新增参数：
  - `semantic_anchors_path`: 锚点文件路径
  - `freeze_codebook`: 是否冻结codebook（默认True）
  - `semantic_consistency_weight`: 语义一致性loss权重
  - **Codebook大小配置**（从锚点文件自动读取，但可在此文档说明）：
    - `codebook_size_coco`: 256
    - `codebook_size_places`: 512
    - `codebook_size_emotion`: 128
    - `codebook_size_ava`: 256
    - `codebook_size_actions`: 256
  - 5个VQ的学习率配置（可分别设置）：
    - `lr_vq_coco`
    - `lr_vq_places`
    - `lr_vq_emotion`
    - `lr_vq_ava`
    - `lr_vq_actions`

**文件**: `lightning-hydra/configs/experiment/intentonomy_clip_vit_multistream.yaml`
- 实验配置，覆盖默认参数

### 6. 训练策略

- **阶段1（冻结Codebook）**: 
  - `freeze_codebook=True`
  - 只训练fusion层和classifier
  - 预期收敛快，因为特征被强制映射到语义空间
  
- **阶段2（解冻微调）**:
  - `freeze_codebook=False`
  - Codebook学习率设为backbone的0.1倍
  - 允许在初始语义附近微调

## 关键文件修改/创建

1. **新文件**:
   - `lightning-hydra/scripts/generate_semantic_anchors.py` - 锚点生成脚本（包含5个维度）
   - `lightning-hydra/src/models/components/semantic_spatial_vq.py` - 空间VQ组件
   - `lightning-hydra/src/models/intentonomy_clip_vit_multistream_module.py` - 主模块（5个VQ）
   - `lightning-hydra/configs/model/intentonomy_clip_vit_multistream.yaml` - 模型配置
   - `lightning-hydra/configs/experiment/intentonomy_clip_vit_multistream.yaml` - 实验配置

2. **可能需要修改**:
   - `lightning-hydra/src/models/__init__.py` - 导出新模块（如果需要）

## 数据准备

需要准备5个数据集的类别列表（JSON或Python列表）：
- COCO: 80类标准类别名
- Places365: 365类场景名
- Emotion: 7类情绪及其同义词扩展
- AVA: 14类风格属性名
- **Stanford 40 Actions: 40类动作名**（新增）

## 可视化验证

实现 `get_code_indices()` 方法，返回每个factor在每个patch位置的编码索引，用于可视化验证：
- 输入海边冲浪图：
  - `VQ_places` 应激活 "ocean" 或 "beach"
  - `VQ_coco` 应激活 "surfboard" 或 "person"
  - `VQ_actions` 应激活 "surfing" 或 "riding wave"（新增验证点）

## 架构变化总结

- **从4个VQ扩展到5个VQ**：新增 `SemanticSpatialVQ_actions`
- **融合层更新**：Sum操作从4个输入改为5个输入
- **锚点文件格式**：新增 `"actions"` 键
- **配置参数**：新增 `lr_vq_actions` 等参数
- **可视化验证**：新增动作维度的验证点
- **Codebook大小配置**：
  - Object (COCO): 256 codes（每类~3个，通过噪声复制）
  - Action: 256 codes
  - Scene (Places365): 512 codes
  - Style (AVA): 256 codes（每类~18个，通过噪声复制）
  - Emotion: 128 codes（通过同义词扩展）

