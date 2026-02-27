# Intent-Aware Slot-Based Visual Classification

## Code Design Plan

This document describes the modular code design for an intent classification framework based on **multi-layer CLIP ViT features** and **intent-conditioned slot attention**.

---

## 1. Overall Architecture

**Pipeline**:
1. Input Image
2. Frozen CLIP ViT Backbone
3. Multi-layer Patch Token Extraction
4. Intent-Conditioned Slot Attention
5. Slot–Intent Agreement Scoring
6. Intent Prediction (Multi-label / Multi-class)

**Key principles**:
- The CLIP backbones (vision and text encoders) remain frozen. Intent queries are initialized from the CLIP text encoder and then fine-tuned for the downstream intent classification task.
- Intent is modeled as **evidence aggregation**, not single-vector classification.
- Slots are latent and unconstrained semantically, with only diversity regularization.

---

## 2. Module Breakdown

### 2.1 Backbone: CLIP ViT Wrapper

**Purpose**:
- Extract patch tokens from selected intermediate layers.
- Avoid using final CLS-only representations.

**Responsibilities**:
- Load pretrained CLIP ViT.
- Register forward hooks or manually capture tokens.
- Output a list of token tensors from selected layers.

**Interface**:
```python
class CLIPBackbone(nn.Module):
    def forward(self, images):
        """
        images: (B, 3, H, W)
        return:
            tokens: List[(B, N_l, D)]  # one per selected layer
        """
```

**Notes**:
- Selected layers are fixed (e.g., [6, 9, 12]).
- 默认不使用 CLS token（可配置切换）。
- 不应用 CLIP 模型内部的 `proj` 或 `ln_post` 层；后续模块通过单独的 Linear 层对齐特征维度和分布（Linear 对齐而非改变 CLIP 内部权重）。
- 保留并使用 CLIP 的原生 positional embeddings，不引入额外的 layer-wise / learned pos embedding。

---

### 2.2 Multi-layer Token Aggregator

**Purpose**:
- Combine tokens from multiple layers into a single token bank.

**Responsibilities**:
- Concatenate tokens along the sequence dimension.
- Optionally apply layer-wise normalization.
 - 不引入额外的位置编码或 layer-id embedding；仅在序列维度上拼接 token，并可选择对每层做规范化。

**Interface**:
```python
class TokenAggregator(nn.Module):
    def forward(self, tokens_per_layer):
        """
        tokens_per_layer: List[(B, N_l, D)]
        return:
            tokens: (B, N_total, D)
        """
```

---

### 2.3 Intent Query Encoder

**Purpose**:
- Provide a semantic anchor for each intent category.

**Responsibilities**:
- Encode intent descriptions using the pretrained CLIP text encoder.
- Store intent queries as fixed or trainable embeddings.
 - Encode intent descriptions using the pretrained CLIP text encoder to initialize intent queries. CLIP 的 text encoder 本身保持冻结；intent queries 从该编码器得到初始向量后作为可训练参数在下游任务中微调。

**Interface**:
```python
class IntentQueryBank(nn.Module):
    def get_queries(self):
        """
        return:
            intent_queries: (I, D)
        """
```

**Notes**:
- 每个样本使用同一组 intent queries（即 intent_queries 的形状为 (I, D)，在 batch 中按样本共享并广播为 (B, I, D)）。
- Default: initialized from the CLIP text encoder and then fine-tuned (trainable) for the task.
- 使用固定 prompt 模板："A photo that expresses the intent of {intent description}." 来构造文本输入以生成初始 intent queries。

---

### 2.4 Intent-Conditioned Slot Attention

**Purpose**:
- Dynamically bind intent-relevant visual evidence.
- Aggregate complementary latent factors.

**Responsibilities**:
- Initialize K latent slots.
- Perform cross-attention between slots and visual tokens.
- Condition attention on the current intent query.
- Iteratively refine slot representations.
 - Conditioning is implemented by adding an intent-derived term to the attention logits (see notes below).
 - Iteratively refine slot representations (iteration count configurable, typically 1–3).

**Interface**:
```python
class IntentConditionedSlotAttention(nn.Module):
    def forward(self, tokens, intent_query):
        """
        tokens: (B, N, D)
        intent_query: (B, D)
        return:
            slots: (B, K, D)
        """
```

**Design Notes**:
- Slot semantics are not predefined.
- Conditioning is implemented via additive attention bias from the intent query.
- Iteration count is small (1–3).

Attention conditioning details:
- We use single-head attention. Slot queries, visual keys and values use linear projections Q_slot, K, V ∈ R^{D} implemented with Linear(D → D).
- The attention logits are computed as:

    logits = Q_slot @ K^T + Q_intent @ K^T

    即在标准 slot-to-token dot-product 之外，加上由 intent query 投影得到的额外项，使得 intent 对 attention 权重施加偏置。
- 多头形式不使用（single-head）；所有投影维度保持为 D，以简化与 CLIP token/intent embedding 的对齐。

---

### 2.5 Slot–Intent Agreement Scoring

**Purpose**:
- Predict intent based on evidence consistency.

**Responsibilities**:
- Compute similarity between each slot and the intent query.
- Aggregate slot-level evidence into a scalar intent score.

**Interface**:
```python
def intent_score(slots, intent_query):
    """
    slots: (B, K, D)
    intent_query: (B, D)
    return:
        score: (B,)
    """
```

**Notes**:
- No MLP or linear classifier.
- Scoring is similarity-based for interpretability and robustness.

**Design Details**:
- 相似度函数采用 cosine similarity（先 L2 归一化再做点积）。
- 将 K 个 slot 与 intent_query 的相似度求和得到最终意图分数：

    score(intent) = sum_{k=1}^K cos(slot_k, intent_query)

- 可选：在需要时引入可学习的温度缩放作为可选超参。

---

### 2.6 Intent Classifier Wrapper

**Purpose**:
- Evaluate all intent categories.

**Responsibilities**:
- Loop over intent queries.
- Apply slot attention per intent.
- Collect intent scores into a prediction matrix.

**Interface**:
```python
class IntentClassifier(nn.Module):
    def forward(self, tokens):
        """
        tokens: (B, N, D)
        return:
            scores: (B, I)
        """
```

---

## 3. Training Objectives

### 3.1 Classification Loss
- OptimizedAsymmetricLoss

### 3.2 Slot Diversity Regularization

**Purpose**:
- Prevent slot collapse and redundancy.

**Loss**:
$[L_{ortho} = |SS^T - I|_F]$

Where:
- \(S\) are L2-normalized slot embeddings.

**Usage**:
```python
loss = cls_loss + λ * slot_orthogonality_loss
```

---

## 4. Training Strategy

- **Backbone**: frozen.
- **Slot attention + intent queries**: trainable.
- **Optimizer**: AdamW.
- **Learning rate**: small (e.g., 1e-4).
- **Slot count K**: 3–4.

---

## 5. Inference

- For each image, compute intent scores independently.
- Supports:
  - Multi-label thresholding.
  - Top-k intent retrieval.

---

## 6. Ablation Configuration Hooks

The codebase should support easy ablation of:
- Number of slots (K).
- Selected layers.
- With / without intent conditioning.
- CLS-only vs token-based input.
- Slot orthogonality on/off.
- Frozen vs trainable intent queries.

---

## 7. Visualization Utilities

- Token attention heatmaps per slot.
- Slot–intent similarity maps.
- Layer-wise contribution analysis.

*These are for qualitative analysis only and not used in training.*

---

## 8. Design Philosophy

- Minimal assumptions on slot semantics.
- No explicit semantic supervision.
- Intent treated as an abstract, multi-factor concept.
- Emphasis on robustness and interpretability.
