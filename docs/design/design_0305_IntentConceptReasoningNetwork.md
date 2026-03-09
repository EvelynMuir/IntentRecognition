# Intent Concept Reasoning Network (ICRN)

## 状态（2026-03-06）

- 该方案已完成多轮实现与调试验证，但整体训练收益和稳定性未达到预期。
- 当前项目决策：**停止继续使用 ICRN 作为主训练方案**。
- 本文档保留 ICRN 的完整设计、实验与调试记录，用于后续复盘。

## 2026-03-06 TODO-2 实施与迭代记录

### 实施内容

- 新增 base 分支（与 `cls_mean_patch` 同构）：
  - 基于指定 `base_layer_idx` 提取未投影 `[CLS; mean_patch]`（2048 维 for ViT-L/14）
  - `base_head: Linear(2048->768)->ReLU->Dropout->Linear(768->28)`
- 输出融合：
  - `logits = z_base + alpha * z_concept`
  - `alpha` 为可学习标量，初始化 `0.0`
- checkpoint 初始化：
  - 新增 `cls_mean_patch_ckpt_path`
  - 可从 `layer_cls_patch_mean` checkpoint 加载 `base_head` 4 个 tensor（已验证）
- 优化器分组学习率：
  - `lr_base_head`
  - `lr_concept_branch`
  - `lr_alpha`

### 快速测试（同一小样本协议）

协议：
- train: 10 个 batch；val: 10 个 batch
- 训练 60 step 后比较 `val_macro/micro`

结果：
- 配置 A（按 TODO 原始建议）：`lr_base=1e-4, lr_concept=5e-4, lr_alpha=1e-3`
  - `val_macro: 0.4755 -> 0.4467`（下降）
  - `alpha: -0.008793`（快速变负）
- 配置 B（迭代-1）：`lr_alpha=1e-4`（其余同 A）
  - `val_macro: 0.4755 -> 0.4293`（下降更明显）
- 配置 C（迭代-2，当前默认）：`lr_base=0.0, lr_concept=5e-4, lr_alpha=1e-4`
  - `val_macro: 0.4755 -> 0.4712`（基本持平，退化最小）
  - `val_micro: 0.5526 -> 0.5482`（基本持平）

结论：
- “加载 cls_mean_patch 权重 + 双分支融合”可行且已打通。
- 小样本稳定性上，原始 TODO 学习率会破坏基线；当前默认改为更稳的 C 配置。

## 2026-03-05 调试迭代归档（已合并）

该部分来自原 `docs/record/icrn_debug_iterations_2026-03-05.md`，为完整保留的历史调试记录。

### 目标

- 排查 `experiment=intentonomy_clip_vit_icrn` 在多个 epoch 后仍接近 chance 的原因，并进行迭代修复。

### 基线现象

- 运行目录：
  - `logs/train/runs/2026-03-05_17-35-08`
  - `logs/train/runs/2026-03-05_21-36-53`
  - `logs/train/runs/2026-03-05_22-28-36`
- TensorBoard 观测：
  - `val/loss` 明显下降（如 `7.34 -> 2.61`）
  - `val/f1_macro` 基本持平（约 `0.118 ~ 0.123`）
  - `val/mAP` 偏低且变化不明显（约 `9.26 -> 9.16`）
  - `val/threshold` 多次塌缩到较低值（如 `0.05`）

### Iteration 1：ICRN head 的 logit 偏置修复

- 文件：`src/models/components/clip_vit_icrn.py`
- 改动：
  - 在分类前对 concept reasoning 输出做居中：
    - `refined_centered = refined - 0.5`
    - `logits = refined_centered @ intent_composition + intent_bias`
- 原因：
  - `sigmoid` concept activation 存在接近 `0.5` 的常数偏置。
  - 不居中时 logits 全局偏正，导致预测概率整体偏高。
- 快速诊断：
  - 修复前：预测均值约 `0.929`（几乎所有类都偏高）
  - 修复后：预测均值约 `0.541`（回到可学习区间）
  - 单步优化 loss：`5.439 -> 5.434`

### Iteration 2：Prior 消融（mini-fit）

- 方法：
  - 固定 3 个训练 mini-batch，优化 60 step，对比同一 mini-set 前后 macro/micro F1。
- 结果：
  - `default`（`init_with_llm_prior=true` + prior regularization）：
    - `macro 0.1176 -> 0.1176`（无变化）
  - `no_prior_reg`（仅关 prior regularization）：
    - `macro 0.1095 -> 0.1125`（小幅提升）
  - `no_init_prior_no_reg`（既不 init prior，也不 prior regularization）：
    - `macro 0.1242 -> 0.1278`（三者中最好）
- 结论：
  - 当时的 prior 初始化/约束在早期优化阶段可能有负作用。

### Iteration 3：默认训练策略更新

- 涉及文件：
  - `configs/experiment/intentonomy_clip_vit_icrn.yaml`
  - `scripts/intentonomy_clip_vit_icrn.sh`
  - `src/utils/metrics.py`
- 调整：
  - `model.net.init_with_llm_prior: false`
  - `model.use_prior_regularization: false`
  - `model.lambda_prior: 0.0`
  - 学习率覆盖为 `1e-3`
- 指标代码修复：
  - 在 `src/utils/metrics.py` 的 `f1_score(...)` 中统一设置 `zero_division=0`
  - 目的：减少标签缺失场景的 warning 并稳定指标计算。

### 当时 sanity check

- 组合配置确认：
  - `init_with_llm_prior=False`
  - `use_prior_regularization=False`
  - `lambda_prior=0.0`
  - `lr=0.001`
- 单 batch 检查：
  - `loss=4.5389`
  - `macro_f1=0.1249`
  - `threshold≈0.46`

### 阶段结论

- 虽然若干局部问题已修复，但 ICRN 主线训练效果仍未达到可接受水平，后续进入 2026-03-06 的双分支改造与继续迭代（见上文）。

### 2026-03-06 训练启动报错修复（`ValueError: some parameters appear in more than one parameter group`）

- 现象：
  - 在 ICRN 配置下训练启动时报 `ValueError: some parameters appear in more than one parameter group`。
- 定位：
  - `configure_optimizers` 在 `compile=true` 场景下，参数命名会出现 `_orig_mod.*` 前缀，分组逻辑需要统一命名并做参数去重。
- 修复：
  - 在 `src/models/intentonomy_clip_vit_icrn_module.py` 的参数分组中增加：
    - 名称归一化：`name.replace("_orig_mod.", "")`
    - 按参数对象 `id` 去重，确保同一参数不会被加入多个 group
  - 当前分组为：`base_head` / `alpha` / `concept_branch`（可选 `vit_blocks`）。
- 验证：
  - 直接实例化 `experiment=intentonomy_clip_vit_icrn` 并调用 `configure_optimizers()`，得到 3 个组，跨组重复参数数为 0。
  - 运行 `python src/train.py experiment=intentonomy_clip_vit_icrn +trainer.fast_dev_run=1 data.num_workers=0 data.batch_size=8` 可完成 1 step train+val+test，无该报错。

## 2026-03-06 实施与测试记录

### 代码修改

- `src/models/components/clip_vit_icrn.py`
  - 新增 `concept_temperature`，将 `concept_logits` 变为 `scaled_scores = logits / tau_c`
  - 新增 visual adapter（`use_visual_adapter`, `adapter_hidden_ratio`, `adapter_dropout`）
  - 图传播改为 `top-k` 稀疏图（`graph_topk`）+ 残差融合（`graph_residual_alpha`）
  - intent logits 使用 `refined_scores @ intent_composition + intent_bias`
- `src/models/intentonomy_clip_vit_icrn_module.py`
  - 新增 `prior_regularization_start_epoch`，支持 prior 正则延迟生效
- `configs/experiment/intentonomy_clip_vit_icrn.yaml`
  - 默认启用 TODO v1 参数：
    - `init_with_llm_prior=true`
    - `use_prior_regularization=false`
    - `prior_regularization_start_epoch=8`
    - adapter/top-k/tau 参数写入
    - `lr=5e-4`
- `scripts/intentonomy_clip_vit_icrn.sh`
  - 默认对齐实验策略（init prior + no prior regularization）

### 快速对比实验（同一小样本、同优化步数）

测试方法：
- 训练集取 3 个 batch，验证集取 5 个 batch
- 先评估，再训练 120 step，再评估
- 指标：`eval_validation_set` 中 `val_macro/val_micro/threshold`

结果：

- `legacy_like`（不加 adapter，`tau=1.0`，dense 图，`alpha=0`）
  - `val_macro`: `0.1160 -> 0.1149`（无提升）
  - `val_loss`: `9.2680 -> 4.1197`
- `todo_v1`（adapter + `tau=0.2` + top-k=16 + `alpha=0.5`）
  - `val_macro`: `0.1288 -> 0.1941`（显著提升）
  - `val_loss`: `50.6406 -> 2.6302`

结论：
- TODO v1 在“保持 concept 机制”的前提下，学习性明显优于旧式链路。
- 下一步建议按 TODO v1 配置跑完整训练并观察 `val/f1_macro` 与 `val/mAP` 的 epoch 曲线。

## 1. Overview

ICRN predicts image intents through an intermediate **shared concept layer**, instead of directly mapping image features to intent labels.

Core assumption: intents are compositions of reusable visual-semantic concepts (e.g., social interaction, emotional cues, contextual situations).

Pipeline:

```text
Image
  -> CLIP visual encoder (frozen)
  -> visual feature v
  -> concept grounding
  -> concept activations c
  -> concept graph reasoning
  -> refined concepts c'
  -> intent composition
  -> intent logits z
```

## 2. Notation

- `D`: visual/text embedding dimension.
- `M`: number of concepts (typically `80-120`).
- `K`: number of intents (here `K = 28`).
- `v in R^D`: image embedding.
- `E_c in R^(M x D)`: concept embedding matrix.
- `c in R^M`: concept activation vector.
- `A in R^(M x M)`: concept graph adjacency/similarity matrix.
- `W in R^(M x K)`: concept-to-intent composition matrix.
- `z in R^K`: intent logits.

## 3. Step-by-Step Design

### Step 1. Visual Feature Extraction

Use a frozen CLIP ViT-L/14 visual encoder.

For each image:

```text
v = CLIP_visual(image)
```

Recommended feature construction:

```text
v = concat(CLS_token, mean(patch_tokens))
v = normalize(v)
```

复用cls_mean_patch

### Step 2. Concept Embedding

Maintain a shared concept pool:

```text
concept_list = [c1, c2, ..., cM]
```

文件在`/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json`

Concept examples:

- hugging
- group gathering
- smiling
- conversation
- collaborative work
- helping another person

Generate concept embeddings via frozen CLIP text encoder:

```text
E_c = CLIP_text(concept_list)
E_c = normalize(E_c)
```

### Step 3. Concept Grounding

Ground concepts by similarity between image and concept embeddings:

```text
c = sigmoid(E_c * v)
```

- `E_c`: `(M x D)`
- `v`: `(D)`
- `c`: `(M)`

Example interpretation:

- `c_smiling = 0.82`
- `c_hugging = 0.76`
- `c_group_gathering = 0.65`

### Step 4. Concept Graph Construction

Concepts are correlated. Build a concept similarity graph from concept embeddings:

```text
A_raw = E_c * E_c^T
A = row_softmax(A_raw)
```

`A` encodes semantic proximity among concepts.

Example relations:

- hugging <-> physical_contact
- smiling <-> joy
- conversation <-> social_interaction

### Step 5. Concept Graph Reasoning

Refine concept activations through graph propagation:

```text
c' = A * c
```

Optional nonlinearity:

```text
c' = ReLU(A * c)
```

This allows related concepts to reinforce each other.

### Step 6. Intent Composition

Model each intent as a weighted combination of concepts:

```text
z = c' * W
y_hat = sigmoid(z)
```

- `c'`: `(M)`
- `W`: `(M x K)`
- `z`: `(K)`

## 4. LLM Intent Prior (Optional but Recommended)

Use an LLM offline to estimate intent-to-concept relevance and form a prior matrix:

- `S in R^(K x M)`

Example:

```text
comforting:
  hugging: 0.9
  emotional_support: 0.8
  physical_contact: 0.7

celebrating:
  smiling: 0.9
  group_gathering: 0.85
  cheering: 0.8
```

文件在`/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json`

Initialize the composition matrix with the prior:

```text
W = S^T
```

Optional prior regularization:

```text
L_prior = ||W - S^T||^2
```

## 5. Training Objective

Base classification loss（复用Optimized Asymmetric Loss）:

```text
L_cls = ASL(z, y)
```

With prior regularization:

```text
L = L_cls + lambda_1 * L_prior
```

Optional concept sparsity (images usually activate only a subset of concepts):

```text
L_sparse = ||c||_1
L = L_cls + lambda_1 * L_prior + lambda_2 * L_sparse
```

## 6. Inference

```text
v = CLIP_visual(image)
c = sigmoid(E_c * v)
c' = A * c
z = c' * W
y_hat = sigmoid(z)
```

Obtain predicted intents by thresholding `y_hat`.

## 7. Sanity Checks

- Dimension consistency is valid across all stages.
- `W` is a full composition matrix (`M x K`), not a single intent vector.
- Use `row_softmax` for `A` to avoid ambiguity in graph normalization.
- Keep naming consistent (`CLIP_visual` during both training/inference).
