# Method

## 1. Overview

本文最终方法建立在三个已经被单独验证有效的部件之上：

1. `Text-only Teacher`
   - 输入仅为 VLM 生成的 rationale 文本特征
   - 作用：提供更稳的语义软分布
2. `Dynamic Gated Distillation`
   - 用人类标注一致性 `omega` 控制 student 向 teacher 学习的强度
   - 作用：避免 student 在高噪声样本上过拟合硬标签
3. `SLR-C Prior`
   - 用 scenario-aware 的 fixed logit prior 提供更强 proposal
   - 作用：把 student 的起点从“裸视觉分类”提升到“强语义 proposal + residual correction”

最终论文版建议采用的主方法是：

> `SLR-C + residual student + text-only teacher + dynamic gated distillation`

对应实现脚本：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

teacher 与基础蒸馏脚本：

- [scripts/analyze_privileged_distillation.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_privileged_distillation.py)

---

## 2. Problem Setup

给定图像 `x`，目标是进行 Intentonomy 的多标签意图识别。

训练数据包含：

- 图像视觉特征 `v_x`
- 二值标签 `y in {0,1}^C`
- 软一致性标签 `omega in {0, 1/3, 2/3, 1}^C`

其中：

- `y` 用于最终监督
- `omega` 反映专家对每个类别的分歧程度

我们还额外拥有两类特权信息：

1. `rationale text`
   - 由 VLM 离线生成
   - 再编码成 BGE 向量
2. `scenario prior`
   - 由 CLIP image feature 与 scenario text embedding 计算得到

---

## 3. Final Method

这一节按模块拆开解释最终方法，而不是只给一个整体公式。论文里建议直接按下面这几个模块写：

1. frozen visual backbone
2. text-only teacher
3. fixed `SLR-C` prior
4. residual student
5. dynamic gated distillation
6. class-wise threshold inference

### 3.1 Text-Only Teacher

teacher 不再接视觉分支，而是只消费 rationale 文本特征：

\[
z^{tea}(x) = f_{tea}(t_x)
\]

其中：

- `t_x`：BGE rationale feature
- `f_tea`：一个轻量 MLP

teacher 训练目标是多标签 ASL：

\[
\mathcal{L}_{tea} = \mathrm{ASL}(z^{tea}, y)
\]

实现里使用：

- `AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05)`

关键代码位置：

- teacher 模型定义：[scripts/analyze_privileged_distillation.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_privileged_distillation.py)
- teacher 训练逻辑：[scripts/analyze_privileged_distillation.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_privileged_distillation.py)

对应代码核心形态：

```python
teacher = TeacherMLP(
    text_dim=1024,
    hidden_dim=1024,
    num_classes=28,
    dropout=0.1,
    input_mode="text_only",
)

logits = model(text_features=text_features, image_features=None)
loss = criterion(logits, labels, reduction="mean")
```

teacher 的实际结构是两层 MLP：

```python
class TeacherMLP(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_classes, dropout):
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
```

这里刻意不接视觉特征，原因来自消融：

- `text_only teacher` 明显优于 `image + text teacher`
- 说明当前 teacher 更像在学习“文本语义决策边界”，而不是多模态融合

从功能上讲，teacher 解决的是：

> 当图像本身视觉边界不清楚时，仅依赖 image feature 很难给出稳定决策；  
> rationale 文本提供了更高层的语义解释，因此可以作为更干净的 soft target。

### 3.2 Fixed SLR-C Prior

`SLR-C` 这一名字建议在正文里直接拆开解释：

- `SLR` = Scenario-based Local Rerank
- `C` = Class-wise threshold calibration

也就是说，`SLR-C` 不是一个单一模块，而是两步串联：

1. `SLR`
   - 用 scenario text prior 改写 baseline logits 的 top-k 候选排序
2. `C`
   - 在验证集上为每个类别单独搜索 threshold，得到最终决策规则

#### 3.2.1 Scenario Prior

对每个类别，我们从 `intent_description_gemini.json` 中提取一组 scenario text queries：

\[
\mathcal{Q}_c = \{q_{c,1}, q_{c,2}, \ldots\}
\]

再用 CLIP text encoder 编码，并对同一类别的多个 query 做平均：

\[
e_c^{scenario} = \mathrm{Normalize}\left(
\frac{1}{|\mathcal{Q}_c|}\sum_j \mathrm{CLIPText}(q_{c,j})
\right)
\]

对任意图像 `x`，其 frozen CLIP image feature 为 `v_x^{clip}`，则 scenario prior logits 为：

\[
s_c^{scenario}(x) = \langle v_x^{clip}, e_c^{scenario} \rangle \cdot \gamma
\]

其中 `gamma` 是 CLIP 的 `logit_scale`。

#### 3.2.2 Local Rerank (SLR)

在最终方法里，我们先用固定的 `SLR-C` 生成 proposal logits：

\[
z^{slr}(x) = z^{base}(x) + \alpha \cdot \tilde s^{scenario}(x), \quad c \in \mathrm{TopK}(z^{base})
\]

其中：

- `z^{base}(x)`：原始视觉 baseline logits
- `s^{scenario}(x)`：scenario 文本先验
- `alpha = 0.3`
- `topk = 10`

这一步对应旧的 `scenario SLR-C`，但在这里我们把它看作一个**固定 prior**，而不是最终输出。

这里的“local”非常重要：scenario prior 只作用于 baseline top-k 候选，而不是全类别空间。也就是说，`SLR` 不是替代视觉 logits，而是在视觉模型已经提名的候选集里做局部重排。

关键代码：

```python
train_prior_logits = _text_logits_from_features(
    train_clip["features"], scenario_text_embeddings, logit_scale
)
train_slr_logits = _apply_slr(
    train_base["logits"], train_prior_logits, topk=10, alpha=0.3
)
```

见：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

`SLR-C` 的完整构造流程是：

1. 从 `intent_description_gemini.json` 中取每类的 scenario text query
2. 用 CLIP text encoder 编码每个类别的 scenario text
3. 用 frozen CLIP image feature 与 scenario text embedding 做相似度
4. 只在 baseline top-k 候选内部加 scenario prior

对应代码：

```python
scenario_pools = _build_scenario_text_pool(class_names, gemini_file)
scenario_text_embeddings = _encode_text_pool(clip_model, scenario_pools)
train_prior_logits = _text_logits_from_features(
    train_clip["features"], scenario_text_embeddings, logit_scale
)
train_slr_logits = _apply_slr(
    train_base["logits"], train_prior_logits, topk=10, alpha=0.3
)
```

`_apply_slr()` 的实际语义是：

1. 先对 `scenario prior logits` 做 per-sample normalization
2. 对每张图只取 baseline logits 的 top-k 类
3. 仅在这些类上加一个 `alpha * prior` residual

因此：

\[
z^{slr}(x)
\]

本质上是一个**strong calibrated proposal**，而不是新的全局分类器。

#### 3.2.3 Class-wise Threshold (C)

`SLR` 只改 score，不给最终 label set。真正让它变成 `SLR-C` 的，是验证集上的 class-wise threshold search：

\[
\tau_c^* = \arg\max_{\tau} \mathrm{F1}_c(\tau)
\]

推理时按：

\[
\hat y_c = \mathbb{I}(\sigma(z_c^{slr}) > \tau_c^*)
\]

这也是为什么 `SLR-C` 在 repo 里一直是一个很强的 frozen proposal system：  
它不仅有文本先验，还带着一套 per-class calibrated decision rule。

### 3.3 Residual Student

student 不从零学习 logits，而是在固定 `SLR-C` 上学习 residual：

\[
z^{stu}(x) = z^{slr}(x) + r_\theta(v_x)
\]

其中：

- `v_x`：冻结 CLIP image feature cache
- `r_\theta`：小型 MLP residual head

关键代码：

```python
class ResidualStudent(nn.Module):
    def forward(self, image_features, slr_logits):
        return slr_logits + self.net(image_features)
```

这使得 student 的职责从“全局识别”收缩为“对强 proposal 做局部修正”。

换句话说，student 不再重复学习“哪些类大体相关”，而只需要学习：

1. fixed `SLR-C` 的剩余误差
2. 图像特征中未被 `SLR-C` 吸收的局部视觉证据
3. teacher soft target 暗示的更细边界

对应 residual head 具体实现：

```python
self.net = nn.Sequential(
    nn.Linear(image_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes),
)
```

### 3.4 Distillation Objective

teacher 提供软分布：

\[
p^{tea}(x) = \sigma(z^{tea}(x) / T)
\]

其中温度固定为 `T = 2.0`。

student 使用两类损失：

1. 监督损失

\[
\mathcal{L}_{sup} = \mathrm{ASL}(z^{stu}, y)
\]

2. 多标签 Bernoulli KL 蒸馏损失

\[
\mathcal{L}_{kd}
=
\sum_c
\mathrm{KL}
\left(
\sigma(z^{stu}_c / T)
\;\|\;
\sigma(z^{tea}_c / T)
\right)
\]

这里用 Bernoulli KL 而不是 softmax KD，是因为任务本身是多标签，多类别之间并不互斥。

### 3.5 Dynamic Gated Distillation

最终主方法采用 sample-wise inverse gate：

\[
w^{tea}_i = 1 - \omega_i
\]

其中：

- `omega_i` 不是 per-class 平均，而是当前样本所有正类 agreement 的 `min`
- 这样只要一个正类很不确定，这个样本就会被视为 harder / noisier

直观上，这个 gate 表达的是：

- `omega_i` 高：更相信硬标签
- `omega_i` 低：更相信 teacher

因此，当前最终版本不是“teacher 永远纠正学生”，而是：

> 只在样本被认为 noisy / hard 的时候，让 teacher 真正介入。

对应总损失：

\[
\mathcal{L}_{total}
=
(1 - w^{tea}_i)\,\mathcal{L}_{sup}
+
w^{tea}_i\,\lambda\,\mathcal{L}_{kd}
\]

这里 `lambda = 1.0`。

关键代码：

```python
teacher_weight = (1.0 - agreement).unsqueeze(1)
supervised_weight = 1.0 - teacher_weight
loss_per_class = (
    supervised_weight * supervised_per_class
    + teacher_weight * kd_per_class
)
loss_per_sample = loss_per_class.sum(dim=1)
```

实现位置：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

与不带 `SLR-C` 的蒸馏相比，这里最关键的区别是：

- student 学习的是 `residual over SLR-C`
- 而不是 `direct logits from image feature`

### 3.6 Inference

推理时只保留 student：

\[
\hat y = \mathbb{I}(\sigma(z^{stu}) > \tau_c)
\]

阈值 `tau_c` 由验证集 class-wise threshold search 得到。

最终推理分两层：

1. frozen prior：
   - `z^{slr}(x)`
2. trainable correction：
   - `r_\theta(v_x)`

所以最终输出仍然是一个“prior + residual”的形式，而不是 end-to-end 重写 prior。

---

## 4. Implementation Details

### 4.1 Data and Caches

主线实验全部基于离线缓存：

- CLIP image feature cache：
  - `logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`
- rationale text feature：
  - `logs/analysis/vlm_full_20260316`

teacher 使用：

- train：`rationale_full_bge_features.npz`
- val/test：`val_rationale_baseline_pred_bge_features.npz` 和 `test_rationale_baseline_pred_bge_features.npz`

### 4.2 Training Pipeline

最终方法的完整训练流程如下：

1. 训练 `text-only teacher`
   - 输入：BGE rationale feature
   - 输出：teacher logits
   - 损失：ASL
2. 重建固定 `SLR-C`
   - 输入：frozen CLIP image cache + scenario text embedding
   - 输出：fixed `SLR-C logits`
3. 在 `SLR-C logits` 上训练 residual student
   - 监督项：ASL
   - 蒸馏项：teacher Bernoulli KL
   - 门控：`1 - omega`
4. 在验证集上做 class-wise threshold search
5. 在测试集上汇报 `Macro / Micro / Samples / mAP / Hard`

如果论文里要用更贴近算法描述的写法，可以把它写成：

**Stage A.** Train text-only rationale teacher  
**Stage B.** Reconstruct frozen `SLR-C` prior from CLIP caches  
**Stage C.** Train residual student on top of the prior with supervised + KD loss  
**Stage D.** Search class-wise thresholds on validation and evaluate on test

可以写成简洁伪代码：

```python
# phase 1: teacher
teacher = train_teacher(text_features, binary_labels)

# phase 2: fixed SLR-C prior
slr_logits = build_slr_c(base_logits, scenario_prior_logits)

# phase 3: residual student
for x in train_cache:
    z_stu = slr_logits[x] + residual_head(image_feature[x])
    loss_sup = ASL(z_stu, y[x])
    loss_kd = KL(sigmoid(z_stu / T), teacher_prob[x])
    loss = (1 - w[x]) * loss_sup + w[x] * loss_kd
```

### 4.3 Optimization

统一设置：

- optimizer：`AdamW`
- lr：`5e-4`
- weight_decay：`1e-4`
- batch_size：`256`
- patience：`4~6`
- teacher hidden dim：`1024`
- student hidden dim：`768`
- dropout：`0.1`
- distillation temperature：`2.0`

### 4.4 Reproducibility

为了防止不同 teacher 配置污染后续 student 初始化，训练时对不同组件使用独立 seed：

- teacher：`seed + 0`
- baseline：`seed + 100`
- standard KD：`seed + 200`
- dynamic KD：`seed + 300`

---

## 5. Core Results

### 5.1 Final Method

最终方法来自：

- `logs/analysis/distillation_slrc_20260317`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slr_c_fixed` | 51.08 | 58.81 | 58.02 | 53.66 | 33.98 |
| `slr_c_residual_sup` | 50.41 | 57.70 | 57.56 | 52.18 | 34.77 |
| `slr_c_residual_standard_kd` | **51.77** | 58.97 | 58.99 | 53.42 | **36.43** |
| `slr_c_residual_dynamic_kd` | 51.52 | **59.75** | **60.30** | **54.04** | 34.60 |

主结论：

1. `SLR-C` 接入蒸馏框架后，整体性能正式提升
2. `standard_kd` 最强于 `Macro / Hard`
3. `dynamic_kd` 最强于 `Micro / Samples / mAP`

如果论文更偏向 overall system，建议主方法写成：

- `SLR-C + residual dynamic KD`

如果论文更偏向 hardest classes，也可以把：

- `SLR-C + residual standard KD`

作为主表 strongest variant。

---

## 6. Ablations

### 6.1 Backbone Ablation

我们对 backbone 做了两层消融：

1. 先比较不带 `SLR-C` 的蒸馏主线
2. 再比较 `ResNet101` 下的完整方法

#### 6.1.1 Plain Distillation Backbone Sensitivity

为了公平比较 backbone，这一部分只比较不带 `SLR-C` 的蒸馏主线：

- teacher：`text_only`
- rationale：`full`
- gate：`sample_inverse`

结果：

| Backbone | Method | Macro | Micro | Samples | mAP | Hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CLIP ViT-L/14` | `dynamic_gated_kd` | **50.31** | **57.97** | **59.07** | **53.02** | **30.24** |
| `ResNet101` | `dynamic_gated_kd` | 39.35 | 44.73 | 46.91 | 39.08 | 21.53 |

ResNet101 结果来源：

- 训练 run：`logs/train/runs/2026-03-17_11-26-46`
- best checkpoint：`logs/train/runs/2026-03-17_11-26-46/checkpoints/epoch_012.ckpt`
- cache：`logs/analysis/resnet101_distill_cache_binarized_20260317`
- distillation output：`logs/analysis/resnet101_privileged_distillation_binarized_20260317`

结论：

1. backbone 质量对整条方法极其重要
2. teacher 没变，但弱 backbone 的 student 明显受限
3. 这说明当前方法不是“纯 teacher 决定一切”，视觉 backbone 仍是基本盘

#### 6.1.2 Full Method with ResNet101 Backbone

进一步，我们把完整方法也迁移到 `ResNet101` backbone 上：

- student backbone：
  - `ResNet101`
- fixed prior：
  - 仍然使用 CLIP cache 重建的 `SLR-C`
- teacher：
  - 仍然使用 `text-only teacher`

也就是说，这一版本的完整方法是：

\[
z^{stu}(x) = z^{slr-c}_{clip}(x) + r_\theta(v^{resnet}_x)
\]

结果目录：

- `logs/analysis/resnet101_distillation_slrc_20260318`

结果：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slr_c_fixed` | 42.25 | 51.84 | 51.17 | 43.08 | 23.83 |
| `slr_c_residual_sup` | **42.87** | **52.96** | **52.08** | 42.65 | 23.21 |
| `slr_c_residual_standard_kd` | 42.14 | 52.10 | 51.72 | **43.34** | 23.22 |
| `slr_c_residual_dynamic_kd` | 41.53 | 51.61 | 51.80 | 43.18 | 22.10 |

这组结果和 `ResNet101` plain distillation 一起看，可以得到更完整的结论：

1. `SLR-C prior` 依然能显著抬高弱 backbone 的整体上限
2. 但在 `ResNet101` 上，完整 residual + KD 结构没有稳定超过 fixed `SLR-C`
3. 说明最终完整方法虽然不是 CLIP-exclusive，但它确实更依赖一个较强的视觉 backbone 才能把 residual distillation 的价值完全释放出来

因此，backbone 消融的最终结论应写成：

> The proposed framework is portable to weaker backbones, but its full benefit is realized only when the visual backbone is already reasonably strong.

### 6.2 Distillation Ablation

不带 `SLR-C` 的蒸馏对照：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 47.40 | 56.42 | 56.41 | 50.67 | 28.45 |
| `standard_kd` | 49.59 | **58.26** | 58.88 | **53.18** | **30.44** |
| `dynamic_gated_kd` | **50.31** | 57.97 | **59.07** | 53.02 | 30.24 |

结论：

1. distillation 本身是成立的
2. `standard_kd` 和 `dynamic_kd` 各有侧重
3. dynamic gate 不是 universal winner，但它最稳地提升了 macro-level robustness

### 6.3 SLR-C Ablation

固定 `SLR-C` 先验上再训练 residual：

- `logs/analysis/distillation_slrc_20260317`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slr_c_fixed` | 51.08 | 58.81 | 58.02 | 53.66 | 33.98 |
| `slr_c_residual_sup` | 50.41 | 57.70 | 57.56 | 52.18 | 34.77 |
| `slr_c_residual_standard_kd` | **51.77** | 58.97 | 58.99 | 53.42 | **36.43** |
| `slr_c_residual_dynamic_kd` | 51.52 | **59.75** | **60.30** | **54.04** | 34.60 |

结论：

1. `SLR-C` 作为 fixed prior 和 distillation 是互补的
2. `SLR-C` 提供强 proposal
3. residual student + KD 负责做局部修正
4. 这是当前整套方法最强的版本

补充说明：

- 当 backbone 从 `CLIP ViT-L/14` 换到 `ResNet101` 时，`SLR-C + residual distillation` 仍然优于 plain ResNet distillation，
  但不能稳定超过 fixed `SLR-C`。
- 因此 `SLR-C` 在弱 backbone 场景下更像主要性能来源，而 residual correction 退化为次要补偿项。

### 6.4 Which Variant Should Go Into the Main Paper Table

如果主表只能放一个最终方法，建议优先用：

- `slr_c_residual_dynamic_kd`

理由：

1. `Micro / Samples / mAP` 全部最好
2. `Macro` 仍高于固定 `SLR-C`
3. `Hard` 也显著高于无 `SLR-C` 主线

如果论文特别强调 hardest classes，可以在正文或 supplementary 里补充：

- `slr_c_residual_standard_kd`

因为它给出了当前最强 `Hard = 36.43`。

---

## 7. Additional Negative Results

这些结果建议写成 appendix 或 supplementary：

### 7.1 Teacher Input

| Teacher Input | Dynamic Macro | Dynamic Hard |
| --- | ---: | ---: |
| `image + text` | 48.57 | 28.58 |
| `text_only` | **50.31** | **30.24** |

### 7.2 Rationale Source

| Text Source | Dynamic Macro | Dynamic Hard |
| --- | ---: | ---: |
| `full` | **50.31** | **30.24** |
| `step1_only` | 49.81 | 29.28 |
| `step1_step2` | 49.69 | 28.82 |

### 7.3 Gate Variants

| Variant | Dynamic Macro | Dynamic Hard |
| --- | ---: | ---: |
| `sample_inverse` | **50.31** | 30.24 |
| `classwise_inverse` | 49.54 | 30.13 |
| `classwise_entropy` | 49.12 | 29.95 |
| `classwise_dynamic_temperature` | 48.99 | **30.68** |
| `0.3 + 0.7 * (1 - omega)` | 49.24 | 28.27 |

### 7.4 Feature-Level Soft-SupCon

| Setting | Method | Macro | Hard |
| --- | --- | ---: | ---: |
| `without_feature` | `standard_kd` | 49.59 | 30.44 |
| `with_feature` | `standard_kd` | **50.74** | **31.29** |
| `without_feature` | `dynamic_gated_kd` | **50.31** | **30.24** |
| `with_feature` | `dynamic_gated_kd` | 49.68 | 29.71 |

结论：

1. Soft-SupCon 不适合当前 dynamic 主线
2. 但可能作为 standard KD 的 regularizer 保留

---

## 8. Reproducible Artifacts

主方法脚本：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

teacher / distillation baseline 脚本：

- [scripts/analyze_privileged_distillation.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_privileged_distillation.py)

案例可视化：

- [scripts/streamlit_baseline_student_mismatch.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/streamlit_baseline_student_mismatch.py)

---

## 9. Recommended Paper Framing

推荐把论文主方法写成：

> A strong scenario-aware frozen proposal system (`SLR-C`) plus a residual visual student,  
> regularized by a text-only rationale teacher through noise-aware distillation.

更具体地说：

1. `SLR-C` 负责全局 proposal 与 calibrated prior
2. residual student 负责视觉局部修正
3. text-only teacher 负责为 noisy / ambiguous samples 提供 softer supervision
4. final method 最终不是“teacher 替代 visual model”，而是“teacher + scenario prior + visual residual”的三方组合

如果要在论文里提前回应 backbone 依赖性，可以补一句：

> The method still works with weaker backbones such as ResNet101, but the gains become smaller and are dominated by the fixed SLR-C prior.
