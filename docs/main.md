# Intent Recognition

## 1. 目标

我的目标是完成一篇期刊投稿，任务聚焦于 **Intentonomy** 上的图像意图识别。

我希望方法同时满足以下三点：

- **性能超过现有 SOTA**，尤其是 `macro F1` 要有明显提升。
- **方法具有足够 novelty**，不能只是把已有的 `CLIP / prompt / prototype / graph` 方法直接套到任务上。
- **论文叙事足够稳**，最好围绕任务中的真实难点提出统一方法，而不是堆很多小 trick。

目前我已经有一个很强的 baseline，并且大幅超过现有 SOTA。因此，当前更关注的问题是：

> 在不破坏强 baseline 的前提下，找到一个既有创新点、又能稳定涨点的方法。

---

## 2. 任务与现状

### 2.1 任务定义：Intentonomy

这是一个 **multi-label intent recognition** 任务，数据集为 **Intentonomy**。该任务具有以下特点：

- 标签是 **intent**，而不是 `object / scene / action`。
- `intent` 具有很强的主观性、语义抽象性和类间相似性。
- 不同 intent 往往会发生“合理混淆”：即使 prediction 与 ground truth 不完全一致，也可能在语义上说得通。
- 训练标签来自多人标注，原始标签是 **soft label**（例如 `1/3`、`2/3`、`1`），更像标注一致性，而不是标准概率。
- 类别分布存在一定长尾。
- 评价指标以 `macro F1` 为主，此外还有 `micro F1` 和 `samples F1`。

### 2.2 现有方法 / SOTA

我目前掌握的较强 SOTA 是一种 **LLM + retrieval + multimodal fusion** 方法，大致流程如下：

1. 用 `Mini-GPT4` 对图像生成文字描述。
2. 用类似 `RAG` 的方式检索 3 个表征相似的图像。
3. 将以下信息一起送入改装后的 LLM：
   - 图像 `visual tokens`
   - `description text tokens`
   - `learnable prompt tokens`
   - 相似图像的 `visual tokens` 和 `labels`
4. 将 LLM 的 `text decoder` 改造成一个可训练线性层，输出 intent 类别和概率。

公开结果大约如下：

| 指标 | 分数 |
| --- | ---: |
| `macro F1` | 43.05 |
| `micro F1` | 54.77 |
| `samples F1` | 56.75 |

### 2.3 相关工作：PIP-Net

我还看过一篇与 Intentonomy 相关的 prototype 方法：**PIP-Net**（Prototype-Based Intent Perception）。

其核心思路包括：

- 每个类学习多个 prototype。
- 丢弃离中心较远的样本。
- 进行 prototype 聚类、动量更新和多样性约束。

但它属于 **class-specific prototype learning**，与我后来更想做的 **shared concept / structure-aware** 路线并不一致。

---

## 3. 已尝试的方法与结果

### 3.1 当前最强 baseline

我目前最强的 baseline 配置如下：

- Backbone：`frozen CLIP ViT-L/14`
- Feature：`global feature`
- 具体形式：`CLS + patch mean`
- Head：`2-layer MLP`
- Loss：`Optimized Asymmetric Loss (ASL)`
- Inference：测试时做 `threshold search`

当前最好结果如下：

| 方法 | `macro F1` |
| --- | ---: |
| Baseline | 46.38 |

这个结果显著高于现有 SOTA `43.05`。

此外我已经验证过：

| Feature 形式 | 相对表现 |
| --- | --- |
| `CLS + patch mean` | 最好 |
| `CLS` | 次之 |
| `patch mean` | 更弱 |

这说明：

- `CLIP` 的 global semantic representation 已经非常强。
- `patch mean` 可以提供一定补充信息。
- 复杂结构未必有帮助。

### 3.2 Slot / Random / Concept / Calibration 方向的尝试

#### 3.2.1 Slot Attention / intent-conditioned slot

最开始我尝试过一类 `slot-based / concept-like` 方法，希望做更细粒度的 intent-aware decomposition，包括：

- `slot attention`
- `intent-conditioned slot attention`
- `orthogonal loss`
- `residual fusion`

结果总体都不如 baseline。

一些关键观察如下：

- `slot` 甚至不如 `random grouping`
- 去掉 `orthogonal loss` 后，`random` 分组很强
- 后来确认这里的 `random` 是真实随机 chunk，而不是固定 spatial chunk

最终排序大致为：

```text
global > random > slot
```

这说明：

- 对于 `Intentonomy + frozen CLIP-L/14`，复杂分组式建模并不占优。
- `global semantic` 仍然是更关键的信息来源。

#### 3.2.2 Intent-conditioned residual calibration

我还设计过一种 `intent-conditioned residual weighted pooling / calibration` 方法，核心做法为：

1. 让 `patch tokens` 与 `intent query` 做 attention。
2. 用 attention-pooled patch feature 去微调全局 mean。
3. 再与 `CLS` 融合后进行分类。

性能明显低于 baseline。之后我还额外尝试过：

- `co-occurrence linear layer`
- `vectored lambda`

结果更差。代表性结果如下：

| 设置 | `macro F1` |
| --- | ---: |
| Baseline | 46.38 |
| Calibration | 40.98 |
| Co-occurrence | 36.33 |

结论是：

> 这条路线结构方向本身不对，会系统性伤害强 baseline。

### 3.3 Semantic concept / ICRN 方向

后来我设计了一种带有 **ICRN（Intent Concept Reasoning Network）** 风格的方法，其思路为：

```text
image -> concept grounding -> concept graph reasoning -> intent composition
```

并且已经完成了如下准备工作：

- 生成了一整套 `shared concept pool`
- 为每个 intent 生成了对应的 concepts 及其权重
- concept 是 **类间共享** 的，而不是每类独立

后来我做了以下修改：

- 改成 `logit-space concept`，不在中间直接做 `sigmoid bottleneck`
- 增加 `concept-MLP`，提高头部容量
- 将图传播改成更可控的形式
- 弱化或延迟 `prior regularization`

ICRN 的最好 `macro F1` 也只有：

| 方法 | `macro F1` |
| --- | ---: |
| Baseline | 46.38 |
| ICRN | 43.42 |

之后我还尝试将 concept branch 作为 residual semantic branch 加到 baseline 上：

```text
baseline logits + alpha * concept logits
```

结果依然会掉点。掉得最少的一组设置是：

- `baseline lr = 0`
- `alpha lr = 1e-4`
- `concept lr = 5e-4`

这说明：

- `concept branch` 最多只能作为很弱的扰动
- 它更像噪声，而不是补充信息

因此目前可以得出结论：

> 无论作为主干分类器还是 residual branch，`ICRN / concept reasoning` 目前都不如 baseline。

### 3.4 LLM semantic weighted loss / semantic prior 的尝试

我还尝试过 `semantic weighted loss`，基本形式为：

```text
semantic_sim = text_embeddings @ text_embeddings.T
semantic_weight = 1 - semantic_sim
```

然后在 `ASL` 上做加权，结果是：**性能略降**。

后续分析认为，原因主要在于：

- semantic similarity 会让模型对“语义相近类”的错误惩罚变小
- 但在 `F1` 评价下，相似类依然是错类
- 因此它实际上会促进相似类别的混淆，而不是提升分类性能

此外我还确认过：

- Intentonomy 原始 `soft label` 本身不是好的 semantic soft target
- 直接用 `soft label` 训练不如 `binarize`
- 使用 `binarized targets` 反而更好

### 3.5 当前阶段的重要经验结论

从目前所有实验来看，我已经比较明确地知道以下几点：

- `frozen CLIP ViT-L/14 + global feature` 非常强
- 对这个任务来说，`global semantic mapping` 往往比局部结构建模更重要
- 以下复杂设计都容易伤害 baseline：
  - `slot`
  - `concept bottleneck`
  - `patch attention calibration`
  - `semantic weighted loss`
  - `co-occurrence linear layer`

换句话说，Intentonomy 的关键难点更像是：

- 监督不确定性
- 结构化标签空间
- 类间合理混淆

而不是视觉特征表达本身不够。
