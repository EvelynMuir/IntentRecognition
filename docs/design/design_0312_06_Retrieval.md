# 实验计划：Retrieval-Based Ambiguity Agent

## 0. 文档目的

本文档将 `retrieval-based ambiguity agent` 的初步想法整理为一份完整实验计划。

该方向的核心目标是：在当前 strongest proposer (`scenario SLR-C`) 已经较强、而静态 verifier / prototype / heavier post-hoc module 收益有限的背景下，验证下面这个问题：

> **是否可以通过“按需从训练记忆中检索支持与反对案例”的方式，为当前候选意图提供更具体、更动态的 comparative evidence，从而比静态 profile 式 verification 更有效地处理 candidate-local ambiguity？**

本文档重点回答：

1. retrieval 方向的研究动机是什么
2. memory bank 应该如何构建
3. 推理时 retrieval agent 应如何工作
4. 实验应该如何分阶段推进
5. 如何判断这条线是否值得继续

---

## 1. 背景与问题定义

当前已有结论可以总结为：

1. 强 frozen CLIP baseline 是系统性能基础
2. `scenario SLR-C` 是当前最稳定、最有效的主方法
3. 主要剩余误差集中在 **candidate-local ambiguity**
4. 现有静态 profile / prototype / heavier verifier 虽然在局部 hard cases 上有一定信号，但整体收益有限且不够稳定

这些结论说明：

- 仅用一个静态的 class-level evidence profile，可能难以覆盖复杂的类内多样性
- 仅用静态 summary，也可能不足以应对某张测试图像面对的具体局部歧义

因此，一个自然的新方向是：

> 不再为每个候选类维护固定 evidence template，而是在测试时，针对当前候选意图，主动从训练记忆中检索支持该候选的案例，以及反驳/竞争该候选的案例，再基于这些具体案例做 comparative verification。

这可以形成一个更清楚的 agent 流程：

```text
propose -> retrieve evidence -> compare candidates -> revise belief
```

---

## 2. 核心研究假设

### Hypothesis 1
静态 class-level evidence profile 会压缩掉类内多样性，而 retrieval 可以保留更具体的 case-level visual patterns。

### Hypothesis 2
对于 hard ambiguity，真正有用的不是“该类的一般性平均证据”，而是“与当前测试图像最相似的支持案例与竞争案例”。

### Hypothesis 3
confusion-aware refute retrieval（从竞争意图中检索反例）会比全局 negative retrieval 更适合 candidate-local disambiguation。

### Hypothesis 4
如果 retrieval 方向有效，其收益应首先体现在：

- `Hard`
- `Macro`
- pairwise / top-2 disambiguation diagnostics
- 可解释的 case study

---

## 3. 总体方法框架

建议将该方向统一命名为：

> **Retrieval-Based Ambiguity Agent**

或在更偏论文的语境下命名为：

> **Case-Based Comparative Verification for Visual Intent Recognition**

总体流程如下：

```text
Image
  -> Strong proposer (scenario SLR-C)
  -> Top-k candidate intents
  -> For each candidate:
       retrieve supporting exemplars
       retrieve competing / refuting exemplars
  -> Comparative evidence aggregation
  -> Gated belief revision
  -> Final reranking
```

它与现有 static verifier 的最大差异在于：

- static verifier：基于预先统计好的 class-level profile 做判断
- retrieval agent：在测试时按需访问训练记忆，用真实案例作为动态证据来源

---

## 4. Memory Bank 设计

## 4.1 基本 memory entry

对训练集每张图像 `x_n`，建立一个 memory item：

$$
M_n = \{v_n, y_n, e_n\}
$$

其中：

- `v_n`：图像 embedding（优先使用 CLIP visual embedding）
- `y_n`：intent 标签集合
- `e_n`：轻量 evidence summary（可选，但建议预留）

### 第一轮必存内容

- image embedding
- labels

### 推荐预留内容

若已有 cache，建议同时保存一个轻量 evidence summary，例如：

- object top-m labels / scores
- scene top-m labels / scores
- style top-m labels / scores
- activity top-m labels / scores

这样后续 Phase 3 可以自然扩展到 evidence-aware retrieval。

---

## 4.2 Memory 组织方式

最简单的 MVP 先按 intent 建索引。

对于每个 intent `c`，建立：

### Support memory

所有满足 `y_{n,c}=1` 的训练样本：

$$
\mathcal{M}_{support}(c)
$$

### Refute memory

有两种版本：

#### 版本 A：global negative refute
所有不包含 `c` 的训练样本

#### 版本 B：confusion-aware refute
只从 `c` 的 confusion neighborhood 中取反证样本，即：

- 若某类 `\tilde c \in \mathcal{N}(c)`
- 且样本带有 `\tilde c`
- 则进入 `c` 的 refute memory

建议 Phase 1 先做 global negative，Phase 2 再转向 confusion-aware refute。

---

## 5. 推理时的 Retrieval Agent 流程

## 5.1 Step 1：Candidate proposal

保留当前 strongest proposer：

- `scenario SLR-C`

输出：

- 全部 intent scores
- top-k candidates `C = {c_1, ..., c_k}`

---

## 5.2 Step 2：Query encoding

对测试图像 `x` 计算：

- image embedding `v_x`
- （可选）evidence summary `e_x`

### MVP 建议
第一版只依赖：

- `v_x`

避免一开始就把 evidence-aware retrieval 做得过重。

---

## 5.3 Step 3：Candidate-wise retrieval

对于每个候选 intent `c_i`，检索两类 exemplars：

### Supporting exemplars
从 `c_i` 的 support memory 中，取与当前图像最相似的 top-r 个样本：

$$
\mathcal{S}(c_i, x) = Top\text{-}r\;NN\;in\;\mathcal{M}_{support}(c_i)
$$

### Refuting exemplars
从 `c_i` 的 refute memory 中，取与当前图像最相似的 top-r 个样本：

$$
\mathcal{R}(c_i, x) = Top\text{-}r\;NN\;in\;\mathcal{M}_{refute}(c_i)
$$

### 初始 similarity
优先使用最简单版本：

- cosine similarity between image embeddings

即：

$$
sim(x,n)=cos(v_x,v_n)
$$

---

## 5.4 Step 4：Case-based evidence scoring

对每个候选 `c_i`，基于支持样本与反证样本构造 case-based evidence score。

### 最小版本：support minus refute

$$
E(c_i, x)=
\frac{1}{r}\sum_{n\in \mathcal{S}(c_i,x)} sim(x,n)
-
\frac{1}{r}\sum_{m\in \mathcal{R}(c_i,x)} sim(x,m)
$$

直觉是：

- 如果当前图像更像支持 `c_i` 的样本，而不像反对 `c_i` 的样本，则该候选更可信

### 为什么这样设计
这一步用最小代价把 retrieval 转成可直接与当前 comparative verifier 对接的 evidence score。

---

## 5.5 Step 5：Comparative candidate verification

当前 strongest insight 是：candidate-local ambiguity 本质上是 comparative 问题。

因此，不建议 retrieval 只作为单类绝对加分项；建议显式支持 comparative form。

对于 top-k 中任意一对候选 `(c_i, c_j)`，定义：

$$
\Delta E(c_i,c_j,x)=E(c_i,x)-E(c_j,x)
$$

解释为：

- 当前图像到底更接近 `c_i` 的支持案例，还是更接近 `c_j` 的支持案例

### MVP 建议
若一开始不想做 full pairwise，可先做：

- top-1 vs top-2
- 或每个候选仅相对当前 strongest competitor

---

## 5.6 Step 6：Gate / belief revision

延续当前已经验证有效的 gate 思路。

例如基于 candidate margin：

$$
g(x)=f(margin(x))
$$

其中：
- margin 小 -> gate 大
- margin 大 -> gate 小

最终更新候选分数：

$$
s^{final}(c_i,x)=s^{base}(c_i,x)+\alpha\cdot g(x)\cdot E(c_i,x)
$$

若使用 comparative version，则让 `\Delta E` 参与 pairwise reranking。

---

## 6. 分阶段实验设计

## Phase 1：最小可行 Retrieval Agent

### 6.1 目标

验证：

> retrieval-based case evidence，是否至少可以接近或超过当前 static comparative verifier。

### 6.2 做什么

- memory bank 只存 image embeddings + labels
- retrieval similarity 只用 image embedding cosine similarity
- support / refute exemplars 各取 top-r
- evidence score = support mean sim - refute mean sim
- 加到 `scenario SLR-C` 上做 gated rerank

### 6.3 对照组

1. `scenario SLR-C`
2. `v2 comparative + gate(add)`
3. `SLR-C + retrieval agent (image-only)`

### 6.4 超参数

建议先试：

- `r = 3, 5, 8`
- gate 复用当前 strongest setting
- candidate top-k 与当前 strongest setting 一致

### 6.5 成功信号

若 retrieval-based image-only case evidence：
- 至少接近 `v2`
- 或在 `Hard / Macro` 上优于 static verifier

则说明 retrieval 路线值得继续。

---

## Phase 2：Confusion-Aware Retrieval

### 7.1 目标

验证：

> refute evidence 应来自竞争意图，而不是泛化的全局负样本。

### 7.2 做什么

使用 confusion matrix 为每个 intent 构建 confusion set `\mathcal{N}(c)`。

refute retrieval 不再从全局 negative memory 取样，而只从 confusion set 中检索。

### 7.3 对照组

1. global negative retrieval
2. confusion-aware retrieval

### 7.4 成功信号

若 confusion-aware retrieval 比 global negative 更强，则支持：

> retrieval 的价值来自 candidate-local ambiguity resolution，而不是粗粒度近邻匹配。

---

## Phase 3：Evidence-Aware Retrieval

### 8.1 目标

验证：

> retrieval 是否还能利用 object / scene / style / activity 证据，而不仅仅是全图 embedding 相似性。

### 8.2 做什么

在 memory item 中引入 evidence summary：

- image embedding
- evidence summary

并定义混合相似度：

$$
sim(x,n)=\lambda_v cos(v_x,v_n)+\lambda_e cos(e_x,e_n)
$$

### 8.3 对照组

1. image-only retrieval
2. image + evidence retrieval

### 8.4 成功信号

若 image+evidence retrieval 稳定优于 image-only，则说明 retrieval agent 不只是 nearest-neighbor 语义匹配，而是真正在用更结构化的证据。

---

## 7. 主实验表设计

## 7.1 主结果表

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` |  |  |  |  |  |
| `v2 comparative + gate(add)` |  |  |  |  |  |
| Retrieval agent Phase 1 |  |  |  |  |  |
| Retrieval agent Phase 2 |  |  |  |  |  |
| Retrieval agent Phase 3 |  |  |  |  |  |

### 主表目标
判断 retrieval-based dynamic case evidence 是否比静态 verifier 更值得继续。

---

## 7.2 检索设置消融

| Retrieval Setting | Macro | Hard |
| --- | ---: | ---: |
| support only |  |  |
| support - global refute |  |  |
| support - confusion refute |  |  |

### 目标
判断 retrieval 的关键是：
- 只找支持案例
- 还是同时显式找反驳/竞争案例

---

## 7.3 r 消融

| r | Macro | Micro | Hard |
| --- | ---: | ---: | ---: |
| 3 |  |  |  |
| 5 |  |  |  |
| 8 |  |  |  |

### 目标
确定检索深度的合理范围。

---

## 7.4 similarity 消融

| Similarity | Macro | Hard |
| --- | ---: | ---: |
| image only |  |  |
| evidence only |  |  |
| image + evidence |  |  |

### 目标
判断 retrieval 是否需要结构化 evidence summary 才能真正发挥作用。

---

## 7.5 Hard subset / pairwise diagnostics

额外报告：

- low-margin subset
- top confusion pair subset
- pairwise ranking accuracy
- top-2 disambiguation accuracy

### 目标
验证 retrieval agent 是否真的在 candidate-local ambiguity 上工作，而不是只在全局平均指标上偶然波动。

---

## 8. 推荐的 case study

如果 retrieval 方向有效，case study 会非常有说服力。

建议展示：

1. 当前测试图像
2. `SLR-C` 的 top-k candidates
3. 每个关键候选检索到的 support exemplars
4. 检索到的 competing / refuting exemplars
5. retrieval-based evidence 如何改变候选排序

这类案例天然具有“agent 主动取证并修正 belief”的解释性。

---

## 9. 成功信号与停止条件

## 9.1 成功信号

若该方向有效，通常至少会出现以下现象中的两项：

1. `Hard` 提升
2. `Macro` 提升
3. pairwise ranking / top-2 disambiguation 提升
4. confusion-aware retrieval 明显优于 global negative retrieval
5. case study 清楚显示动态案例比静态 profile 更合理

## 9.2 停止条件

若出现以下情况，则不建议继续深入：

1. image-only retrieval 明显不如当前 `v2 comparative + gate`
2. confusion-aware retrieval 相比 global negative 没有明显增益
3. retrieval 主要只是在复制近邻相似性，而没有提供更好的 comparative disambiguation
4. image + evidence retrieval 的收益仍然很弱，不足以支撑 memory-based 方法复杂度

---

## 10. 风险与注意事项

## 风险 1：memory 太大、检索开销高

建议：
- Phase 1 先只做按 intent 索引
- top-r 检索保持小范围
- 必要时可先缓存每类近邻候选

## 风险 2：retrieval 只学到“更像什么图”，而不是“更像什么意图证据”

因此必须做：
- support vs support-refute 对照
- global negative vs confusion-aware refute 对照

## 风险 3：retrieval 与现有 verifier 方向高度重合，实际增益有限

这正是本实验要验证的关键。如果 retrieval 不能比当前 static verifier 更强，就应及时停止，而不要继续堆更多 memory 设计。

---

## 11. 推荐的最小可行版本（MVP）

第一轮建议只做下面这个最小闭环：

1. base = `scenario SLR-C`
2. memory = 训练集 image embeddings + labels
3. 对 top-k candidate intents
4. 每个 candidate 检索 top-r support exemplars
5. 每个 candidate 从 confusion set 中检索 top-r refute exemplars
6. score = mean support sim - mean refute sim
7. 经 gate 后加回 base score

不做：

- region grounding
- graph reasoning
- learned controller
- evidence-aware similarity

这个 MVP 已经足够回答：

> retrieval-based case evidence 是否比当前 static verifier 更有前景？

---

## 12. 一句话总结

这个实验方向本质上是在验证：

> **能否把静态 evidence verification 改写成一个 case-based ambiguity agent：系统先提出候选意图，再主动从训练记忆中检索支持与反驳案例，并基于案例证据更新最终 belief。**
