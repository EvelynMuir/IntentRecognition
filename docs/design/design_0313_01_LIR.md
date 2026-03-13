# 实验计划：LIR — Latent Intent Basis with Label Refinement

## 0. 文档目的

本文档将 `LIR` 的初步设想整理为一份可以直接开做的实验计划文档。

LIR 的核心目标不是继续增加一个更复杂的后验 verifier，而是验证下面两个更基础的问题：

1. 当前任务真正有效的判别单元，是否并不是固定的 `object / scene / style / activity` 语义 bank，而应当是**数据驱动的潜在判别基元**。
2. Intentonomy 的监督是否存在较强噪声与不确定负类，使得任何基于硬标签的证据建模都会受到明显限制，因此需要**软标签修复**。

因此，LIR 由两个核心部分组成：

- **Latent Intent Basis**：学习一组数据驱动、可稀疏组合的潜在意图基元
- **Label Refinement**：把标签从硬 0/1 观测修正为带有不确定性的软监督目标

---

## 1. 背景与问题定义

当前已有实验结果说明：

1. 强 frozen CLIP baseline 是当前系统性能基础
2. `scenario SLR-C` 是最稳定、最有效的主系统
3. fixed semantic bank、comparative verifier、region-grounded agent 等方向虽然在 hard ambiguity 上有局部信号，但整体收益有限且不稳定

这提示两个更本质的问题：

### 问题 A：固定语义 bank 可能不是合适的判别坐标系

如果真正有效的判别因素并不天然对应人工定义的 `object / scene / style / activity`，那么继续在固定 bank 上做更复杂的 verifier，收益就会受限。

### 问题 B：标签监督可能存在显著噪声

Intentonomy 中存在：
- 同一视觉内容对应多个可能意图
- 同一意图对应多种视觉模式
- annotator 未标全 / 模糊负类 / 近邻标签混淆

在这种情况下，直接把所有未标注标签当成真实负类，可能会系统性压制模型学习。

因此，LIR 的目标是：

> 用数据驱动的 latent basis 替代固定 semantic bank，用软标签修复缓解 noisy / ambiguous negatives，并重新验证该任务的有效判别单元与监督方式。

---

## 2. 核心研究假设

### Hypothesis 1
固定 semantic bank 失败，不代表“分解式判别”无效，而更可能说明：**固定 bank 的坐标系不适合该任务**。

### Hypothesis 2
从 patch tokens 学习一组共享、可稀疏组合的 latent intent basis，会比固定 semantic bank 更适合表达类内多样性与跨类复用判别因素。

### Hypothesis 3
hierarchy 更适合作为连接与约束，而不是直接作为强分层 decoder 主干。

### Hypothesis 4
若 soft label refinement 有效，则说明 noisy negatives / missing positives 是当前瓶颈之一。

---

## 3. 总体方法框架

建议方法名统一为：

> **LIR: Latent Intent Basis with Label Refinement**

总体结构如下：

```text
Image
  -> CLIP ViT-L/14 backbone
  -> CLS token + multi-layer patch tokens
  -> Latent Intent Basis module
  -> Global + Basis classification head
  -> Optional hierarchy residual branch
  -> Label refinement / soft supervision
  -> Final multi-label prediction
```

与前面各类 verifier 方法的主要差异在于：

- verifier 系列：主要在 inference / reranking 侧修正候选排序
- LIR：直接重构前端判别表示与训练监督方式

---

## 4. 输入与 Backbone 设置

## 4.1 Backbone

建议优先使用当前已经稳定的 backbone：

- CLIP `ViT-L/14`
- frozen 或部分解冻（第一轮建议 frozen）

## 4.2 输入特征

建议使用：

- `CLS token`
- 最后 `2–4` 层 patch tokens

### 原因
LIR 要验证的是“更细粒度的 latent 判别基元是否有效”，因此不能只依赖 CLS，否则方法很容易退化成另一个线性分类头。

---

## 5. 模块一：Latent Intent Basis

## 5.1 目标

从 patch tokens 中学习一组共享但可稀疏组合的 latent basis：

$$
B = \{b_1, b_2, ..., b_K\}, \quad b_k \in \mathbb{R}^d
$$

这里的 basis 不预先绑定到 object / scene / style / activity，而由数据驱动学出。

## 5.2 可实现方案

建议保留两种实现候选，但第一轮先只做一种最小版本。

### 方案 A：Slot-style latent basis

- 初始化 `K` 个 learnable basis queries
- basis queries 与 patch tokens 做 cross-attention
- 得到每个 basis 对当前图像的响应
- 对 basis activation 做 sample-wise sparse selection

### 方案 B：Dictionary routing

- patch tokens 经 router 得到对 `K` 个 basis 的 assignment
- basis 作为 dictionary atoms 被加权组合

## 5.3 第一轮推荐

第一轮建议先做：

- **Slot-style latent basis**

因为：
- 实现更直接
- 更容易做可视化与 basis 使用分析
- 更适合作为最小可行验证版本

## 5.4 输出表示

得到图像的 basis 表达：

$$
z^{basis} = \sum_{k=1}^{K} \alpha_k b_k
$$

其中 `\alpha` 应满足稀疏性约束。

## 5.5 关键约束

为了避免 latent basis 塌缩为普通全连接层，必须加入：

### (1) 稀疏约束

每张图只激活少量 basis，例如：

$$
\mathcal{L}_{sparse}=||\alpha||_1
$$

或使用 top-r routing regularization。

### (2) 多样性 / 去相关约束

避免所有 basis 学成相似模板：

$$
\mathcal{L}_{div}=||BB^\top - I||_F
$$

---

## 6. 模块二：层级条件化分类头

这一部分不建议完全复刻复杂 hierarchical decoder，而建议做轻量 residual-style hierarchy conditioning。

## 6.1 Label embeddings

为 coarse / middle / fine 三层定义 learnable label embeddings：

$$
Q^c, Q^m, Q^f
$$

这些 embedding 必须是 learnable 的，用来增强 class-related label representation。

## 6.2 分类方式

对 fine label `y_j`，其 logit 不只来自全局特征，而是：

$$
s_j = s_j^{global} + s_j^{basis} + s_j^{hier}
$$

其中：

- `s_j^{global}`：baseline 全局分支
- `s_j^{basis}`：latent basis 分支
- `s_j^{hier}`：来自 coarse/middle label embeddings 的层级残差

## 6.3 设计原则

hierarchy 在 LIR 中的作用是：

- 做连接与约束
- 帮助不同层级间信息交互

而不是直接替代 fine-grained 主判别器。

---

## 7. 模块三：Label Refinement

## 7.1 动机

当前多标签监督中的一部分负类可能实际上属于：

- missing positives
- uncertain labels
- 与正类高度邻近的 ambiguous negatives

因此，不应把所有 observed negatives 都当成真实硬负类。

## 7.2 软标签形式

定义 soft target：

$$
\tilde y = \lambda_1 y_{obs} + \lambda_2 y_{teacher} + \lambda_3 y_{graph} + \lambda_4 y_{nn}
$$

其中：

- `y_obs`：原始观测标签
- `y_teacher`：EMA teacher 输出
- `y_graph`：标签图传播结果
- `y_nn`：近邻样本标签支持

## 7.3 组成建议

### 第一阶段
先做最简单版本：

- `EMA teacher soft target`

### 第二阶段
再逐步加入：

- graph prior
- neighbor support

## 7.4 不确定负类处理

把 negative labels 分为两类：

### reliable negative
满足：
- logit 很低
- teacher 低
- neighbors 也不支持

→ 使用正常 negative loss

### uncertain negative
满足：
- teacher 不低
- neighbors 有支持
- confusion / hierarchy graph 相邻

→ 降低 negative loss 权重，或视为 weak negative

---

## 8. 损失函数设计

总损失建议写为：

$$
\mathcal{L}=
\mathcal{L}_{cls}
+ \lambda_s \mathcal{L}_{sparse}
+ \lambda_d \mathcal{L}_{div}
+ \lambda_h \mathcal{L}_{hier}
+ \lambda_u \mathcal{L}_{soft}
$$

其中：

## 8.1 主分类损失 `\mathcal{L}_{cls}`

建议优先继续使用你当前熟悉的：

- `Asymmetric Loss`

## 8.2 层级一致性损失 `\mathcal{L}_{hier}`

让 fine 的预测聚合后与 middle / coarse 层预测保持一致，可采用：

- KL consistency
- BCE consistency

## 8.3 软标签损失 `\mathcal{L}_{soft}`

对 uncertain labels 使用 soft target，而不是硬负监督。

---

## 9. 关键验证问题与实验设计

## Validation 1：fixed bank 失败，究竟是 bank 的问题还是分解思路的问题？

### 比较
1. Global baseline
2. Fixed semantic bank
3. Learnable latent basis（不加 label refinement）

### 关注指标
- Macro
- Hard
- rare classes
- confusion pairs

### 目标
若 `latent basis > fixed bank`，则说明：

> 问题不在“分解”本身，而在“固定 bank 坐标系不适合该任务”。

---

## Validation 2：后验 verifier 失败，是否说明前端表征才是更关键的？

### 比较
1. baseline
2. baseline + comparative verifier
3. latent basis classifier
4. latent basis classifier + lightweight pair margin loss

### 目标
若 `3` 明显优于 `2`，则说明：

> 真正有用的是前端表征重构，而不是继续做更复杂的后验 verifier。

---

## Validation 3：标签噪声是否是当前瓶颈之一？

### 比较
1. hard labels only
2. EMA teacher soft labels
3. EMA + graph prior
4. EMA + graph + neighbor support

### 关注指标
- Hard F1
- calibration
- recall
- label cardinality
- confusion pairs precision/recall

### 目标
若加 soft label 后 Hard / recall / calibration 提升，则说明 noisy negatives 是当前瓶颈之一。

---

## Validation 4：hierarchy 应放在哪里最有效？

### 比较
1. no hierarchy
2. hierarchy only as auxiliary supervision
3. hierarchy as residual conditioning
4. hierarchy as strong decoder

### 目标
验证 hierarchy 更适合做连接与约束，而不是直接做主干判别器。

---

## 10. 实验清单

## 10.1 Baselines

### 基础 baseline
- CLS + linear head
- CLS + MLP head
- patch mean pooling + linear head

### 已有强方法
- `scenario SLR-C`
- `comparative verifier best`
- `fixed semantic bank best`
- （如已完成）retrieval / confusion-aware / hierarchy-aware 版本

这样可以让后续论文叙事更完整：LIR 不是凭空提出，而是在前面多条线都验证过之后收敛出的方向。

---

## 10.2 主方法消融

### A. Latent basis 结构消融
- `K = 16 / 32 / 64 / 128`
- active basis 数 `r = 2 / 4 / 8`
- 使用最后 `1 / 2 / 4` 层 patch tokens
- slot-style vs dictionary routing
- 有无稀疏约束
- 有无去相关约束

### B. 分类头消融
- global only
- basis only
- global + basis
- global + basis + hierarchy residual

### C. Label refinement 消融
- 无 refinement
- teacher only
- teacher + graph
- teacher + graph + neighbor support
- uncertain negative down-weight on/off

### D. hierarchy 消融
- no hierarchy
- coarse/fine only
- coarse/middle/fine
- auxiliary-only vs residual-conditioning

---

## 10.3 关键对比实验

### 对比 1：fixed bank vs learnable basis
重点看：
- Macro
- Hard
- confusion subset

### 对比 2：一阶段 latent basis vs 二阶段 verifier
重点看：
- hard classes
- calibration
- training stability

### 对比 3：hard labels vs soft labels
重点看：
- recall
- rare class F1
- label cardinality

---

## 10.4 诊断实验

### 1. Basis usage statistics
统计：
- 每类样本平均激活多少个 basis
- basis 的跨类复用情况

### 2. Basis-label similarity visualization
画出：
- basis 与 fine labels 的相似度矩阵
- basis 与 hierarchy 标签的关联

### 3. t-SNE / UMAP
比较：
- baseline feature
- fixed bank feature
- latent basis feature

### 4. Calibration / reliability diagram
看 soft label refinement 是否改善过度自信 negative judgment。

### 5. Error breakdown
分析：
- improvement 最大的类
- improvement 最小的类
- 仍然失败的 confusion pairs

---

## 10.5 泛化与鲁棒性

建议补以下实验：

### 1. 少样本类表现
按样本数将类别分为：
- head / medium / tail

### 2. 部分标签缺失模拟
随机 drop 一部分 positive labels，比较不同方法鲁棒性。

### 3. Threshold sensitivity
比较不同 threshold / class-wise threshold 下的稳定性。

### 4. 多随机种子
至少 `3` 次，报告均值和方差。

---

## 11. 推荐推进顺序

## Phase 1：先验证 latent basis 是否比 fixed bank 强

只做：
- baseline
- fixed bank best
- latent basis

暂时不加 label refinement。

### 目标
先回答：

> 分解式判别在 learnable basis 下是否成立？

若 latent basis 连 fixed bank 都打不过，则这条线应立即重新评估。

---

## Phase 2：在 latent basis 上加入 hierarchy residual

只做轻量 residual-style hierarchy，不上复杂 decoder。

### 目标
验证：

> hierarchy 作为连接与约束是否有效。

---

## Phase 3：加入 soft label refinement

从最简单的 teacher soft target 开始，再逐步加 graph / neighbor。

### 目标
验证：

> noisy negatives 是否是当前瓶颈之一。

---

## Phase 4：补完整消融与诊断

包括：
- basis 可视化
- calibration
- label noise 模拟
- rare / hard classes 分析

---

## 12. 预期结果模式

如果 LIR 方向成立，通常应看到如下趋势：

### 预期 1
`latent basis > fixed bank`

即使只提升 `0.3–0.8`，也足以说明“固定 bank 坐标系不对”。

### 预期 2
`latent basis + hierarchy residual > latent basis only`

说明 hierarchy 适合作为连接与约束。

### 预期 3
`latent basis + soft label refinement` 在 `Hard` 上的涨幅大于 `Macro`

说明 noisy negatives 与 hard ambiguity 有直接关联。

### 预期 4
soft label refinement 改善 recall 与 calibration，而不是只是堆 precision。

---

## 13. 成功信号与停止条件

## 13.1 成功信号
若该方向有效，通常至少应看到：

1. latent basis 稳定优于 fixed bank
2. hierarchy residual 带来小幅但稳定增益
3. soft label refinement 改善 Hard / recall / calibration
4. basis 使用分析显示：basis 既不是 per-class memorization，也不是完全塌缩

## 13.2 停止条件
若出现以下情况，则不建议继续深入：

1. latent basis 明显不如 fixed bank 或 baseline
2. 稀疏 basis 学不出稳定结构，严重塌缩
3. hierarchy 只会带来干扰，没有稳定正收益
4. soft label refinement 对 Hard / calibration 没有明确帮助

---

## 14. 最小可行版本（MVP）

第一轮建议只做这个最小闭环：

### 模型
- CLIP `ViT-L/14` patch tokens
- `K = 32` latent basis
- top-4 sparse routing
- global + basis classifier
- 不加 hierarchy
- 不加 label refinement

### 第一组实验
- baseline
- fixed semantic bank best
- latent basis

### 第二组实验（若第一组有正信号）
在 latent basis 上增加：
- hierarchy residual
- EMA soft target

这个 MVP 已经足够回答：

> 数据驱动的 latent basis 是否比固定 semantic bank 更适合作为当前任务的判别单元？

---

## 15. 一句话总结

LIR 这条线本质上是在验证：

> **能否用数据驱动的潜在意图基元替代固定 semantic bank，并通过软标签修复缓解 noisy supervision，从而比现有 verifier / bank / retrieval 分支更直接地重构该任务的有效判别表示。**
