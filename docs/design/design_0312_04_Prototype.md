# 实验计划：Prototype Evidence Memory for Candidate-Local Comparative Verification

## 0. 文档目的

本文档将 `prototype evidence memory` 的初步想法整理为一份可直接执行的实验计划。

目标不是重做整条主方法线，而是在当前已有 strongest base system 的基础上，验证下面这个问题：

> 当单一 intent-level evidence profile 难以覆盖类内多样性时，是否可以通过为每个 intent 建立多个 prototype evidence patterns，在 candidate-local comparative verification 中获得更稳定的局部消歧能力？

本文档面向实验设计，因此重点回答：

1. 研究假设是什么
2. 方法要怎么最小改动实现
3. 应该先跑哪些实验
4. 如何判断这个方向是否值得继续

---

## 1. 背景与问题定义

当前已有方法线可以概括为：

1. 用 frozen CLIP `ViT-L/14` 建立强 visual baseline
2. 用 `scenario` text prior 做 local rerank，得到 `SLR-C`
3. 在 top-k 候选上加入 comparative evidence verification

现有 verifier 的一个默认前提是：

> 每个 intent 可以被一个统一的 evidence profile 充分刻画。

但这一前提未必成立。因为很多高层 intent 本身具有明显的**类内多样性**。同一个 intent 可能以多种不同的视觉模式出现，例如：

- 室外自然场景 vs 室内艺术场景
- 动作主导表达 vs 场景氛围主导表达
- 物体线索强 vs 风格线索强

如果仍然只学习一个统一 profile，就可能出现两个问题：

1. 多种子模式被平均化，判别性下降
2. comparative verification 在 hard pairs 上不够精准，因为当前图像真正匹配的只是某个子模式，而不是整体类均值

因此，这个实验的核心思想是：

> 不再给每个 intent 只学习一个 evidence profile，而是学习多个 **prototype evidence profiles**；测试时让图像先选择最匹配的 prototype，再做 candidate-local comparative verification。

---

## 2. 核心研究假设

### Hypothesis 1
单一 intent-level evidence profile 会掩盖同一 intent 内部的多模态表达，导致局部 verification 能力受限。

### Hypothesis 2
为每个 intent 建立少量 prototype evidence profiles，可以更好覆盖类内多样性，从而提高 candidate-local disambiguation 的针对性。

### Hypothesis 3
prototype memory 的收益主要应体现在：

- `Hard`
- `Macro`
- 局部 pairwise ranking accuracy
- hard ambiguity case studies

而不一定要求全局所有指标都大幅上涨。

---

## 3. 方法设计（最小改动版）

## 3.1 总体原则

这个实验必须遵循“**只改 verifier profile，不改 base proposal system**”的原则。

即：

- `baseline` 不变
- `scenario SLR-C` 不变
- comparative verification 框架不变
- gate / residual fusion 不变

唯一改变的是：

> 把每个 intent 的 single evidence profile，替换为多个 prototype-specific evidence profiles。

---

## 3.2 Step A：构造训练样本的 evidence representation

对每张训练图像 `x`，继续沿用当前 verifier 中已有的 evidence extraction pipeline，得到 evidence representation：

- `object`
- `scene`
- `style`
- `activity`

将它们拼接为一个统一 evidence vector：

$$
e(x) = [e^{obj}(x); e^{scene}(x); e^{style}(x); e^{act}(x)]
$$

### 建议
优先使用当前已有 verifier 中已经有效的设置：

- 使用现有 expert bank
- 使用当前激活 / 稀疏化设置
- 若需要二选一，优先用 **focused evidence vector**，而不是过于原始的 full dense vector

这样可以减少方法变量，先验证 prototype 思路本身。

---

## 3.3 Step B：每个 intent 内做 prototype clustering

对于每个 intent `c`，收集训练集中所有正样本的 evidence vectors：

$$
\{e(x_n) \mid y_{n,c}=1\}
$$

然后在该 intent 内单独做聚类，得到 `K` 个 prototype centers：

$$
p_{c,1}, p_{c,2}, \dots, p_{c,K}
$$

### 初始聚类方法
建议最先只用：

- `KMeans`

### 初始 prototype 数量
建议只试：

- `K = 2`
- `K = 3`

不要一开始用更大 `K`，因为：

- 容易导致统计不稳
- 容易过拟合
- 难以解释

### 数据不足时的 fallback
对样本较少的 intent，允许退回 single-profile：

- 若该类正样本数 `< n_min_for_clustering`，则不做 multi-prototype
- 直接保留单一 profile

这应作为默认的安全机制，而不是失败情况。

---

## 3.4 Step C：为每个 prototype 学 prototype-specific relation profile

不能只把 cluster center 直接拿来做最终 verification profile。更合理的做法是：

- 先用 clustering 把该 intent 的正样本分到不同 prototype clusters
- 再在每个 cluster 内单独学习 discriminative relation profile

即，对于 prototype `(c, k)`，从其对应 cluster 中估计 relation：

- `hard_negative_diff`
- （可选）`support_contradiction`

并保留 sparse top-N evidence elements。

最终，每个 intent 不再只有一个 profile，而是有一个 prototype profile set：

$$
\mathcal{P}_c = \{\mathcal{P}_{c,1}, \dots, \mathcal{P}_{c,K}\}
$$

其中每个 `\mathcal{P}_{c,k}` 都是一个 sparse discriminative profile。

### 初始建议
MVP 里先只用：

- `hard_negative_diff`

待 prototype 思路证明成立后，再补 `support_contradiction` 分支。

---

## 3.5 Step D：测试时进行 prototype selection

测试时，仍然先由 `scenario SLR-C` 产生 top-k candidates。

对于某个测试图像 `x` 和某个候选 intent `c`，先计算图像 evidence vector `e(x)` 与该 intent 各个 prototype center 的相似度：

$$
s_{proto}(x,c,k) = sim(e(x), p_{c,k})
$$

然后选择最匹配的 prototype：

$$
k^*(x,c) = \arg\max_k s_{proto}(x,c,k)
$$

后续 verification 时，对候选 `c` 使用的就不是统一 profile，而是：

$$
\mathcal{P}_{c,k^*(x,c)}
$$

### 初始相似度函数
优先试最简单的：

- cosine similarity

如果有必要，再比较：

- dot product

---

## 3.6 Step E：comparative verification 结构保持不变

prototype memory 的目的不是发明新的 comparative verifier，而是替换 verifier 所依赖的 class profile。

因此，candidate-local comparative verification 框架保持不变：

- top-k candidates 仍由 `SLR-C` 提供
- comparative pairwise scoring 仍按当前 strongest implementation 进行
- gate 保持不变
- fusion 保持不变

唯一变化是：

- 以前对 `c_i` 用统一 profile
- 现在对 `c_i` 用其 best-matched prototype profile

即：

- `c_i -> \mathcal{P}_{c_i,k_i^*}`
- `c_j -> \mathcal{P}_{c_j,k_j^*}`

再照常做 pairwise comparative verification。

---

## 4. 实验设置建议

## 4.1 固定不动的部分

为了把变量控制住，下面这些默认先固定：

- base system：`scenario SLR-C`
- top-k：沿用当前 strongest setting
- gate：沿用 v2 comparative verifier 当前设置
- fusion：沿用当前 best `add`
- evidence experts：沿用现有 strongest subset（默认先用 `all`）
- activation top-m / sparse top-N：沿用当前 strongest verifier setting

## 4.2 本轮主要变量

prototype 实验只优先比较以下变量：

1. `K = 2` vs `K = 3`
2. `single profile` vs `best prototype profile`
3. prototype source：
   - `full evidence vector`
   - `focused evidence vector`

若时间有限，优先级如下：

1. `single profile` vs `best prototype (K=2)`
2. `best prototype (K=3)`
3. source ablation

---

## 5. 必做实验

## EXP-1. 主结果表

比较：

1. `scenario SLR-C`
2. `v2 comparative + gate`
3. `v2 + prototype memory (K=2)`
4. `v2 + prototype memory (K=3)`

报告指标：

- Macro F1
- Micro F1
- Samples F1
- mAP
- Hard

### 目的
判断 prototype memory 是否在当前 strongest verifier 之上带来稳定增益。

---

## EXP-2. Prototype selection 消融

比较：

1. `single profile`
2. `best prototype`
3. （可选）`weighted average over prototypes`

### 目的
判断收益是否来自：

- prototype 数量本身
- 还是“按样本选择最匹配 prototype”这一机制

### 优先级
若时间有限，先只做：

- `single profile`
- `best prototype`

---

## EXP-3. Prototype source 消融

比较：

1. prototypes from `full evidence vector`
2. prototypes from `focused evidence vector`

### 目的
判断 prototype 聚类更适合建立在：

- 全 evidence representation
- 还是已经过筛的 focused representation

### 预期
我更看好 `focused evidence vector`，因为噪声更少，也更接近 verifier 的实际判别空间。

---

## EXP-4. Hard subset / pairwise diagnostics

除全局指标外，还应额外报告：

- low-margin subset
- top confusion pair subset
- pairwise ranking accuracy
- top-2 disambiguation accuracy

### 目的
验证 prototype memory 的收益是否真的来自：

- 更好的类内模式匹配
- 更强的局部 comparative disambiguation

---

## EXP-5. Prototype fallback 分析

统计：

- 多少个 intent 使用了 multi-prototype
- 多少个 intent 因样本不足退回 single-profile
- 各 prototype cluster 的样本量分布

### 目的
判断这个方向是否存在明显的数据规模瓶颈。

---

## 6. 建议的 case study

如果 prototype memory 有效，case study 会非常重要。

建议展示：

1. 同一个 intent 的两个 prototype
2. 每个 prototype 对应的代表性 evidence pattern
3. 某个测试图像最终匹配到了哪个 prototype
4. 为什么该 prototype 比统一 profile 更合理
5. comparative verification 因此如何更好地区分混淆候选

### 理想展示点
如果某个 intent 本身就存在明显两种视觉模式，这种案例会非常有说服力。

---

## 7. 成功信号与停止条件

## 7.1 成功信号
若这个方向有效，通常应至少出现以下几类现象中的两类：

1. `Hard` 提升
2. `Macro` 提升
3. pairwise ranking accuracy 提升
4. case study 显示 prototype selection 明显比 single profile 更合理

## 7.2 停止条件
如果出现以下情况，则不建议继续深入：

1. `K=2 / 3` 都没有比 single profile 更稳定
2. global 指标不涨，hard/pairwise 指标也不涨
3. prototype cluster 样本严重碎裂，relation profile 不稳定
4. case study 看不出 prototype selection 的明确语义意义

---

## 8. 风险与注意事项

## 风险 1：prototype 过多导致统计不稳
因此初始只试：

- `K=2`
- `K=3`

## 风险 2：小类样本不足
必须提供 fallback：

- 小类直接回退到 single profile

## 风险 3：聚类空间选得太噪
因此建议优先尝试：

- focused evidence vector

## 风险 4：收益只体现在局部，但没有体现到整体
这种情况不一定说明方向无效，但意味着它更适合作为：

- hard-case extension
- 或 discussion branch

而不是主方法。

---

## 9. 推荐的最小可行版本（MVP）

第一轮建议只做下面这个最小闭环：

1. 使用当前 verifier 的 evidence representation
2. 每个 intent 在训练正样本上做 `KMeans (K=2)`
3. 每个 cluster 单独统计 `hard_negative_diff`
4. 每个 prototype 保留 sparse top-N evidence
5. 测试时每个候选 intent 选择 cosine-sim 最匹配的 prototype
6. comparative + gate + fusion 全部保持不变

这个 MVP 已经足以回答：

> prototype evidence memory 是否比 single profile 更适合当前 candidate-local verification 任务？

---

## 10. 一句话总结

这个实验的本质可以概括为：

> **把每个 intent 的单一 evidence template 升级为多个从训练数据中自动发现的 evidence prototypes；测试时让图像先选择最匹配的 prototype，再进行 candidate-local comparative verification。**

如果成功，它最可能提升的不是简单全局平均指标，而是：

- hard ambiguity 的判别能力
- 类内多样性覆盖能力
- comparative verification 的局部匹配精度
