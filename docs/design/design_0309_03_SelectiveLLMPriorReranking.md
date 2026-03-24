# 2026-03-09 Selective LLM-Prior Reranking

## 1. 状态

这份文档定义一个新的主线方法：

> **Selective LLM-Prior Reranking**

它不是新的全局分类器，而是在当前 strongest baseline 的 `top-k` 候选内部，选择性地引入 LLM text prior 做局部语义校正。

这条线的出发点不是拍脑袋，而是已经有明确实验依据。相关分析见：

- `docs/design/design_0309_Analysis.md`
- `logs/analysis/full_text_prior_boundary_20260309/summary.json`

---

## 2. 经验事实

固定 baseline 为：

- `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

在统一分析协议下：

- baseline test: `macro = 0.4598`, `hard = 0.2686`
- best rerank: `topk = 10`, `mode = add_norm`, `alpha = 0.3`
- best rerank test: `macro = 0.4843`, `hard = 0.2879`
- relative gain: `macro +0.0245`, `hard +0.0193`

同时还有三个关键观察：

1. `text-only zero-shot` 基本无效。
2. `top-k rerank` 明显有效。
3. `retrieval prior` 只有弱增益，明显不如 rerank。

这说明：

> 当前 hard gap 的主问题更像是“局部边界与语义消歧”，而不是“缺少邻居 reference”。

更细一点地说：

- baseline 往往已经把正确类放进候选集
- 错误主要发生在候选内部的局部排序
- LLM prior 的价值更像 `local semantic calibrator`
- 但它不是对所有类、所有样本都稳定有益

---

## 3. 前置阶段：先把 plain rerank 做扎实

在进入 Selective 方法之前，还有两组前置工作不能跳过。

原因很简单：

- 如果 plain rerank 的增益本身不稳定，Selective 机制就没有坚实地基
- 如果 plain rerank 的关键变量没有系统消融，后面的 gating 方法也会显得像堆 trick

因此，这两项应该保留为正式 TODO，并且优先级高于 `SLR-v1/v2/v4` 的实现。

### 3.1 TODO 1：确认 rerank 结果稳定性

当前最优 plain rerank 的单次结果很强：

- baseline: `macro = 0.4598`, `hard = 0.2686`
- plain rerank: `macro = 0.4843`, `hard = 0.2879`

但论文里不能只报单次最优，需要确认这不是偶然。

建议协议：

1. 至少 `3` 个 seed 跑 baseline
2. 至少 `3` 个 seed 跑 baseline + plain rerank
3. 保持同一 val 选配置规则
4. 汇报 `mean ± std`

重点看：

- `macro` 是否稳定上涨
- `hard` 是否稳定上涨
- 最优模式是否仍然偏向 `add_norm`
- `topk=10, alpha=0.3` 是否仍在稳定区间附近

如果这一项成立，plain rerank 就不再只是一个单次巧合，而是可作为正式方法基线。

### 3.2 TODO 2：做更完整的 plain rerank 消融

在进入 Selective 方法前，plain rerank 需要先有一份论文式 ablation。

核心不是证明“我调到了最好”，而是证明：

- local reranking 明显优于 text-only classification
- LLM prior 的价值来自局部判别，而不是全局替代
- `add_norm` 这种受控融合比粗暴加权更稳

建议的消融轴：

1. `text source`
   - `short`
   - `detailed`
   - `llm`
2. `top-k`
   - `3 / 5 / 10 / 20`
3. `fusion mode`
   - `add`
   - `add_norm`
   - 可再补一个更保守的 `gated add`
4. `alpha`
   - `0.1 / 0.2 / 0.3 / 0.5`

重点看：

- 最优配置是否稳定集中在 `add_norm`
- `topk=10` 是否真的是最合理候选范围
- `llm` 是否在 rerank 中显著优于 `short / detailed`
- 某些配置是否虽然 overall 涨，但会系统性伤害 `hard` 或高风险类

如果这项做扎实，Selective 方法的贡献就会更清楚：

- 不是“随便挑一个 rerank 配置再堆 gate”
- 而是“先证明 plain rerank 是有效机制，再证明 selective integration 进一步提升其稳定性和可控性”

### 3.3 前置阶段与 Selective 方法的关系

这一点需要写清楚：

- `TODO 1/2` 不是可有可无的补充
- 它们是 Selective 方法的前置验证阶段

可以把整体路线写成：

1. `Phase 1`: 确认 plain rerank 稳定且可解释
2. `Phase 2`: 引入 class-wise / uncertainty-aware selection
3. `Phase 3`: 进一步抑制过泛化并提升方法完整性

---

## 4. 为什么不能停留在普通 rerank

直接做：

```text
z'_c = z_c + alpha * s_text(c)
```

虽然已经有增益，但论文味还不够强，原因有两个：

1. 看起来更像工程 trick，而不是任务特定机制。
2. 现有 hard-case 已经说明它会系统性伤害一部分类。

具体表现为：

- 容易过度推高 `Playful`
- 容易误伤 `Attractive`
- 容易误伤 `FineDesignLearnArt-Art`
- 容易误伤 `EnjoyLife`
- 一部分 degraded case 会直接把预测压成空，或引入新的 false positive

因此，下一步不是否定 rerank，而是把它变成：

> 只有在“该类值得信 text prior”且“该样本当前处于边界附近”时，才让 text prior 发挥作用。

这就是 **Selective** 的含义。

---

## 5. 核心假设

Selective LLM-Prior Reranking 依赖三个假设。

### 5.1 类别选择性

不是所有类都该同等信任 LLM prior。

- 抽象类、语义氛围类、边界模糊类更容易受益
- 视觉原型稳定、细粒度边界强的类更容易被误伤

### 5.2 样本选择性

不是所有样本都该同等信任 LLM prior。

- baseline 很确定时，text prior 不该强行介入
- baseline 犹豫时，text prior 更可能有价值

### 5.3 候选局部性

text prior 不该在全类别空间里大范围改写分布，而应主要在 baseline 已经筛出的候选内部起作用。

这也是为什么：

- `text-only classification` 不行
- `top-k rerank` 却有效

---

## 6. 方法定义

### 6.1 基础记号

设 baseline 输出 logits：

```text
z in R^C
```

对应概率：

```text
p_c = sigmoid(z_c)
```

LLM text prior 分数记为：

```text
s_text(x, c)
```

其中 `s_text` 来自图像特征与该类别多条 LLM descriptions 的 CLIP similarity 聚合。

只对 baseline 的 `top-k` 候选集合 `T_k(x)` 做 rerank。

### 6.2 当前已验证有效的基础版本

已经验证最有效的基础版本是：

```text
z'_c = z_c + alpha * s_hat_text(x, c),   if c in T_k(x)
z'_c = z_c,                              otherwise
```

其中：

- `s_hat_text` 是按样本做过标准化的 `add_norm` 文本分数
- 当前最佳超参是：
  - `topk = 10`
  - `alpha = 0.3`

这个版本可记为：

> **SLR-v0: Plain Local Reranking**

---

## 7. Selective 机制

完整方法由 3 个控制模块组成：

1. `class-wise gate`
2. `uncertainty-aware local gate`
3. `anti-overgeneralization constraint`

最终目标是：

> 保留 rerank 的 hard-class 收益，同时压住对 `Attractive / EnjoyLife / Art / Communicate` 等类的误伤。

### 7.1 Class-Wise Gate

对每个类别定义一个静态 gate：

```text
g_c in [0, 1]
```

表示“类别 `c` 值不值得引入 LLM prior”。

最简单的做法直接来自 validation 上的 per-class rerank gain：

```text
Delta_c = F1_rerank(c) - F1_base(c)
```

然后定义两种版本。

#### 版本 A：二值 gate

```text
g_c = 1, if Delta_c > 0
g_c = 0, otherwise
```

这是最小实现，优先级最高。

#### 版本 B：连续 gate

```text
g_c = sigmoid(gamma * Delta_c)
```

或：

```text
g_c = clip((Delta_c - Delta_min) / (Delta_max - Delta_min), 0, 1)
```

这个版本更平滑，但仍然是静态 gate，不引入额外学习网络。

### 7.2 Uncertainty-Aware Local Gate

对每个样本-类别对定义一个局部 gate：

```text
q(x, c) in [0, 1]
```

表示“当前样本在该类别上是否处于边界附近，是否值得让 text prior 介入”。

最简单的定义直接使用 baseline probability 的边界不确定性：

```text
u(x, c) = 1 - |2 * p_c - 1|
```

它有两个性质：

- `p_c` 接近 `0.5` 时，`u` 大
- `p_c` 接近 `0` 或 `1` 时，`u` 小

于是最简单的局部 gate 可以取：

```text
q(x, c) = u(x, c)
```

#### Rank-aware 版本

若要更贴合 `top-k rerank` 设定，还可以乘一个 rank decay：

```text
d(x, c) = exp(-tau * (rank_k(x, c) - 1))
q(x, c) = u(x, c) * d(x, c)
```

它的直觉是：

- 候选集末端、且接近决策边界的类，更值得被 rerank 修正
- 候选顶部、且很自信的类，不应被 text prior 强行拉动

#### Hard-mask 版本

如果想先做最稳的工程版本，也可以只保留开关：

```text
q(x, c) = 1, if u(x, c) > delta
q(x, c) = 0, otherwise
```

推荐先试：

- `delta in {0.2, 0.3, 0.4}`

### 7.3 Anti-Overgeneralization Constraint

已有 hard-case 说明，现有 rerank 的典型问题是：

- 压成空
- 引入更泛的情绪类 false positive
- 尤其容易把图像往 `Playful / EasyLife / Happy` 方向推

因此，需要约束 LLM prior 的作用方式。

最推荐的最小约束是：

> **non-suppressive rerank**

即不允许 text prior 直接作为负向抑制项去压低 baseline logits，只允许它做有限增强。

做法是把文本分数截断为非负：

```text
s_pos(x, c) = max(0, s_hat_text(x, c))
```

这会减少：

- 候选被整体压低
- 空预测
- 候选内语义过度排斥

---

## 8. 最终公式

完整版本定义为：

```text
z'_c = z_c + alpha * g_c * q(x, c) * s_pos(x, c),   if c in T_k(x)
z'_c = z_c,                                         otherwise
```

其中：

- `g_c`：类别静态 gate
- `q(x, c)`：样本-类别局部 gate
- `s_pos(x, c)`：经过归一化且截断为非负的 LLM prior

这个方法可记为：

> **SLR-v4: Selective LLM-Prior Reranking**

它和普通 rerank 的差别是：

- 不是对所有类一刀切地加 text prior
- 不是对所有样本一刀切地加 text prior
- 不是允许 text prior 无约束地上推或下压 logits

---

## 9. 版本划分

为了实验清晰，建议按下面的版本序列推进。

### SLR-v0

Plain local rerank:

```text
z'_c = z_c + alpha * s_hat_text(x, c)
```

仅在 `top-k` 内使用，当前最佳配置为 `topk=10, add_norm, alpha=0.3`。

#### SLR-v0 详细定义

`SLR-v0` 不是新的分类器，而是在固定 baseline 上做一次纯推理期的局部语义重排。

它的实际流程如下：

1. 固定 baseline
   - checkpoint 固定为：
     - `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
   - baseline 模型输出 logits：
     - `z in R^C`

2. 构造 LLM text prior
   - 文本源固定为 `llm`
   - 文件：
     - `Intentonomy/data/intent_description_gemini.json`
   - 对每个类别，聚合以下文本：
     - `intent`
     - `Core Difference`
     - `Visual Elements`
     - `Text Query`
   - 然后用 frozen CLIP text encoder 编码，并对同一类别的多条文本 embedding 做平均

3. 计算 image-text similarity
   - 对每张图像，用 frozen CLIP image encoder 提取 image embedding
   - 与各类别 text embedding 做 cosine similarity
   - 再乘 CLIP 自带 `logit_scale`
   - 得到：

```text
s_text(x, c)
```

这里的 `s_text` 本质上就是 text prior 的原始 logit 分数。

4. 做样本内标准化
   - `SLR-v0` 的最优版本不是直接加 raw text logit，而是先在每个样本内部按类别维做 z-score：

```text
s_hat_text(x, c) = norm_c(s_text(x, c))
```

也就是当前脚本里的 `add_norm`。

这样做的目的有两个：

- 保留“哪几个类更像”的相对排序信息
- 避免 raw text logit 的绝对尺度直接冲击 baseline logits

5. 只对 baseline top-k 候选生效
   - 先按 baseline logits 取：

```text
T_k(x) = TopK(z),   k = 10
```

   - 然后只在这 10 个候选类内部加 text prior：

```text
z'_c = z_c + alpha * s_hat_text(x, c),   if c in T_k(x)
z'_c = z_c,                              otherwise
```

其中：

- `alpha = 0.3`

候选集外类别完全不动。

这一点很关键，因为 `SLR-v0` 的成功恰恰建立在：

- baseline 已经能把正确类压进候选集
- text prior 只需要做局部排序修正，而不是全局重写类别分布

6. 概率化与阈值评估
   - 将 `z'` 过 `sigmoid` 得到最终分数
   - 在 `val` 上搜索一个全局最佳阈值
   - `test` 直接复用该 `val` 阈值

因此，`SLR-v0` 的完整口径可以写成：

> frozen baseline logits 提供候选集，frozen CLIP LLM text prior 提供局部语义重排信号，`add_norm` 控制 text prior 尺度，只在 `top-10` 内用 `alpha=0.3` 做加性修正。

#### 为什么 SLR-v0 有效

从现有结果看，`SLR-v0` 有效的根本原因不是“LLM 文本本身足够强”，而是：

1. baseline 已经把问题缩小成局部候选排序问题
2. LLM prior 对抽象类提供了额外的高层语义区分
3. `add_norm` 避免 text prior 直接粗暴覆盖 baseline 的分布

换句话说：

- `text-only classification` 失败
- 但 `top-k + add_norm` 成功

这正是 `SLR-v0` 成为后续 `SLR-v1~v4` 出发点的原因。

### SLR-v1

Class-wise gate only:

```text
z'_c = z_c + alpha * g_c * s_hat_text(x, c)
```

### SLR-v2

Uncertainty gate only:

```text
z'_c = z_c + alpha * q(x, c) * s_hat_text(x, c)
```

### SLR-v3

Class-wise + uncertainty gate:

```text
z'_c = z_c + alpha * g_c * q(x, c) * s_hat_text(x, c)
```

### SLR-v4

Class-wise + uncertainty gate + positive-only prior:

```text
z'_c = z_c + alpha * g_c * q(x, c) * s_pos(x, c)
```

这是当前最推荐的主方法候选。

---

## 10. 最小实现建议

### 10.1 不增加新训练

第一阶段不需要改训练，不需要引入新网络。

实现应当完全基于：

- baseline logits
- 现有 LLM text prior
- validation 上的统计量

这有两个好处：

1. 可以快速判断 selective 机制本身是否成立。
2. 即使最后要写成训练式方法，也能先验证推理侧机制。

### 10.2 Class-wise Gate 的构造

直接从现有分析产物取数：

- `logs/analysis/full_text_prior_boundary_20260309/summary.json`

使用 `best rerank` 相对 baseline 的 per-class gain 生成 `g_c`。

优先级：

1. 二值 gate
2. 连续 gate

### 10.3 Uncertainty Gate 的构造

直接用 baseline logits 算：

```python
p = torch.sigmoid(z)
u = 1.0 - torch.abs(2.0 * p - 1.0)
```

如果要做 hard-mask：

```python
q = (u > delta).float()
```

如果要做 soft gate：

```python
q = u
```

### 10.4 伪代码

```python
# z: baseline logits, [B, C]
# s_text: normalized LLM prior, [B, C]
# g: class-wise gate, [C]
# topk_idx: baseline top-k indices, [B, K]

p = torch.sigmoid(z)
u = 1.0 - torch.abs(2.0 * p - 1.0)

s_pos = torch.clamp(s_text, min=0.0)
delta = alpha * g[None, :] * u * s_pos

mask = torch.zeros_like(z, dtype=torch.bool)
mask.scatter_(1, topk_idx, True)

z_new = torch.where(mask, z + delta, z)
```

如果只做 `class-wise gate only`，把 `u` 去掉即可。

---

## 11. 实验设计

### 11.1 实验顺序

实验顺序应当明确分成两阶段：

1. Plain rerank validation
   - 稳定性验证
   - 完整消融
2. Selective rerank validation
   - `SLR-v1`
   - `SLR-v2`
   - `SLR-v4`

不要一开始就跳到 Selective 版本，否则很难判断增益来自哪里。

### 11.2 主比较组

固定 baseline 与 text source，不要一开始同时改太多变量。

第一组实验：

1. baseline
2. baseline + SLR-v0
3. baseline + SLR-v1
4. baseline + SLR-v2
5. baseline + SLR-v4

### 11.3 固定项

优先沿用当前最优 plain rerank 配置：

- `text source = llm`
- `topk = 10`
- `fusion mode = add_norm`
- `alpha = 0.3`

在验证 selective 机制前，不建议同时重新搜索 `topk / mode / alpha`。

### 11.4 指标

主指标：

- `macro F1`
- `hard`

次指标：

- `micro F1`
- `samples F1`
- `mAP`
- `easy / medium`

### 11.5 诊断项

除了均值指标，还必须继续看：

- 被 `Playful` 抢走的样本数量是否下降
- `Attractive / FineDesignLearnArt-Art / EnjoyLife` 的 per-class F1 是否恢复
- 空预测样本数量是否下降
- improved/degraded hard cases 的结构是否更健康

---

## 12. 当前实验结果（2026-03-09）

Selective 版本已经在统一分析脚本中实现，并完成了一轮全量评估。

输出目录：

- `logs/analysis/full_text_prior_boundary_selective_20260309`

核心文件：

- `summary.json`
- `selective_rerank_leaderboard.csv`

### 12.1 基准：SLR-v0（plain rerank）

固定基准为当前最优 plain rerank：

- `topk = 10`
- `mode = add_norm`
- `alpha = 0.3`

对应 test：

- `macro = 0.4843`
- `micro = 0.5773`
- `samples = 0.5715`
- `mAP = 53.78`
- `hard = 0.2879`

### 12.2 各版本最佳结果

| Variant | Best config | test macro | test hard | 相对 SLR-v0 |
| --- | --- | ---: | ---: | --- |
| `SLR-v1` | `continuous_g12` | 0.4782 | 0.2830 | `macro -0.0061`, `hard -0.0049` |
| `SLR-v2` | `binary_d0.2` | 0.4843 | 0.2879 | 基本等同 `SLR-v0` |
| `SLR-v3` | `continuous_g12 + binary_d0.2` | 0.4782 | 0.2830 | 与 `SLR-v1` 基本相同 |
| `SLR-v4` | `continuous_g12 + binary_d0.2 + positive_only` | 0.4756 | 0.2814 | 略低于 `SLR-v1` 与 `SLR-v0` |

直接结论：

- 目前真正起作用的是 `class-wise gate`
- `uncertainty gate` 在当前定义下几乎没有带来额外收益
- `positive-only` 约束暂时没有带来净收益，反而略有下降

### 12.3 如果按 hard 最优来选

如果不是按 `val_macro`，而是按 `val_hard` 选当前最优 selective 版本，则最佳是：

- `SLR-v1`
- `continuous_g8`

对应 test：

- `macro = 0.4831`
- `hard = 0.3017`

相对 `SLR-v0`：

- `macro -0.0012`
- `hard +0.0139`

这个结果很关键，因为它说明：

- `class-wise gate` 确实有能力进一步拉高 `hard`
- 代价只是 very small 的 `macro` 回落

所以从研究价值看，`SLR-v1` 不是失败，而是：

> 已经显示出“更偏 hard 优化”的方向性，但还没在 overall 指标上同时压过 plain rerank。

### 12.4 当前版本最重要的实验观察

1. `SLR-v1` 是 4 个 selective 版本里唯一真正有稳定信号的版本。
2. `SLR-v2` 几乎完全复现了 `SLR-v0`，说明当前 uncertainty gate 设计太弱，或者阈值太宽。
3. `SLR-v3` 与 `SLR-v1` 几乎重合，说明加入 uncertainty gate 后没有提供新信息。
4. `SLR-v4` 略降，说明 `positive-only` 约束在当前版本下抑制过头了。

### 12.5 对错误模式的影响

和 baseline 相比，当前 best-overall selective 版本：

- 空预测数量从 `120` 降到 `115`
- 仍然最容易新增 `Playful` false positive
- 也仍然会伤到 `EnjoyLife / Attractive / FineDesignLearnArt-Art`

这说明：

- `class-wise gate` 已经有“筛掉一部分不该信 prior 的类”的效果
- 但还没有真正解决过泛化偏置
- 当前版本距离“同时涨 overall 且显著压住 `Playful` 偏置”还有一步

---

## 13. 成功判据

Selective 方法要成立，至少应满足下面 3 条中的 2 条：

1. 相比 plain rerank，`hard` 不降，且 `macro` 持平或继续上涨。
2. `Attractive / FineDesignLearnArt-Art / EnjoyLife / Communicate` 中至少若干负收益类被明显救回。
3. `Playful` 的新增 false positive 数量明显下降。

最理想的结果是：

- 维持或超过 `SLR-v0` 的 `macro`
- `hard` 再涨一点
- degraded cases 显著减少

---

## 14. 风险与应对

### 风险 1：Class-wise gate 过拟合 validation

如果 `g_c` 完全按 validation gain 构造，可能会有过拟合风险。

应对：

- 先把它当作验证机制的分析版方法
- 如果有效，再考虑更稳的 meta-gate 或多 seed 统计

### 风险 2：Uncertainty gate 太弱，效果接近 plain rerank

这说明当前主要矛盾是“哪些类该信 prior”，而不是“哪些样本该信 prior”。

这时应优先保留 class-wise gate。

### 风险 3：Positive-only 约束削弱了 hard 收益

这说明一部分 hard 样本确实需要负向抑制，而不是只靠增强。

这时可以保留 selective 结构，但把 `s_pos` 改回 signed 版本，只在高风险类上做非负约束。

---

## 15. 论文叙事位置

如果这条线成功，它的叙事会比普通 rerank 更完整：

1. Intentonomy 的错误主要来自抽象语义边界模糊。
2. LLM prior 具有局部语义校正价值，但并非全局可靠。
3. 因此，需要 `selective` 地整合语义先验，而不是一刀切。

对应的方法贡献可以写成：

- 我们提出一种 `Selective LLM-Prior Reranking` 机制
- 它只在 baseline 候选内部、且只在合适类别与不确定样本上引入 LLM prior
- 它显式抑制 text prior 的过泛化副作用

这个叙事比“再做一个 text rerank trick”强很多。

---

## 16. 当前结论

基于现有分析，Selective LLM-Prior Reranking 是当前最值得推进的下一步，原因是：

- 它直接建立在已验证有效的 plain rerank 之上
- 它直接利用了 hard-case 分析得到的类别选择性证据
- 它比 retrieval 或重新做全局分类器更贴近当前误差结构

但基于当前第一轮结果，优先级需要更精确地调整为：

1. 优先继续打磨 `SLR-v1`
   - 尤其是 `continuous gate`
   - 当前它是唯一显示真实增益趋势的 selective 版本
2. 暂缓把 `SLR-v2` 当作主线
   - 当前 uncertainty gate 几乎没有提供额外价值
3. 暂缓 `SLR-v4`
   - 当前 positive-only 约束略伤指标，不应直接当默认方案

因此，现阶段最合理的结论不是“4 个版本一起推进”，而是：

> 先把 `SLR-v1` 做强，再决定是否需要 uncertainty / positive-only 作为第二阶段补充。
