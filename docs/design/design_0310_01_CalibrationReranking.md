# 一、当前论文的核心问题

这篇小论文要解决的问题，不应再表述成：

> 我们想在 Intentonomy 上做一个更复杂的 intent 分类模型。

更准确的表述应该是：

> **图像意图识别并不是普通的 object/scene/action 识别，而是高层、抽象、主观的语义感知任务。**
> 在强 frozen CLIP visual baseline 已经足够强的情况下，任务剩余难点不再主要来自视觉表征不足，而来自：
>
> 1. 候选 intent 之间的高层语义歧义
> 2. 多标签决策规则与 reranked score geometry 的不匹配

因此，这篇论文的目标是：

# **在不破坏强视觉 baseline 的前提下，提升高层意图感知中的局部语义判别与最终决策质量。**

---

# 二、动机（Motivation）

## 2.1 强 baseline 已经足够强，复杂全局建模未必有效

你已经验证：

* frozen CLIP ViT-L/14 + global feature + MLP baseline 显著超过已有 SOTA
* semantic weighted loss、ICRN、SUIL、UABL 等复杂语义/结构化方法并没有继续提升，反而常常伤害 baseline

这说明：

> **Intentonomy 的瓶颈并不主要在“学更强的视觉表示”，也不一定在“全局语义结构建模”。**

---

## 2.2 全局 semantic prior 不可靠，但局部候选中的 semantic prior 有价值

你已经验证：

* text-only zero-shot classification 基本无效
* raw LLM descriptions 作为全局 classifier 不可靠
* 但在 baseline 的 top-k 候选中，semantic prior 做 local rerank 明显有效

这说明：

> **semantic prior 的问题不在“有没有用”，而在“怎么用”。**
> 它不适合作为全局分类器替代视觉模型，但适合作为局部候选判别的辅助信号。

---

## 2.3 reranking 改善了候选排序，但最终多标签决策还需要 calibration

你已经验证：

* plain SLR 明显提高 macro / hard
* 在 reranked scores 上重新做 calibrated decision rule，尤其 class-wise threshold，可以进一步显著提升 overall performance

这说明：

> **Intent perception 可以自然分解成两个阶段：**
>
> 1. **candidate proposal / local semantic disambiguation**
> 2. **calibrated multi-label decision**

这正是你当前方法的核心逻辑。

---

# 三、当前方法的详细表述（Method）

两阶段框架：

# **SLR-C: Semantic Local Reranking with Calibrated Decision Rules**

其中有两个核心模块：

1. **Heterogeneous Semantic Local Reranking**
2. **Class-wise Calibrated Decision**

---

## 3.1 问题设定

给定图像 (x)，类别集合为 (\mathcal{Y}={1,\dots,C})。

强视觉 baseline 输出每个类别的 logits：

[
z = f_{\theta}(x) \in \mathbb{R}^{C}
]

对应概率为：

[
p_c = \sigma(z_c)
]

由于任务是 multi-label intent recognition，最终预测为：

[
\hat y_c = \mathbb{I}(p_c > t_c)
]

其中 (t_c) 是决策阈值。

我们的目标不是重新学习一个新主干，而是：

* 先用 baseline 提供 plausible intent candidates
* 再用 semantic prior 在候选集内做局部重排
* 最后用 calibrated decision rule 输出最终标签集合

---

## 3.2 Stage 1: Heterogeneous Semantic Prior Construction

对每个 intent 类别 `(c)`，当前使用 4 类文本先验：

### (1) lexical prior

由**短语化**的类别描述构成，不再使用缩写类名。

### (2) canonical prior

沿用原来的 detailed intent definition，提供较完整的规范语义描述。

### (3) scenario prior

使用原始 Gemini 生成结果中的 `Text Query`，强调典型视觉场景。

### (4) discriminative prior

使用原始 Gemini 结果中的 `Core Difference`，强调类间区分信息。

当前实验表明：

* 单源 strongest 是 `scenario`
* `lexical` 与 `canonical` 各自也有效
* `discriminative` 单独可用，但不如 `scenario`
* `lexical + canonical` 的 source ensemble 是一个强且稳定的替代 prior

因此，对每个类别可以得到：

[
s_c^{lex}(x), \quad
s_c^{can}(x), \quad
s_c^{scn}(x), \quad
s_c^{dis}(x)
]

对于 heterogeneous source ensemble，当前最简单有效的形式是：

[
s_c^{lex+can}(x)=\frac{1}{2}\left(s_c^{lex}(x)+s_c^{can}(x)\right)
]

这里的 `s_c(x)` 只作为 semantic prior score，用于局部 rerank，而不是全局分类分数。

---

## 3.3 Stage 2: Semantic Local Reranking

baseline 先提供 top-k 候选集合：

[
\mathcal{T}_k(x)=\text{TopK}(z)
]

你当前最优设置里，(k=10)。

然后只在候选集合内部进行局部 reranking。当前最强版本是 plain rerank，即：

[
z'_c =
\begin{cases}
z_c + \alpha ,\tilde s_c(x), & c \in \mathcal{T}_k(x) \
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
]

其中：

* (\alpha) 是 rerank strength
* (\tilde s_c(x)) 是对 text prior 归一化后的分数
* 当前最优配置来自 `add_norm`

也就是说：

> **semantic prior 并不全局替代视觉分类器，而只在视觉模型已经提出的 plausible intent candidates 中，帮助完成更细的语义判别。**

这是整篇方法最核心的设计点。

---

## 3.4 Stage 3: Calibrated Decision Rule

由于 local reranking 改变了候选类之间的 score geometry，原始 baseline 的决策阈值不再最优。

因此，我们在 reranked logits (z') 上引入 calibrated decision rule：

[
\hat y_c = \mathbb{I}(\sigma(z'_c) > t_c)
]

其中 (t_c) 为 class-wise threshold。

你当前实验表明：

* global threshold 有用
* frequency-group threshold 有一定作用
* **class-wise threshold 最强**

因此最终方法采用：

# **class-wise calibrated decision rule**

这一步不是附属后处理，而是 SLR 框架中不可缺少的第二阶段，因为：

* reranking 负责改进 candidate ordering
* calibration 负责把排序改进转化为最终 multi-label outputs

---

# 四、当前方法的效果（Effectiveness）

这部分建议写成“发现 + 结果”两层。

## 4.1 关键发现

### 发现 1：global semantic 使用方式无效

* text-only zero-shot 基本无效
* global semantic branches 不稳甚至掉点

### 发现 2：local semantic reranking 有效

* plain SLR 明显优于 baseline
* hard subset 也能提升

### 发现 3：复杂 gate / learnable fusion 不是主要增益来源

* class-wise hand-crafted gate 无额外收益
* uncertainty gate 无明显收益
* learnable fusion 不优于 source-matched plain rerank

说明：

> **关键不在于更复杂的参数化融合，而在于 semantic prior 被限制在局部候选集内使用。**

### 发现 4：calibration 进一步释放 reranking 的收益

* `scenario SLR + class-wise threshold` 达到当前 overall 最优
* `scenario SLR + frequency-group threshold` 在 hard 上达到当前最强
* `lexical+canonical` source ensemble 提供了强而稳定的替代语义源

---

## 4.2 当前最好结果

你现在至少有两个应该在论文里明确报告的 strongest variants：

### Overall 最强

**scenario SLR + class-wise threshold**

* macro = **0.5128**
* micro = **0.5913**
* samples = **0.5847**
* hard = **0.3398**

### Hard 最强

**scenario SLR + frequency-group threshold**

* macro = **0.4941**
* hard = **0.3599**

### Ensemble 强结果

**lexical + canonical + class-wise threshold**

* macro = **0.5065**
* micro = **0.5737**
* samples = **0.5746**
* hard = **0.3238**

建议主方法先用 overall 最强版本，hard-best 作为 trade-off 补充，`lexical+canonical` 作为 heterogeneous source 的稳定替代版本。

---

# 五、当前方法的核心贡献

## Contribution 1

We establish a strong frozen CLIP baseline that substantially surpasses prior SOTA on Intentonomy, showing that high-level visual intent perception is not mainly bottlenecked by feature extraction.

## Contribution 2

We show that semantic priors are ineffective as global classifiers but become highly effective when restricted to local candidate reranking, revealing a more appropriate usage pattern for semantic information in intent perception.

## Contribution 3

We propose a two-stage framework, SLR-C, which combines heterogeneous semantic local reranking with class-wise calibrated decision rules, significantly improving both overall macro F1 and hard-case performance.

---

# 六、实验补充

## 6.1 主表（已补）

### 表 1：总体主表

| Method | macro | micro | samples | mean F1 | hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior SOTA | 43.05 | 54.77 | 56.75 | 51.52 | - |
| baseline | 45.98 | 56.54 | 56.40 | 52.97 | 26.86 |
| baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 51.07 | 29.71 |
| baseline + SLR-v0 (scenario) | 47.14 | 56.52 | 55.11 | 52.92 | 28.32 |
| baseline + SLR-v0 (scenario) + class-wise threshold | **51.28** | **59.13** | **58.47** | **56.29** | 33.98 |
| baseline + lexical+canonical SLR | 49.61 | 57.20 | 57.29 | 54.70 | 33.11 |
| baseline + lexical+canonical SLR + class-wise threshold | 50.65 | 57.37 | 57.46 | 55.16 | 32.38 |

这张表已经能够支撑全文主结果：

- overall 最强：`SLR-v0 (scenario) + class-wise threshold`
- heterogeneous source 的 strongest alternative：`lexical+canonical SLR + class-wise threshold`

### 表 2：semantic prior usage mode 对比

这张表的目的不是再堆方法，而是把全文核心发现钉死：

| Usage mode | test macro | test hard | 结论 |
| --- | ---: | ---: | --- |
| text-only global classification | 12.18 | 6.96 | 全局 semantic classifier 基本无效 |
| local rerank (`SLR-v0`, scenario) | 47.14 | 28.32 | local semantic 有效 |
| local rerank + class-wise threshold | **51.28** | 33.98 | calibration 显著释放 rerank 收益 |
| local rerank + learnable fusion | 49.63 | 32.49 | learnable fusion 仍不如 calibrated decision rule |

这里最重要的结论是：

> global semantic 无效，local semantic 有效，而“更复杂的 learnable fusion”并不是当前增益来源。

### 表 3：prompt/source 与 calibration 消融

| Source | global threshold | class-wise threshold |
| --- | ---: | ---: |
| lexical | 49.42 / 32.34 | 49.63 / 32.49 |
| canonical | 48.91 / 31.98 | 49.22 / 31.88 |
| lexical + canonical | **49.61 / 33.11** | **50.65 / 32.38** |

上表记法为：

- `macro / hard`

这张表支持两个判断：

1. `lexical + canonical` 的 heterogeneous semantic prior 确实互补
2. calibration 在新 source 体系里仍然有作用，但不再对 ensemble 一定单调增益

---

## 6.2 支撑实验（已补）

### 实验 A：threshold calibration 在不同 score space 上的收益

这一步已经做完，结论如下：

- baseline + class-wise threshold：
  - `macro: 45.98 -> 48.23`
  - `hard: 26.86 -> 29.71`
  - 但 `micro / samples` 明显下降
- scenario SLR-v0 + class-wise threshold：
  - `macro: 47.14 -> 51.28`
  - `micro: 56.52 -> 59.13`
  - `samples: 55.11 -> 58.47`
  - `hard: 28.32 -> 33.98`

这说明：

> calibration 不是 baseline 上的通用小技巧，而是对 reranked score geometry 尤其重要。

### 实验 B：group-wise calibration

这一步已经做完，优先试了两种分组：

1. semantic prior 增益分组
2. 类频次分组

结果：

- semantic prior gain grouping：现在会得到两个不同阈值，但收益仍然有限，不如 class-wise threshold
- frequency grouping：有效，但明显弱于 class-wise threshold
  - `macro = 49.41`
  - `hard = 35.99`

因此：

- 如果只追求 strongest result，选 `class-wise threshold`
- 如果需要更“方法化”的 compact rule，`frequency-based group-wise threshold` 可以保留

### 实验 C：source ensemble

这一步已经完成，而且结论很明确：

- `lexical + canonical` 仍明显优于单独 `canonical`
- `lexical + canonical` 的 global threshold 版本达到：
  - `macro = 49.61`
  - `hard = 33.11`
- `lexical + canonical + class-wise threshold` 达到：
  - `macro = 50.65`
  - `hard = 32.38`

它在 `hard` 上已经非常强，且和 overall 最强方案构成了很好的 trade-off 补充。

### 仍未做但值得补的支撑实验

目前状态已经分化成两部分：

1. `candidate recall / oracle analysis`
   - 已完成
2. 多 seed 稳定性
   - 脚本已补齐
   - 结果等待后续补跑

---

### 实验 D：candidate recall / oracle analysis（已完成）

输出目录：

- `logs/analysis/full_candidate_recall_oracle_20260310`

核心结果：

- baseline top-1 label recall: `36.78%`
- baseline top-3 label recall: `61.86%`
- baseline top-5 label recall: `72.12%`
- baseline top-10 label recall: `83.91%`

对应 sample-level coverage：

- top-10 `sample_any_recall = 95.07%`
- top-10 `sample_all_recall = 75.33%`

这说明：

> baseline 的 top-10 candidate proposal 已经非常强，绝大多数样本至少有一个 GT 在 top-10 内，且约四分之三样本的全部 GT 都已被候选集覆盖。

更关键的是 oracle 上界：

- oracle@top10:
  - `macro = 87.42`
  - `micro = 91.25`
  - `samples = 89.67`

这说明：

> 当前真正的瓶颈远不是 candidate proposal capacity，而是 top-k 候选内部的局部语义判别与最终决策规则。

这组结果非常直接地支持了两阶段叙事：

- baseline 负责 candidate proposal
- SLR 负责 candidate disambiguation
- calibrated decision rule 负责把候选内的排序收益转成真正的 multi-label 输出

---

### 实验 E：多 seed 稳定性（脚本已补齐，结果待后续更新）

已补脚本：

- `scripts/train_intentonomy_layer_cls_patch_mean_seed.sh`
- `scripts/run_multiseed_slr_calibration.sh`
- `scripts/aggregate_multirun_stability.py`

用途分别是：

1. 固定 seed 跑 baseline 训练
2. 对每个 seed 跑 calibrated decision rule 分析
3. 聚合 `baseline / scenario SLR / scenario SLR + class-wise threshold / lexical+canonical + class-wise threshold` 的 `mean ± std`

因此，这一项现在不再缺执行入口，只缺后续把 seed 跑完并把汇总结果写回文档。

---

# 七、分析补充

## 7.1 关于 calibrated decision rule 的核心分析

当前最重要的机制性结论是：

1. `SLR-v0` 先把候选间的局部语义关系排得更合理。
2. `class-wise threshold` 再把新的 score geometry 转成更合适的 multi-label output。

也就是说：

- rerank 负责 candidate disambiguation
- calibration 负责 decision alignment

这两步缺一不可。

## 7.2 关于 heterogeneous semantic prior 的分析

现有结果已经足够支持：

> heterogeneous semantic priors are complementary across intent categories.

证据是：

- `lexical` 与 `canonical` 单独都有效
- `lexical + canonical` ensemble 能稳定超过单独 `canonical`
- `scenario` 是当前最强的 single-source semantic prior

这说明当前 semantic prior 的有效形态不是单一的，而更像三种互补信息：

- `lexical`：紧凑短语
- `canonical`：规范语义定义
- `scenario`：典型视觉场景

## 7.3 关于复杂融合为何不是关键

learnable fusion 已经按新 source 体系重跑，结果仍然很明确：

- class-wise affine 依然好于 shared MLP
- 但 strongest learnable fusion 也只到 `macro = 49.63 / hard = 32.49`
- 它仍低于 `scenario + class-wise threshold`

这说明：

> 当前问题的主要收益不来自更复杂的参数化融合，而来自 semantic prior 的局部使用方式和 calibrated decision rule。

所以主线不应写成：

- “我们提出一个更强 fusion network”

而应写成：

- “我们提出一种两阶段 intent decision framework：local reranking + calibrated decision”

## 7.4 Abstract vs concrete intent analysis（已补）

输出目录：

- `logs/analysis/full_abstract_concrete_analysis_20260310`

我们将 28 个 intent 粗分为两组：

- `abstract`
  - 更偏高层状态、关系、价值或主观语义
- `concrete`
  - 更偏清晰的视觉原型、动作或场景线索

在 baseline 上，两组均值已经明显不同：

- abstract mean F1 = `41.93`
- concrete mean F1 = `53.26`

也就是说，abstract intents 本身就是当前任务里更难的一组。

### 关键结果

| Method | abstract mean F1 | concrete mean F1 | 相对 baseline 的 abstract 增益 | 相对 baseline 的 concrete 增益 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 41.93 | 53.26 | - | - |
| baseline + class-wise threshold | 44.61 | 54.75 | +2.67 | +1.49 |
| scenario SLR | 44.44 | 52.01 | +2.51 | -1.25 |
| scenario SLR + class-wise threshold | **48.69** | 55.93 | **+6.76** | +2.67 |
| scenario SLR + frequency-group threshold | 46.02 | 55.51 | +4.09 | +2.25 |
| lexical+canonical SLR | 45.85 | 56.38 | +3.92 | +3.12 |
| lexical+canonical SLR + class-wise threshold | 47.01 | **57.20** | +5.08 | **+3.94** |
| best learnable fusion | 46.84 | 54.65 | +4.90 | +1.39 |

### 结果解读

这张表支持一个非常重要的论文结论：

> 当前方法的主要价值，确实更偏向提升高层、抽象 intent perception，而不是只提升本来就容易的具体视觉类。

证据有三层：

1. abstract 组在 baseline 上明显更难
   - `41.93 << 53.26`
2. `scenario SLR` 的第一阶段收益，主要集中在 abstract intents
   - abstract `+2.51`
   - concrete `-1.25`
3. `scenario SLR + class-wise threshold` 对 abstract 的增益远大于 concrete
   - abstract `+6.76`
   - concrete `+2.67`

也就是说：

- local semantic reranking 首先在抽象类上补充了高层语义判别
- calibrated decision rule 再把这种判别收益转成最终输出

### 类别级信号

以 `scenario SLR + class-wise threshold` 为例，增益最大的 abstract 类包括：

- `WorkILike`
- `PassionAbSmthing`
- `ManagableMakePlan`
- `SocialLifeFriendship`
- `HardWorking`

这些类都有一个共同特点：

- 不是简单 object / scene / action 标签
- 更依赖高层关系、状态或价值语义

而 `lexical+canonical + class-wise threshold` 则更均衡：

- 对 abstract 仍然有明显收益
- 对 concrete 的提升更稳

这正好对应两条 strongest variant 的分工：

- `scenario + class-wise threshold`：更像 abstract intent enhancer
- `lexical+canonical + class-wise threshold`：更像 balanced overall performer

### 分析结论

因此，现在可以把论文里的表述写得更强一些：

> 该方法提升的不是普通视觉类别识别，而是更偏 high-level、abstract intent perception。

这比单纯说 “macro/hard 涨了” 更有说服力，也更贴近论文动机。

---

## 7.5 仍可选补充的分析

当前论文主线已经具备：

- 主表
- source / calibration 支撑实验
- candidate recall / oracle analysis
- abstract vs concrete analysis

如果还要继续补，优先级较高的只剩：

1. 多 seed 稳定性结果回填
2. hard-case taxonomy

其中第一项更重要，因为它直接关系到结果可信度。

---

# 八、当前论文最自然的主线

建议最终写成：

# **图像意图感知需要两阶段处理：**

## 1. visual candidate proposal

## 2. local semantic disambiguation + calibrated decision

对应到方法就是：

* 强视觉 baseline：负责产生 plausible intent candidates
* SLR：负责在候选集内利用 heterogeneous semantic priors 做局部重排
* class-wise threshold：负责把 reranked scores 转为合适的 multi-label 输出

这条线很清楚，也很适合你后面服务博士大论文里的“intent-aware perception”。

---

# 十、顺序

按优先级：

### 第一优先

回填多 seed 稳定性：

* baseline
* scenario SLR
* scenario SLR + class-wise threshold
* lexical+canonical + class-wise threshold

### 第二优先

补 hard-case taxonomy

### 第三优先

如果篇幅允许，再补更细的类别叙事分析：

* abstract vs concrete 内部再细分
* 典型类别对的 semantic confusion 解释
