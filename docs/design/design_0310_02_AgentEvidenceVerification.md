# 技术报告：Multi-Expert Evidence Verification for Visual Intent Recognition

## 0. 文档信息

- 日期：`2026-03-10`
- 状态：内部技术报告
- 任务：Intentonomy 多标签视觉意图识别
- 研究主线：在当前 strongest `SLR-C` 基础上引入显式证据核验

---

## 1. 摘要

本报告总结了 `Multi-Expert Evidence Verification with Calibration (MEV-C)` 的设计、实现与实验结果。研究出发点是：在强 frozen CLIP baseline 已经足够强的前提下，Intentonomy 的剩余难点不再主要来自视觉特征抽取，而来自候选意图之间的局部语义消歧与最终多标签决策。已有 `SLR-C` 已经证明 `scenario prior + local rerank + class-wise calibration` 是有效路线，但该方法仍然依赖类级文本先验，缺少对图像中具体支持证据的显式核验。

为此，本工作将意图识别重构为四阶段流程：`hypothesis generation -> evidence extraction -> hypothesis verification -> calibrated decision`。系统首先使用 `scenario SLR` 生成 top-k 候选意图，再从 benchmark label sets 构建的多专家证据空间中提取 `object / scene / style / activity` 证据，随后将这些证据与 intent-specific template 进行匹配，并将 verification score 作为 residual signal 回注到候选重排序过程，最后继续使用 class-wise threshold 输出最终标签集合。

实现上，本报告完成了两个版本。`MEV v1` 采用 benchmark bank + image-conditioned similarity matching + template-aware fixed aggregation；`MEV v2` 在此基础上进一步引入 class-wise expert routing，用验证集 per-class gain 决定每个类别应依赖的 expert。结果表明，`v1` 已完整跑通但未正式超过当前 strongest `SLR-C`；`v2` 明显优于 `v1`，并在 hard subset 上基本追平甚至略高于 `SLR-C`，说明 expert routing 是比固定 all-expert aggregation 更合理的方向。

---

## 2. 背景与问题定义

### 2.1 任务背景

Intentonomy 不是普通的 object / scene / action 分类任务。其类别包含明显的高层、抽象、主观语义，例如：

- `Enjoy life`
- `Appreciating fine design`
- `Exploration`
- `Being happy and content`

这类标签通常需要多种视觉线索共同支持，而不是由单个局部概念直接决定。因此，本任务更适合被理解为：

> 图像是否提供了足够一致的证据来支持某个高层 intent hypothesis。

### 2.2 当前主线的已知事实

现有实验已经验证：

1. 强 frozen CLIP baseline 已显著优于许多更复杂的结构化方法。
2. `text-only zero-shot` 基本无效，说明全局 semantic classifier 不是正确方向。
3. semantic prior 在 baseline 的 top-k 候选内部做 local rerank 时有效。
4. reranked score geometry 需要重新做 calibration，尤其是 class-wise threshold。

因此，当前更准确的问题表述是：

> 在不破坏强视觉 baseline 的前提下，能否通过显式证据核验进一步提升候选意图的局部判别质量，并把这种提升转化为更好的多标签输出。

### 2.3 基础记号

给定图像 `x`，类别集合为 `Y = {1, ..., C}`。强视觉 baseline 输出 logits：

$$
z(x) \in \mathbb{R}^{C}
$$

当前 strongest `SLR-C` 在 top-k 候选内部引入 `scenario prior`：


$$\tilde z_c =
\begin{cases}
z_c + \alpha \tilde s_c^{scn}(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}$$


其中：

- $\mathcal{T}_k(x)$ 为 baseline 提出的 top-k 候选集
- $\tilde s_c^{scn}(x)$ 为归一化后的 scenario prior score

MEV-C 在此基础上再加入 verification signal：

$$
u_c =
\begin{cases}
\tilde z_c + \beta q_c(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
$$

最终通过 calibrated decision rule 得到预测：

$$
\hat y_c = \mathbb{I}(\sigma(u_c) > t_c)
$$

其中 `t_c` 为 class-wise threshold。

---

## 3. 当前基线与研究动机

### 3.1 当前 strongest baseline

当前主线中最有代表性的几组结果如下：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| Baseline + global threshold | 45.98 | 56.54 | 56.40 | 26.86 |
| Baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 29.71 |
| Scenario SLR + global threshold | 47.14 | 56.52 | 55.11 | 28.32 |
| Scenario SLR + class-wise threshold (`SLR-C`) | **51.28** | **59.13** | **58.47** | **33.98** |

这里可以看到两点：

- local rerank 本身有效
- class-wise calibration 对 reranked score space 至关重要

### 3.2 Candidate proposal 已经足够强

已有 candidate recall / oracle analysis 表明：

- top-10 label recall = `83.91%`
- top-10 `sample_any_recall = 95.07%`
- top-10 `sample_all_recall = 75.33%`

因此，当前系统的主要瓶颈不是 candidate proposal capacity，而是：

- top-k 候选内部的局部语义边界
- rerank 后得分空间与最终 decision rule 的不匹配

### 3.3 从 rerank 走向 evidence verification

当前 `SLR-C` 仍然主要依赖整体图文相似度，它回答的问题更像：

> “这个 intent 的文本描述与图像整体像不像？”

但对 Intentonomy 来说，更自然的问题是：

> “图像里是否同时存在支持该 intent 的 object / scene / style / activity 证据？”

这正是 evidence verification 的切入点。

---

## 4. 方法概述

### 4.1 整体框架

MEV-C 将意图识别组织为如下流程：

```text
Image
  -> strong visual baseline / scenario SLR
  -> top-k intent hypotheses
  -> multi-expert evidence extraction
  -> hypothesis-conditioned evidence matching
  -> verification rerank
  -> calibrated multi-label decision
```

该流程对应的 agent 叙事是：

1. hypothesis generation
2. evidence collection
3. evidence verification
4. calibrated decision

### 4.2 Stage 0：Candidate Proposal

候选提出阶段直接沿用当前 strongest pipeline：

- frozen CLIP visual backbone
- `scenario prior`
- local rerank
- `top-k = 10`

MEV-C 不试图替代 proposal，而是把工作重点放在 candidate-local residual correction 上。

### 4.3 Stage 1：Intent Evidence Template Construction

对每个 intent `c`，构建 evidence template：

$$
T_c = \{T_c^{obj}, T_c^{scene}, T_c^{style}, T_c^{act}\}
$$

其中：

- `T_c^{obj}`：相关 object / attribute 证据
- `T_c^{scene}`：典型场景与环境
- `T_c^{style}`：风格、氛围、构图、审美线索
- `T_c^{act}`：典型活动或 interaction 线索

模板来源为：

- `intent_description_gemini.json`
- `intent2concepts.json`

模板的作用是定义每个 intent 期望出现的支持性证据，而不是直接充当分类器。

### 4.4 Stage 2：Benchmark-Based Evidence Extraction

最终实现中，证据抽取不再使用手工 generic bank，而使用标准 benchmark label sets：

| Expert | Label Set | Size |
| --- | --- | ---: |
| object | COCO | 80 |
| scene | Places365 | 365 |
| style | Flickr Style | 20 |
| activity | Stanford40 | 40 |

证据 bank 的作用是提供一个公开、稳定、可复用的 visual concept space。图像特征与 bank 中所有标签的 CLIP text embedding 做相似度，形成每个 expert 的 evidence score matrix。

> Recognizing Image Style https://arxiv.org/pdf/1311.3715

### 4.5 Stage 3：Hypothesis-Conditioned Evidence Matching

对 expert `e` 与候选类别 `c`，定义匹配分数：

$$
m_e(x, c) = Match(B_e(x), T_c^e)
$$

当前实现采用 `benchmark bank + image-conditioned similarity matching`：

1. 先使用图像特征与 benchmark bank text embeddings 计算每个 expert 的 evidence score matrix
2. 对 bank label 和 template phrase 分别编码到 CLIP text space
3. 计算 template phrase 与 bank labels 的 text similarity
4. 用图像侧的 bank evidence score 与 template-bank similarity 共同计算 class-specific support

该设计的核心优点是：

- 保留 benchmark label space 的公开性与可解释性
- 不要求 template phrase 与 bank label 完全一致
- 能够在统一 CLIP space 中处理 label mismatch
- matching 过程显式使用图像特征，而不仅仅依赖 top-k 候选类别

### 4.6 Stage 4：Verification Aggregation

#### v1：Template-Aware Fixed Aggregation

`v1` 使用固定但 template-aware 的聚合方式：

$$
q_c^{(v1)}(x) = \sum_{e \in \mathcal{E}} w_{c,e} m_e(x, c)
$$

其中 `w_{c,e}` 在 template 缺失时置零，并对剩余专家归一化。

#### v2：Class-Wise Expert Routing

`v1` 的问题在于所有类别共享同一套 expert aggregation policy，容易稀释强单 expert 的有效信号。

因此 `v2` 引入 class-wise routing：

$$
q_c^{(v2)}(x) = \sum_{e \in \mathcal{E}} R_{c,e} \, m_e(x, c)
$$

其中 `R \in \mathbb{R}^{C \times |\mathcal{E}|}` 为 routing matrix，由 validation per-class gain 相对 `SLR-C` 构造。

当前实现评估了以下 routing modes：

- `top1_always`
- `top1_positive`
- `top2_soft`

其中 `top1_positive` 的含义是：

- 对每个类别只保留验证集上相对 `SLR-C` 有正增益的最佳 expert
- 若无正增益 expert，则该类不启用 MEV，直接退回 `SLR-C`

### 4.7 Stage 5：Verification Rerank + Calibrated Decision

verification score 作为 residual signal 注入候选 logits：

$$
u_c =
\begin{cases}
\tilde z_c + \beta q_c(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
$$

实验表明，verification 之后仍需使用 class-wise threshold，才能将局部排序收益转化为最终 multi-label 输出。

---

## 5. 实现与工程形态

### 5.1 代码入口

核心实现位于：

- `src/utils/evidence_verification.py`
- `scripts/analyze_agent_evidence_verification.py`

其中：

- `evidence_verification.py` 负责 template 构建、benchmark bank 构建、matching 与 aggregation
- `analyze_agent_evidence_verification.py` 负责完整分析流程、搜索与落盘

### 5.2 实验产物

当前已完成的主要输出目录：

- `logs/analysis/full_agent_evidence_verification_20260310`
- `logs/analysis/full_agent_evidence_verification_v2_20260310`

产物包括：

- `summary.json`
- `search_results.csv/json`
- `routing_search_results.csv/json`
- `evidence_templates.json`
- `expert_phrase_banks.json`
- `expert_dependency.csv`
- `val_top_evidence_preview.json`

### 5.3 运行协议

当前方法的实现形态可以概括为：

- **无需额外训练新的 verification 模块**
- 但**不是严格意义上的 zero-training**

更准确地说：

1. 方法依赖一个已经训练好的 baseline checkpoint
2. evidence bank、template matching 与 rerank 本身不引入新的可训练参数
3. 最终结果仍然需要 validation-time calibration / search
   - 例如 `beta`
   - class-wise thresholds
   - `v2` 中的 routing matrix

因此，当前 MEV 更适合被描述为：

> **training-free add-on on top of a pretrained baseline, with validation-time calibration/search**

`v1` 主实验口径：

```bash
python scripts/analyze_agent_evidence_verification.py \
  --bank-source benchmark \
  --match-mode similarity \
  --output-dir logs/analysis/full_agent_evidence_verification_20260310
```

`v2` 主实验口径：

```bash
python scripts/analyze_agent_evidence_verification.py \
  --bank-source benchmark \
  --match-mode similarity \
  --routing-modes top1_always,top1_positive,top2_soft \
  --routing-gamma-list 4,8 \
  --routing-gain-floor-list 0.0 \
  --output-dir logs/analysis/full_agent_evidence_verification_v2_20260310
```

---

## 6. 实验设置

### 6.1 数据集与指标

数据集：

- `Intentonomy`

主指标：

- Macro F1
- Micro F1
- Samples F1
- Hard subset F1

其中 `hard` 使用仓库内既有 hard subset 定义。

### 6.2 对比系统

本报告关注以下几组系统：

| System | 说明 |
| --- | --- |
| Baseline | frozen CLIP baseline |
| Baseline + class-wise threshold | baseline 的 class-wise calibration |
| SLR-C | `scenario SLR + class-wise threshold` |
| MEV v1 | benchmark bank + similarity matching + fixed aggregation |
| MEV v2 | benchmark bank + similarity matching + class-wise expert routing |

### 6.3 模型选择原则

正式结果均采用 validation-selected 配置，不使用 test 最优值作为主结论。

单 expert 的 best test 结果只作为诊断，不作为正式主表。

---

## 7. 实验结果

### 7.1 主结果

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| Baseline + global threshold | 45.98 | 56.54 | 56.40 | 26.86 |
| Baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 29.71 |
| Scenario SLR + global threshold | 47.14 | 56.52 | 55.11 | 28.32 |
| `SLR-C` | **51.28** | **59.13** | **58.47** | 33.98 |
| `MEV v1` | 50.84 | 58.17 | 57.21 | 33.53 |
| `MEV v2` | 51.10 | 59.00 | 58.24 | **33.99** |

表中结论：

- `MEV v1` 相比 `SLR-C` 仍略弱
- `MEV v2` 明显优于 `MEV v1`
- `MEV v2` 在 hard 上基本追平甚至略高于 `SLR-C`
- `MEV v2` 的 macro / micro / samples 仍略低于 `SLR-C`

### 7.2 MEV v1 结果

`v1` 的 validation-selected 最优配置为：

- subset: `all`
- fusion: `add_norm`
- `beta = 0.1`

其 test 指标为：

- `macro = 50.84`
- `micro = 58.17`
- `samples = 57.21`
- `hard = 33.53`

相对 `SLR-C`：

- `macro -0.44`
- `hard -0.45`

因此，`v1` 的主要结论是：

> benchmark-bank evidence verification 已经完整跑通，但固定 all-expert aggregation 尚不足以正式超过 strongest `SLR-C`。

### 7.3 单 expert 诊断

将每个 single expert 在 test 上的最好结果作为诊断信号，可得到：

| Expert | Best Test Macro | Best Test Hard | 说明 |
| --- | ---: | ---: | --- |
| object | 51.11 | 34.35 | 有用，但不是最强 |
| scene | 51.05 | 33.68 | 有用，但收益有限 |
| style | **51.86** | **35.21** | 当前最强单 expert |
| activity | 51.58 | 34.79 | 次强单 expert |

这些结果不用于正式模型选择，但用于回答“哪类 expert 更有潜力”。结论很清楚：

- `style` 与 `activity` 是最有潜力的证据源
- 简单 all-expert aggregation 并不能自动把这些强信号转化为更好的正式结果

### 7.4 MEV v2 结果

`v2` 的 validation-selected 最优配置为：

- routing mode: `top1_positive`
- `beta = 0.3`
- fusion: `add_norm`
- `num_routed_classes = 20`

其 test 指标为：

- `macro = 51.10`
- `micro = 59.00`
- `samples = 58.24`
- `hard = 33.99`

相对 `v1`：

- `macro +0.26`
- `hard +0.46`

相对 `SLR-C`：

- `macro -0.18`
- `hard +0.01`

`v2` 的 best routing distribution 为：

- `object`: 2 classes
- `scene`: 7 classes
- `style`: 1 class
- `activity`: 10 classes

剩余 8 个类别没有被 route 到正增益 expert，直接退回 `SLR-C`。

### 7.5 Global Threshold 的补充观察

`v2` 在 global threshold 下的最佳配置为：

- routing mode: `top2_soft`
- `gamma = 8`
- `beta = 0.1`

其 test 指标为：

- `macro = 49.25`
- `hard = 35.01`

这说明：

- softer routing 对 hard subset 仍然有吸引力
- 但正式主表仍应以 class-wise threshold 结果为主

---

## 8. 结果分析

### 8.1 证据核验是可行的，但 aggregation 方式决定上限

`v1` 已经证明：

- benchmark label sets 可用于构建统一 evidence space
- template-conditioned matching 可以真正运行
- verification signal 可以与 `SLR-C` 共存

但 `v1` 的固定聚合方式也清楚暴露出问题：

- 有效 expert 的强信号会被无效 expert 稀释
- 不同类别共享同一 expert policy 并不合理

### 8.2 Routing 比 all-expert fixed aggregation 更重要

`v2` 相比 `v1` 的提升不是来自更多参数，而来自结构上的改变：

- `v1`：所有类共享固定 expert aggregation
- `v2`：每个类独立选择更合适的 expert

这说明当前更重要的问题不是“有没有更多 expert”，而是：

> 如何让不同 intent 依赖不同证据结构。

### 8.3 并非所有类别都需要 verification

`top1_positive` 的最佳 routed config 只对 20 个类别启用了 expert verification，剩余类别直接退回 `SLR-C`。这意味着：

- verification 不是必须全局启用
- 某些类别上引入 expert evidence 没有正增益
- selective activation 是合理方向

### 8.4 Hard subset 受益更明显

虽然 `v2` 的 overall macro 仍略低于 `SLR-C`，但其 hard 指标已经基本追平甚至略高于 `SLR-C`。这说明：

- evidence verification 更接近“困难类别局部消歧器”
- 后续若能把这种 hard-case 增益稳定转化为 overall 增益，方法线就更有说服力

---

## 9. 当前结论

截至目前，可以给出以下技术判断。

### 9.1 已经被验证的结论

1. `MEV` 不是概念性构想，而是已经实现并完整跑通的系统。
2. `benchmark bank + similarity matching + calibration` 是可行技术路线。
3. `MEV v1` 不能正式超过 strongest `SLR-C`，原因主要不在 extraction，而在 aggregation。
4. `MEV v2` 的 class-wise expert routing 明显优于 `v1`，并基本追平 strongest `SLR-C`。

### 9.2 当前最稳妥的主结论

> 当前最合理的 MEV 方向不是继续扩 bank，也不是直接引入 learnable head，而是继续沿着 class-wise routing 与 selective verification 深挖。

### 9.3 当前最值得继续优化的点

- class-wise routing 的构造方式
- uncertainty-aware selective activation
- style / activity 为主的 routing 约束
- 将 hard-case 增益更稳定地转化为 overall gain

---

## 10. 局限性与风险

### 10.1 模板噪声

当前模板由 `scenario prior` 与 `intent2concepts` 自动构造，仍然存在：

- 非视觉短语混入
- 语义过泛
- 同义表达冗余

### 10.2 Label space mismatch

虽然 benchmark label sets 可解释性强，但它们与 intent ontology 并不天然对齐，因此仍需要 CLIP text space 作为桥接层。

### 10.3 当前结果仍是单 checkpoint / 单分析口径

本报告的结论基于当前 strongest baseline checkpoint 与统一分析协议，尚未完成多 seed 稳定性验证。

### 10.4 Routing 仍是手工规则化版本

`v2` 已经表明 routing 有价值，但当前 routing 仍然由 validation gain 构造，尚未引入更细的 sample-level selectivity。

---

## 11. 后续工作

建议按以下顺序推进。

### 11.1 短期优先级

1. 在 `v2` 上加入 uncertainty-aware selective activation
2. 继续搜索 routing gain floor 与更保守的 activation 规则
3. 对 `style` / `activity` 做更强约束的 routed variant
4. 将 `v2` 结果补入正式主表与论文草稿

### 11.2 中期优先级

1. 做多 seed 稳定性验证
2. 做更完整的 per-class / hard-case 分析
3. 评估 template source 对 routing 的影响

### 11.3 暂不优先的方向

以下方向目前不应抢主线：

- 大规模 specialist multi-model pipeline
- 复杂 learnable fusion
- 端到端重训练主干
- negative evidence / multi-step agent loop

原因很简单：当前主问题已经被收束为 routing 与 selective verification，而不是 extraction capacity 不足。

---

## 12. 总结

本报告完成了从设计到实现、从 `v1` 到 `v2` 的一轮闭环验证。结论可以简洁概括为：

- `SLR-C` 仍是当前 strongest overall system
- `MEV v1` 已跑通但未正式超过 `SLR-C`
- `MEV v2` 通过 class-wise expert routing 明显优于 `v1`
- `MEV v2` 在 hard subset 上已经与 `SLR-C` 基本持平，整体表现也已非常接近

因此，MEV 这条线当前已经从“设计假设”进入“可持续优化的技术路线”，后续重点应聚焦于 routing 与 selective verification，而不是继续无条件扩展 expert bank。
