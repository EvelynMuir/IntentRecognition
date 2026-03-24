# 技术报告：Multi-Expert Evidence Verification for Visual Intent Recognition

## 0. 文档信息

- 日期：`2026-03-11`
- 状态：内部技术报告
- 任务：Intentonomy 多标签视觉意图识别
- 研究主线：在当前 strongest `SLR-C` 基础上引入显式证据核验

---

## 1. 摘要

本报告总结了 `Multi-Expert Evidence Verification with Calibration (MEV-C)` 的设计、实现与实验结果。研究出发点是：在强 frozen CLIP baseline 已经足够强的前提下，Intentonomy 的剩余难点不再主要来自视觉特征抽取，而来自候选意图之间的局部语义消歧与最终多标签决策。已有 `SLR-C` 已经证明 `category-level prior + local rerank + class-wise calibration` 是有效路线；在本版设计中，候选阶段只使用 category-level prior，例如 `short`、`detailed`，以及最简单的 `short + detailed` 加权求和 ensemble，不再使用场景级 prior，但该方法仍然缺少对图像中具体支持证据的显式核验。

为此，本工作将意图识别重构为四阶段流程：`hypothesis generation -> evidence extraction -> hypothesis verification -> calibrated decision`。系统首先使用 `category-level prior rerank` 生成 top-k 候选意图，再从 benchmark label sets 构建的多专家证据空间中提取 `object / scene / style / activity` 证据，随后将这些证据与 intent-specific template 进行匹配，并将 verification score 作为 residual signal 回注到候选重排序过程，最后继续使用 class-wise threshold 输出最终标签集合。

实现上，本报告完成了两个版本。`MEV v1` 采用 benchmark bank + image-conditioned similarity matching + template-aware fixed aggregation；`MEV v2` 在此基础上进一步引入 class-wise expert routing，用验证集 per-class gain 决定每个类别应依赖的 expert。`2026-03-11` 的重跑结果表明：在新的 `category-level prior only` 设定下，当前 strongest overall system 变为 `short + detailed` 加权求和后的 `SLR-C` baseline；`MEV v1/v2` 仍然可以在部分 source 上提供 hard-case residual gain，但尚未正式超过这个 strongest baseline。其中最强的 `MEV` 变体是 `short + MEV v2`，说明 expert routing 仍然有价值，但其收益已经明显受 Stage 0 source 选择约束。

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
3. category-level semantic prior 在 baseline 的 top-k 候选内部做 local rerank 时有效。
4. reranked score geometry 需要重新做 calibration，尤其是 class-wise threshold。

因此，当前更准确的问题表述是：

> 在不破坏强视觉 baseline 的前提下，能否通过显式证据核验进一步提升候选意图的局部判别质量，并把这种提升转化为更好的多标签输出。

### 2.3 基础记号

给定图像 `x`，类别集合为 `Y = {1, ..., C}`。强视觉 baseline 输出 logits：

$$
z(x) \in \mathbb{R}^{C}
$$

当前 Stage 0 设计在 top-k 候选内部引入 `category-level prior`：


$$\tilde z_c =
\begin{cases}
z_c + \alpha \tilde s_c^{cat}(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}$$


其中：

- $\mathcal{T}_k(x)$ 为 baseline 提出的 top-k 候选集
- $\tilde s_c^{cat}(x)$ 为归一化后的 category-level prior score
- prior source 只取自 `short`、`detailed` 或最简单的 `short + detailed` ensemble

若采用简单 ensemble，则可写为：

$$
s_c^{cat}(x)=\frac{1}{2}\left(s_c^{short}(x)+s_c^{detailed}(x)\right)
$$

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

按 Stage 0 更新后的 `category-level prior only` 重跑结果，当前最有代表性的 baseline 如下：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| Baseline + global threshold | 45.98 | 56.54 | 56.40 | 26.86 |
| Baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 29.71 |
| `short` prior + class-wise threshold | 50.66 | 57.23 | 57.34 | **33.28** |
| `detailed` prior + class-wise threshold | 50.25 | 57.27 | 57.22 | 33.23 |
| `short + detailed` weighted prior + class-wise threshold | **50.86** | 57.19 | 57.24 | 33.23 |

这里可以看到两点：

- local rerank 本身有效
- class-wise calibration 对 reranked score space 至关重要
- 在 category-level prior-only 设定下，最强 baseline 来自 `short + detailed` 的简单加权求和，而不是单一 source

作为历史参考，旧的 `scenario`-based `SLR-C` 为 `51.28 / 59.13 / 58.47 / 33.98`，整体仍高于当前 category-only proposal。

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
  -> strong visual baseline / category-level prior rerank
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

候选提出阶段改为只使用 category-level prior：

- frozen CLIP visual backbone
- `short` prior
- `detailed` prior
- optional weighted `short + detailed` ensemble
- local rerank
- `top-k = 10`

不使用场景级 prior。MEV-C 不试图替代 proposal，而是把工作重点放在 candidate-local residual correction 上。

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

- `logs/analysis/full_agent_evidence_verification_lexical_20260311`
- `logs/analysis/full_agent_evidence_verification_canonical_20260311`
- `logs/analysis/full_agent_evidence_verification_short_plus_detailed_20260311`

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

`short` source 主实验口径：

```bash
python scripts/analyze_agent_evidence_verification.py \
  --slr-source lexical \
  --bank-source benchmark \
  --match-mode similarity \
  --routing-modes top1_always,top1_positive,top2_soft \
  --routing-gamma-list 4,8 \
  --routing-gain-floor-list 0.0 \
  --output-dir logs/analysis/full_agent_evidence_verification_lexical_20260311
```

`detailed` source 主实验口径：

```bash
python scripts/analyze_agent_evidence_verification.py \
  --slr-source canonical \
  --bank-source benchmark \
  --match-mode similarity \
  --routing-modes top1_always,top1_positive,top2_soft \
  --routing-gamma-list 4,8 \
  --routing-gain-floor-list 0.0 \
  --output-dir logs/analysis/full_agent_evidence_verification_canonical_20260311
```

`short + detailed` weighted source 主实验口径：

```bash
python scripts/analyze_agent_evidence_verification.py \
  --slr-source short_plus_detailed \
  --bank-source benchmark \
  --match-mode similarity \
  --routing-modes top1_always,top1_positive,top2_soft \
  --routing-gamma-list 4,8 \
  --routing-gain-floor-list 0.0 \
  --output-dir logs/analysis/full_agent_evidence_verification_short_plus_detailed_20260311
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
| `SLR-C (short)` | `short prior rerank + class-wise threshold` |
| `SLR-C (detailed)` | `detailed prior rerank + class-wise threshold` |
| `SLR-C (short + detailed)` | `short / detailed` prior logits 加权求和后 rerank，再做 class-wise threshold |
| `MEV v1` | benchmark bank + similarity matching + fixed aggregation |
| `MEV v2` | benchmark bank + similarity matching + class-wise expert routing |

### 6.3 模型选择原则

正式结果均采用 validation-selected 配置，不使用 test 最优值作为主结论。

单 expert 的 best test 结果只作为诊断，不作为正式主表。

---

## 7. 实验结果

### 7.1 主结果

| Stage-0 Source | Method | Macro | Micro | Samples | Hard |
| --- | --- | ---: | ---: | ---: | ---: |
| `short` | `SLR-C` | 50.66 | 57.23 | 57.34 | 33.28 |
| `short` | `MEV v1` | 50.84 | 57.87 | 57.73 | 33.13 |
| `short` | `MEV v2` | 50.85 | 57.31 | 57.62 | **33.66** |
| `detailed` | `SLR-C` | 50.25 | 57.27 | 57.22 | 33.23 |
| `detailed` | `MEV v1` | 50.07 | 57.67 | 57.48 | 32.32 |
| `detailed` | `MEV v2` | 50.08 | 57.43 | 57.32 | 33.61 |
| `short + detailed` weighted | `SLR-C` | **50.86** | 57.19 | 57.24 | 33.23 |
| `short + detailed` weighted | `MEV v1` | 50.57 | **58.76** | **58.10** | 32.62 |
| `short + detailed` weighted | `MEV v2` | 49.94 | 58.40 | 57.97 | 32.37 |

表中结论：

- 在新的 `category-level prior only` 设定下，overall macro 最强的是 `short + detailed` 加权求和后的 `SLR-C` baseline。
- 当前 `MEV` 没有在任何 source 上正式超过这个 strongest baseline。
- `short` 是唯一能让 `MEV` 在 macro 上带来稳定正增益的 source。
- `MEV v2` 对 hard subset 仍然有帮助，`short` 与 `detailed` 两个单源下都高于各自的 `SLR-C`。

作为历史参考，旧的 `scenario`-based `SLR-C` 为 `51.28 / 59.13 / 58.47 / 33.98`，仍然高于当前 category-only setting。

### 7.2 MEV v1 结果

`v1` 在不同 Stage-0 source 下的 validation-selected 最优配置分别为：

- `short`: `scene` expert only + `beta = 0.3`
- `detailed`: `all` experts + `beta = 0.3`
- `short + detailed`: `activity` expert only + `beta = 0.2`

对应 test 指标为：

- `short + MEV v1`: `50.84 / 57.87 / 57.73 / 33.13`
- `detailed + MEV v1`: `50.07 / 57.67 / 57.48 / 32.32`
- `short + detailed + MEV v1`: `50.57 / 58.76 / 58.10 / 32.62`

这里最关键的现象是：

- `v1` 对 source 非常敏感，并不存在统一最优的 expert subset。
- `short` source 下，`scene` expert 能带来小幅 macro 增益，但 hard 略低于 baseline。
- `short + detailed` weighted source 下，`v1` 会把 micro / samples 推高，但无法保住 macro / hard。

因此，`v1` 的主要结论变为：

> benchmark-bank evidence verification 仍然可行，但 fixed aggregation 的收益高度依赖 Stage 0 source，尚不足以稳定超过 strongest weighted `SLR-C`。

### 7.3 Source-Dependent Expert Preference

在新的 Stage 0 设定下，不同 prior source 对 expert 的偏好明显不同：

- `short` prior 最优配置偏向 `scene`，说明 lexical category name 留下的 residual ambiguity 更像场景消歧问题。
- `detailed` prior 最优配置反而是 `all` experts，说明 canonical semantics 已经覆盖较多信息，单 expert 不再明显占优。
- `short + detailed` weighted prior 最优配置偏向 `activity`，说明加权 ensemble 后剩余可补充的信息更像 interaction / activity residual。

这说明当前更重要的问题已经不是“哪个 expert 全局最强”，而是：

> 不同 Stage 0 source 会改变 verification signal 的有效形态。

### 7.4 MEV v2 结果

`v2` 在不同 Stage-0 source 下的 validation-selected 最优配置分别为：

- `short`: `top2_soft + gamma=4 + beta=0.2`
- `detailed`: `top1_positive + beta=0.2`
- `short + detailed`: `top1_always + beta=0.1`

对应 test 指标为：

- `short + MEV v2`: `50.85 / 57.31 / 57.62 / 33.66`
- `detailed + MEV v2`: `50.08 / 57.43 / 57.32 / 33.61`
- `short + detailed + MEV v2`: `49.94 / 58.40 / 57.97 / 32.37`

其中最强的 `MEV v2` 是 `short + MEV v2`。它相对 `short SLR-C`：

- `macro +0.19`
- `hard +0.38`

但相对 strongest weighted `SLR-C`：

- `macro -0.11`
- `hard +0.43`

这说明：

- routing 仍然能作为 hard-case residual corrector 工作
- 但它还不能稳定保住 strongest baseline 的 overall macro
- 当 Stage 0 已经是较强的 weighted prior 时，当前 routing 反而容易过度修正

`v2` 的 best routing 形态也随 source 改变：

- `short` 最优是 softer `top2_soft` routing
- `detailed` 最优是更保守的 `top1_positive`，只激活 `17` 个 routed classes
- `short + detailed` 最优却退化为 `top1_always`，并对 `28` 个类别全部路由，最终导致 macro 明显下降

### 7.5 Calibration 的补充观察

新的重跑结果继续说明：

- class-wise threshold 仍然是正式主表的必要组成
- verification 的收益并不自动等价于最终 macro 增益
- 在 weighted prior setting 下，当前 `MEV` 更容易把 ranking 改善转成 micro / samples，而不是 macro / hard 的稳定提升

---

## 8. 结果分析

### 8.1 Stage 0 source 现在比 verification 本身更决定上限

新的重跑显示，单是把 Stage 0 改成 `short + detailed` weighted prior，就足以得到当前最强的 category-only baseline；反过来，把当前 `MEV` 叠加到它上面反而会掉 macro。说明当前上限更多由候选阶段的 source 质量决定，而不是 bank 容量本身。

### 8.2 Verification 已经变成 source-dependent residual module

在旧结论里，`MEV` 更像一个统一的 candidate-local verifier；但在新的结果里，它更像一个明显依赖 Stage 0 source 的 residual module：

- `short` source 能从 `MEV v2` 中得到正的 macro / hard 增益
- `detailed` source 只能稳定拿到 hard 增益
- weighted `short + detailed` source 会被当前 verification 结构过度修正

因此，后续设计不应再默认“同一个 verification policy 对所有 Stage 0 source 都成立”。

### 8.3 Routing 仍然重要，但必须更保守、更 source-aware

`v2` 仍然优于 `v1` 的核心原因没有变：routing 比 fixed aggregation 更合理。但新的结果说明，routing 的价值现在主要体现在：

- 对较弱单源 prior 做 selective residual correction
- 在 hard subset 上提供额外消歧能力

而不是：

- 无条件覆盖 strongest weighted baseline

### 8.4 当前主问题已转向“不要破坏强 baseline”

此前的问题更像“如何让 verification 更强”；现在更准确的问题是：

> 如何让 verification 只在真正需要时启动，并且不破坏已经足够强的 Stage 0 baseline。

这意味着当前瓶颈已经从“有无 expert signal”转向：

- source-aware activation
- 更保守的 residual strength control
- weighted prior 上的 no-op fallback

---

## 9. 当前结论

截至目前，可以给出以下技术判断。

### 9.1 已经被验证的结论

1. `MEV` 不是概念性构想，而是已经实现并完整跑通的系统。
2. `benchmark bank + similarity matching + calibration` 在 `category-level prior only` 设定下仍然可行。
3. 在当前设定下，strongest overall baseline 是 weighted `short + detailed SLR-C`，其 test 指标为 `50.86 / 57.19 / 57.24 / 33.23`。
4. 当前最强的 `MEV` 变体是 `short + MEV v2`，其 test 指标为 `50.85 / 57.31 / 57.62 / 33.66`；它改善了 hard subset，但尚未正式超过 strongest baseline 的 macro。

### 9.2 当前最稳妥的主结论

> 在 `category-level prior only` 的候选阶段下，当前最稳妥的系统仍是 weighted `short + detailed SLR-C`；MEV 的下一步不应是继续扩 bank，而应是做 source-aware selective verification，在不破坏强 baseline 的前提下保留 hard-case 增益。

### 9.3 当前最值得继续优化的点

- `short : detailed` 的权重搜索，而不是只停留在 `0.5 : 0.5`
- source-aware routing / activation 的构造方式
- uncertainty-aware selective activation
- weighted prior 上的保守 fallback 机制
- 将 hard-case 增益更稳定地转化为 overall macro gain

---

## 10. 局限性与风险

### 10.1 模板噪声

当前模板由 `short / detailed` 类别描述与 `intent2concepts` 自动构造，仍然存在：

- 非视觉短语混入
- 语义过泛
- 同义表达冗余

### 10.2 Label space mismatch

虽然 benchmark label sets 可解释性强，但它们与 intent ontology 并不天然对齐，因此仍需要 CLIP text space 作为桥接层。

### 10.3 当前结果仍是单 checkpoint / 单分析口径 / 单一 ensemble 权重

本报告的结论基于当前 strongest baseline checkpoint 与统一分析协议，尚未完成多 seed 稳定性验证；同时 `short + detailed` 目前只验证了最简单的 `0.5 : 0.5` 加权形式。

### 10.4 Routing 仍是手工规则化且 source-aware 不足

`v2` 已经表明 routing 有价值，但当前 routing 仍然由 validation gain 构造，且没有显式建模 Stage 0 source 与 residual verification 的相互作用，因此在 weighted prior 上容易过度修正。

---

## 11. 后续工作

建议按以下顺序推进。

### 11.1 短期优先级

1. 搜索 `short : detailed` 的 ensemble 权重与 `alpha`
2. 在 weighted prior baseline 上加入更保守的 activation / no-op fallback
3. 继续做 `short` source 下的 `v2` routing 与 uncertainty-aware selective activation
4. 将新的 category-only 结果补入正式主表与论文草稿

### 11.2 中期优先级

1. 做多 seed 稳定性验证
2. 做更完整的 per-source / per-class / hard-case 分析
3. 评估 template source 与 Stage 0 source 的交互影响

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

- 在新的 `category-level prior only` 设计下，当前 strongest overall system 是 weighted `short + detailed SLR-C`
- `MEV v1` 与 `MEV v2` 都已完整跑通，但尚未正式超过这个 strongest baseline
- `short + MEV v2` 是当前最强的 `MEV` 变体，并在 hard subset 上给出最明显收益
- verification 的价值仍然存在，但已经明显受 Stage 0 source 选择约束

因此，MEV 这条线当前仍然是可持续优化的技术路线，但后续重点应从“扩 expert bank”转向“source-aware residual design”，尤其是如何在不破坏 weighted prior baseline 的前提下保留 hard-case 增益。
