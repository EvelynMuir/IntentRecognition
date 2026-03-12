# 设计：Confusion-Aware、条件触发的多 Agent 协作 Ambiguity Resolution Framework（v3）

## 0. 文档目的

本文档给出下一版方法设计：在保留强 base system（baseline + SLR-C）的前提下，把后续 evidence verification 从“全量、偏重的 verifier pipeline”改写为一个**confusion-aware、条件触发的、多 agent 协作 ambiguity resolution framework**。

目标不是替代 `SLR-C`，而是：

1. 明确 `SLR-C` 是主系统
2. 把后续模块限定为**hard ambiguity resolver**
3. 引入真实的 interaction / routing 机制，使方法不只是模块换名
4. 让整套方法更适合作为论文中的独立贡献来表述

---

## 1. 方法设计

### 1.1 问题定义

当前实验已经表明：

- 强 frozen CLIP baseline 已经提供了足够强的视觉表示
- `scenario SLR-C` 是当前性能提升的主要来源
- 剩余错误主要不在全局召回，而在 **top-k 候选内部的局部混淆**
- 这些混淆通常发生在语义相近 intent 之间，例如：
  - `Happy` vs `Playful`
  - `EnjoyLife` vs `Happy`
  - `FineDesign` vs `FineDesign-Art`

因此，v3 的核心问题不是“如何再做一个更强的全局分类器”，而是：

> 如何在 `SLR-C` 已经给出高召回候选的前提下，仅对高歧义样本触发一个多 agent 协作模块，用显式、比较式的证据完成局部 intent disambiguation？

---

### 1.2 Base System：Baseline + SLR-C

v3 明确把 `SLR-C` 视为主系统，而不是前处理。

#### (1) Visual baseline

输入图像 `x` 后：

- 使用 frozen CLIP `ViT-L/14` visual encoder
- 取最后一层 `CLS + mean(patch)` 特征
- 用 MLP classifier 输出 baseline logits：

$$
z^{base}(x) \in \mathbb{R}^{C}
$$

其中 `C = 28`。

#### (2) SLR: Semantic Local Rerank

对每个 intent，利用 `scenario` 文本描述（来自 `intent_description_gemini.json`）计算 semantic prior：

$$
s^{scenario}(x) \in \mathbb{R}^{C}
$$

仅在 baseline top-k 候选内做局部重排：

$$
\tilde z_c =
\begin{cases}
z_c^{base} + \alpha \cdot \tilde s_c^{scenario}, & c \in \mathcal{T}_k(x) \\
z_c^{base}, & c \notin \mathcal{T}_k(x)
\end{cases}
$$

当前默认：

- `top-k = 10`
- `alpha = 0.3`
- prior fusion = `add_norm`

#### (3) C: Class-wise calibration

在 validation set 上对每个类别单独搜索阈值 `t_c`，最终输出：

$$
\hat y_c^{slr-c} = \mathbb{I}(\sigma(\tilde z_c) > t_c)
$$

#### (4) Base system 的角色

在 v3 中，`SLR-C` 的作用是：

- 产生高质量 top-k candidate proposals
- 提供 margin / uncertainty / confusion signals
- 为后续 ambiguity resolution 提供触发基础

换言之，v3 的后续模块**不负责全局识别**，只负责处理 `SLR-C` 的剩余高歧义错误。

---

### 1.3 v3 总体框架

v3 由四类 agent 组成：

1. **Proposal Agent**：即 `SLR-C`，负责候选生成与初始决策
2. **Ambiguity Router Agent**：判断是否需要进入二阶段 disambiguation，并决定调用哪些 specialist agents
3. **Specialist Evidence Agents**：从不同语义视角为候选对提供比较式证据
4. **Resolver Agent**：聚合多 agent 输出，完成局部裁决并修正最终排序

整体流程如下：

```text
Image
  -> Proposal Agent (baseline + scenario SLR-C)
  -> top-k candidates + uncertainty / confusion signals
  -> Ambiguity Router Agent
       -> if easy: directly use SLR-C output
       -> if ambiguous: dispatch specialist evidence agents
  -> Specialist Evidence Agents (object / scene / style / activity)
  -> pairwise comparative evidence over confusion neighborhood
  -> Resolver Agent
  -> residual local rerank on top-k candidates
  -> class-wise thresholds
  -> final multi-label prediction
```

---

### 1.4 Agent 1：Proposal Agent

Proposal Agent 就是当前最强 base system：`baseline + scenario SLR-C`。

它输出三类信息：

1. reranked logits `\tilde z`
2. top-k 候选集 `\mathcal{T}_k(x)`
3. ambiguity signals，例如：
   - top-1 / top-2 margin
   - top-k score entropy
   - 候选间文本语义相似度
   - 当前候选是否落在高频 confusion pairs / groups 中

Proposal Agent 的目标不是解决所有错误，而是：

> 把问题缩小到一个小候选集，并告诉系统“这个样本是否值得进入更昂贵的局部推理”。

---

### 1.5 Agent 2：Ambiguity Router Agent

这是 v3 与 v2 的关键差异之一。v2 只有 gate，v3 要把 gate 扩展成真正的 **routing / interaction 机制**。

#### 输入

- `SLR-C` 的 top-k 候选
- reranked logits / probabilities
- top-1 / top-2 margin
- top-k candidate semantic similarity
- 训练集 confusion statistics

#### 输出

Router Agent 需要做两个决策：

1. **是否触发二阶段 disambiguation**
2. **触发后调用哪些 specialist agents、作用于哪些 candidate pairs / neighborhoods**

#### 触发规则候选

可以从以下三类信号开始：

- **低 margin 触发**：top-1 与 top-2 分数接近
- **confusion-neighborhood 触发**：top-k 中出现已知高混淆 intent pair / group
- **高语义接近度触发**：候选标签文本嵌入过于接近

最简单的 v3 初版可以从 rule-based routing 开始：

- 若 `m(x) < \tau_m`，触发二阶段
- 若 top-k 中包含 confusion pair，则只对该 pair / group 调用 agents
- 若候选不属于 ambiguity region，则直接输出 `SLR-C`

#### 设计意义

Router Agent 的作用是把后半段从“全量 verifier”改写成：

- 条件触发
- 只针对 hard cases
- 只在局部候选邻域中工作

这是 v3 论文叙事的关键组成部分。

---

### 1.6 Agent 3：Specialist Evidence Agents

v3 不再把 evidence bank 写成一个统一大 verifier，而是写成多个 **role-specialized agents**。

当前保留四类 specialist agents：

1. **Object Agent**：基于 COCO object evidence
2. **Scene Agent**：基于 Places365 scene evidence
3. **Style Agent**：基于 Flickr Style evidence
4. **Activity Agent**：基于 Stanford40 activity evidence

#### 输入

每个 specialist agent 的输入为：

- 图像 `x`
- 被 router 选中的 confusion pair / group
- 相应候选类的 relation profile

#### 输出

每个 agent 不直接输出全局类分数，而是输出：

> 对候选 `c_i` 相比 `c_j` 的**比较式支持证据**

例如：

$$
V_a(c_i, c_j, x)
$$

其中 `a` 表示某个 specialist agent。

#### Evidence computation

保留当前 data-driven evidence verification 的核心思想，但用途改写为 pairwise disambiguation：

1. 先计算该 agent 对图像的 evidence activation
2. 利用训练集学习到的 relation profiles，得到某类对这些 evidence 的相对支持
3. 对候选对 `(c_i, c_j)` 计算差异支持：

$$
\Delta V_a(c_i,c_j,x)=\sum_{z \in A_a(x)} w_z(x) [R_a(c_i,z)-R_a(c_j,z)]
$$

其中：

- `A_a(x)` 是该 agent 激活的稀疏 evidence 集
- `R_a(c,z)` 是类别 `c` 对 evidence `z` 的 relation score

#### v3 中 relation 的使用原则

- 默认以判别式 relation 为主：`hard_negative_diff`
- `support_contradiction` 作为第二优先实验分支
- 不再做大而散的 relation family 搜索
- 优先围绕 confusion neighborhood 重新定义 relation 的作用范围

---

### 1.7 Agent 间 interaction 机制

为了避免“只是并列模块”的问题，v3 必须引入真实 interaction。建议至少包含以下三种：

#### (1) Router-to-Agent dispatch

Router 不一定总是调用全部 specialist agents，而是根据 ambiguity 类型做选择：

- 若混淆更偏动作语义，优先调用 `Activity Agent`
- 若混淆更偏场景语义，优先调用 `Scene Agent`
- 若混淆更偏审美/设计语义，优先调用 `Style Agent`
- 初版若不做语义型 dispatch，也至少做“部分调用 vs 全调用”比较

#### (2) Candidate-pair-specific activation

specialist agents 不对所有 top-k 候选统一打分，而是：

- 只对被 router 选中的 pair / group 工作
- 对不同 pair 产生不同的 comparative evidence

#### (3) Resolver-mediated aggregation

最终不是简单把各 agent 分数直接相加，而是由 Resolver Agent 做统一裁决，见下节。

---

### 1.8 Agent 4：Resolver Agent

Resolver Agent 负责把多个 specialist agents 的 comparative evidence 转换成最终的 candidate rerank residual。

#### 输入

- Proposal Agent 给出的 reranked logits `\tilde z`
- Router 选出的 ambiguity pairs / groups
- 各 specialist agents 的 pairwise evidence outputs

#### 核心作用

Resolver 负责回答：

> 在当前 ambiguity region 内，多个 evidence agents 的意见综合起来，哪个候选更应该被保留在更高位置？

#### 一种基本实现

对任意候选对 `(c_i, c_j)`，聚合所有被调用 agent 的输出：

$$
S(c_i,c_j,x)=\sum_{a \in \mathcal{A}(x)} \lambda_a(x,c_i,c_j) \cdot \Delta V_a(c_i,c_j,x)
$$

其中：

- `\mathcal{A}(x)` 是 router 选中的 specialist agent 集
- `\lambda_a` 是 resolver 对 agent `a` 的权重，可为固定权重、rule-based 权重或 learned weight

然后把 pairwise preference 聚合到每个候选上：

$$
V^{res}(c_i,x)=\frac{1}{|\mathcal{N}(c_i)|} \sum_{c_j \in \mathcal{N}(c_i)} S(c_i,c_j,x)
$$

其中 `\mathcal{N}(c_i)` 是当前 confusion neighborhood。

最终只对 ambiguity neighborhood 内的候选做局部修正：

$$
z_c^{final}=\tilde z_c + \beta \cdot g(x) \cdot V^{res}(c,x)
$$

其中：

- `g(x)` 仍保留 uncertainty-aware gate
- 但现在 gate 属于 **router + resolver 共同控制的条件介入机制**，不再只是一个孤立缩放项

#### Resolver 的论文意义

Resolver 的作用是把系统从“多分数拼接”改写成：

- router 触发
- specialists 提供局部证据
- resolver 裁决冲突意见

这一步是 v3 是否能被合理写成 multi-agent framework 的关键。

---

### 1.9 v3 与 v2 的关键差异

相对于 v2，v3 的变化不是单纯多一个超参数，而是方法组织方式的变化：

1. **从全量 verifier 到条件触发 disambiguator**
   - v2：所有样本都可进入同一 verifier 流程
   - v3：只有 ambiguity cases 触发局部多 agent 协作

2. **从统一 evidence scoring 到 specialist agents**
   - v2：evidence spaces 更像统一 bank
   - v3：object / scene / style / activity 变成角色明确的 specialist agents

3. **从 gate 到 router + resolver**
   - v2：gate 主要决定 verifier 强度
   - v3：router 决定是否触发、调用谁；resolver 决定如何裁决

4. **从全 top-k 比较到 confusion neighborhood 比较**
   - v2：以 top-k pairwise compare 为主
   - v3：优先在高混淆 pair / group 中进行 targeted comparison

---

### 1.10 推荐的 v3 最小可行版本（MVP）

为了避免一开始做得过重，建议 v3 先落一个最小闭环：

#### Step A. 保持 Proposal Agent 不变
- 直接复用当前 `scenario SLR-C`

#### Step B. 先做 rule-based Router
- 用 top-1/top-2 margin + confusion pair 命中规则决定是否触发
- 暂时不做复杂 learned router

#### Step C. 保持四类 specialist agents，但先支持“全调用”和“部分调用”两种模式
- 全调用：作为稳妥 baseline
- 部分调用：作为 interaction 机制的关键验证

#### Step D. Resolver 先做简单可解释版
- 对各 agent comparative outputs 做加权和
- 再聚合成 candidate residual
- 权重先固定或 rule-based，再考虑学习

这样可以先验证：

- 多 agent 条件触发是否有效
- interaction 是否真的比 v2 更有价值
- targeted disambiguation 是否能在 hard cases 上给出更清晰收益

---

## 2. 实验清单

下面按“必须做 / 建议做 / 加分项”列出实验。

### 2.1 必须做：Base system 与 v3 的主对比

#### E1. 主结果对比
比较以下系统：

1. baseline
2. `scenario SLR-C`
3. v2 best method
4. v3 MVP（rule-based router + all specialists + resolver）
5. v3 routed（router + selective specialists + resolver）

报告指标：

- Macro F1
- Micro F1
- Samples F1
- mAP
- Hard

目标：证明 v3 至少在 `Hard` 与 `Macro` 上有清晰改进，且不会因为 agent 化导致整体崩坏。

---

### 2.2 必须做：ambiguity-triggered 机制验证

#### E2. 全量调用 vs 条件触发
比较：

1. 所有样本都调用 specialist agents
2. 只对低 margin 样本调用
3. 只对 confusion-neighborhood 样本调用
4. 同时使用 margin + confusion routing

目标：验证“条件触发”不是叙事包装，而是真的：

- 更符合任务结构
- 更节省计算
- 更适合 hard ambiguity

需要额外报告：

- 触发比例
- 平均每样本调用 agent 数
- 推理额外开销

---

### 2.3 必须做：specialist 分工价值验证

#### E3. Unified verifier vs specialist agents
比较：

1. 单一 unified evidence scorer
2. object / scene / style / activity 四个 specialist agents 并行
3. specialist agents + selective routing

目标：验证 specialist decomposition 是否有实际价值，而不是单一 verifier 就足够。

---

### 2.4 必须做：interaction 机制验证

#### E4. No interaction vs routed interaction
比较：

1. 所有 specialists 固定全调用 + 直接求和
2. router 决定是否触发，但仍全调用 specialists
3. router 决定触发 + selective specialist dispatch
4. router + selective dispatch + resolver aggregation

目标：证明 v3 的增益不仅来自“多几个模块”，而来自：

- routing
- dispatch
- resolver

这组实验是 reviewer 最关心的。

---

### 2.5 必须做：Resolver 作用验证

#### E5. 简单融合 vs resolver
比较：

1. 各 agent 输出直接平均
2. 固定加权和
3. pairwise resolver aggregation
4. pair/group-aware resolver

目标：验证 resolver 是否真正提升 candidate-local adjudication，而不是装饰性命名。

---

### 2.6 必须做：hard ambiguity 子集评估

#### E6. Hard subset / confusion subset 分析
除了全局指标，额外构建并报告：

1. low-margin subset
2. top confusion pairs subset
3. semantic-neighbor subset

对这些子集报告：

- Macro F1
- pairwise ranking accuracy
- top-2 disambiguation accuracy
- Hard

目标：证明 v3 主要解决的是其宣称的目标问题，而不是依赖全局平均指标讲故事。

---

### 2.7 必须做：关键消融

#### E7. Router 消融
- margin only
- confusion only
- semantic similarity only
- margin + confusion
- margin + confusion + semantic similarity

#### E8. Specialist 消融
- object only
- scene only
- style only
- activity only
- all specialists
- routed subset

#### E9. Relation 消融
- `hard_negative_diff`
- `support_contradiction`
- confusion-neighborhood version

目标：确定 v3 中哪些 interaction / evidence / relation 是真正有效的。

---

### 2.8 建议做：效率与复杂度分析

#### E10. 额外计算成本分析
统计：

- 平均每张图调用多少 specialists
- 条件触发后整体额外 FLOPs / latency
- 与 v2 相比的额外代价

目标：支持“v3 是 targeted disambiguation，而不是无节制加重模块”。

---

### 2.9 建议做：定性案例分析

#### E11. Case studies
至少展示 4~6 个典型样本：

- `SLR-C` 的 top-k 候选
- router 是否触发及原因
- 哪些 specialists 被调用
- 每个 specialist 支持谁、为什么
- resolver 最终如何改写排序
- 最终是否修正成功

优先选：

- `Happy` vs `Playful`
- `EnjoyLife` vs `Happy`
- `FineDesign` vs `FineDesign-Art`

目标：给 reviewer 直观证明：

- v3 真在做 comparative ambiguity resolution
- “agent”不是空洞命名

---

### 2.10 建议做：稳定性实验

#### E12. 稳定性与鲁棒性
- 多随机种子
- 不同 routing threshold
- 不同 confusion pair 数量
- 不同 neighborhood 大小

目标：缓解 reviewer 对 tuned heuristic / validation overfitting 的担忧。

---

### 2.11 可选加分项：学习式 Router / Resolver

#### E13. Learned router
在 rule-based router 跑通后，可尝试轻量学习式 router：

- 输入：margin、entropy、candidate similarity、confusion statistics
- 输出：是否触发 / 调用哪些 specialists

#### E14. Learned resolver
在固定 resolver 跑通后，可尝试：

- 小 MLP 预测 `\lambda_a`
- pair-aware adaptive weighting

注意：这部分是加分项，不应早于 MVP 验证。

---

## 3. 当前建议的实施顺序

建议按以下顺序推进：

### Phase 1：先证明 v3 的框架价值
1. 固定 Proposal Agent = 当前 `scenario SLR-C`
2. 做 rule-based Router
3. 保留当前 evidence computation
4. 实现 simplest Resolver
5. 先完成 E1 / E2 / E4 / E6

### Phase 2：再验证多 agent interaction 的必要性
6. 做 specialist vs unified 对比（E3）
7. 做 resolver 消融（E5）
8. 做 router / relation / specialist 消融（E7-E9）

### Phase 3：补强论文说服力
9. 做效率分析（E10）
10. 做案例分析（E11）
11. 做稳定性分析（E12）

只有在前面都成立时，再考虑 learned router / learned resolver（E13-E14）。

---

## 4. 一句话总结

v3 的核心不是把旧 pipeline 改名成 agent，而是把后半段重新定义为：

> **在强 `SLR-C` proposal system 之上，一个 confusion-aware、条件触发的、多 specialist agent 协作的局部 ambiguity resolution framework。**

它的目标不是全面替代 base classifier，而是：

- 只在 high-ambiguity cases 中介入
- 只在 confusion neighborhood 内比较候选
- 用多视角显式证据完成局部裁决
- 以最小额外复杂度提供更像“方法贡献”的 hard-case 改进
