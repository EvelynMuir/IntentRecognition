# 实验计划：Region-Grounded Agentic Intent Reasoner

## 0. 文档目的

本文档将 `Region-Grounded Agent` 的初步设想整理为一份完整实验计划。

目标是把当前主要依赖后处理式 verification 的局部消歧思路，升级为一个**进入主干决策过程的、region-grounded 的 belief update 框架**，并验证下面这个核心问题：

> **区域级证据与多步 belief update，能否比现有 post-hoc comparative verifier 带来更显著、也更稳定的性能提升？**

这份文档重点回答：

1. 这个方向的研究假设是什么
2. 最小可行版本应该怎么做
3. 三阶段实验分别验证什么
4. 应该如何设计主表、消融和停止条件

---

## 1. 背景与问题定义

当前方法线已经得到几个比较清楚的结论：

1. 强 frozen CLIP baseline 是性能基础
2. `scenario SLR-C` 是当前最有效、最稳定的主方法
3. comparative evidence verifier 能在 hard ambiguity 上提供帮助
4. 但现有 verifier 整体仍然更像 **post-hoc candidate correction**，而不是进入主干决策过程的 reasoning module

换句话说，当前 verifier 的工作方式仍然是：

- 先由 base proposer 给出相对固定的候选分数
- 再在候选上做后验式 evidence correction

这会带来一个限制：

> 证据读取并没有真正参与“形成 belief”的过程，而更像在结果上追加一层残差修正。

因此，`Region-Grounded Agentic Intent Reasoner` 的核心动机是：

- 不再把 object / activity / style / scene 证据仅仅当作后处理打分器
- 而是让系统围绕当前候选 belief，主动读取图像中的局部/全局证据，并逐步更新 intent belief

---

## 2. 核心研究假设

### Hypothesis 1
当前 post-hoc verifier 的局限之一在于：证据只在最终排序阶段介入，没有真正进入候选 belief 的形成过程。

### Hypothesis 2
对候选 intent 进行 **candidate-conditioned region grounding**，能比统一的后处理 verifier 更准确地读取局部证据，尤其是在 hard ambiguity 场景下。

### Hypothesis 3
如果系统能够根据当前 belief 状态，动态决定更该读取哪类证据（object / activity / scene / style），则会比“无差别读取所有证据”更有效，也更符合 agent-style reasoning 的定义。

### Hypothesis 4
若前两步成立，再引入轻量 graph reasoning，有可能进一步建模 candidate–region–evidence 的关系，从而提升 hardest ambiguity 的消歧能力。

---

## 3. 总体方法框架

建议将该方向统一命名为：

> **Region-Grounded Agentic Intent Reasoner**

总体流程可以写成：

```text
Image
  -> Frozen / partially trainable CLIP visual encoder
  -> Global image feature + patch tokens
  -> Initial intent belief (from SLR-C or a lighter proposer)
  -> Region-grounded evidence extraction
  -> Agent controller selects which evidence to read
  -> Belief update module updates candidate scores
  -> Repeat for T steps
  -> Final refined intent prediction
```

与当前 verifier 的最大不同在于：

- 当前 verifier：更像 post-hoc reranking / residual correction
- 新方法：把 evidence reading 和 belief update 直接纳入 candidate-level reasoning process

---

## 4. 方法设计原则

为了保证实验具有解释性，本方向必须遵循以下原则：

### Principle 1
**先固定 base proposer，再验证新 reasoning module。**

也就是：
- 第一阶段优先复用当前 strongest proposer：`scenario SLR-C`
- 不在一开始同时改 proposer 和 reasoner

### Principle 2
**先做 region-grounded belief update，再做 controller。**

也就是：
- 先验证“区域级证据 + belief update”本身是否有效
- 再验证“agent 决定读什么”是否有效

### Principle 3
**先用 CLIP patch tokens，不上 detector。**

因为：
- patch token 方案实现简单
- 更容易与当前 strongest baseline 对接
- 可以先回答“region grounding 是否值得做”，再考虑更复杂 region proposals

### Principle 4
**scene/style 初始阶段保留偏全局，object/activity 优先做局部。**

这是为了让第一版结构尽量合理、轻量且稳定：

- `object / activity`：更适合区域级建模
- `scene / style`：更适合作为全局上下文特征

---

## 5. 三阶段实验路线

## Phase 1：Region-Conditioned Belief Update（无真实 controller）

### 5.1 目标

验证：

> **区域级 evidence + candidate-conditioned belief update 本身，是否比当前 post-hoc verifier 更有潜力？**

这一阶段先不做真正的 expert 选择，避免一开始过于复杂。

### 5.2 输入表示

从 CLIP `ViT-L/14` 提取：

- 全局图像特征 `g`
- patch tokens `P = {p_1, ..., p_N}`

其中：
- `g` 用于全局 scene/style context
- `P` 用于 region-grounded evidence reading

### 5.3 初始 belief

直接复用当前 strongest proposer：

- `scenario SLR-C`

输出：

- 初始 candidate scores `s^(0)`
- top-k candidate intents

### 5.4 Region-grounded evidence extraction

第一版建议：

#### region-level
- object
- activity

#### image-level
- scene
- style

原因：
- object / activity 更依赖局部区域
- scene / style 更偏全图语义

### 5.5 Candidate-conditioned cross-attention

对于每个候选 intent `c`，使用其 intent embedding / text query 作为 query，对 patch tokens 做 cross-attention：

$$
h_c = CrossAttn(q_c, P)
$$

其中：
- `q_c` 是 candidate intent query
- `P` 是图像 patch tokens
- `h_c` 是 candidate-conditioned region summary

### 5.6 Belief update

将以下信息输入一个轻量 updater：

- 初始 candidate score `s_c^(0)`
- candidate-conditioned region summary `h_c`
- image-level scene evidence
- image-level style evidence

得到更新后的 candidate score：

$$
s_c^{(1)} = f(s_c^{(0)}, h_c, e^{scene}, e^{style})
$$

### 5.7 训练方式

初始建议：

- 冻结 CLIP visual encoder
- 可先冻结 text encoder
- 只训练：
  - candidate-conditioned cross-attention
  - belief updater

loss：
- 直接使用最终 multi-label classification loss

### 5.8 阶段对照组

至少比较：

1. `scenario SLR-C`
2. `SLR-C + v2 comparative + gate`
3. `Phase 1: region-conditioned belief update`

### 5.9 成功信号

如果 Phase 1 就能：
- 接近或超过 `v2 comparative + gate`
- 或至少在 `Hard / Macro` 上更有潜力

则说明该方向值得继续。

---

## Phase 2：加入多专家 Agent Controller

### 6.1 目标

验证：

> **不是所有 evidence 都应同时读取；由 agent controller 决定当前该读哪类证据，是否更有效？**

### 6.2 专家定义

建议只保留 4 个 expert：

1. `Region-Object Expert`
2. `Region-Activity Expert`
3. `Global-Scene Expert`
4. `Global-Style Expert`

每个 expert 输出一个 evidence summary：

$$
e_t^{obj}, e_t^{act}, e_t^{scene}, e_t^{style}
$$

### 6.3 轻量 controller

第一版不要做 RL，不要做复杂 policy learning。

controller 输入：

- 当前 belief state `b_t`
- candidate margin / entropy
- （可选）上一步已读取 evidence type

controller 输出：

- 当前 step 各 expert 的 routing weights
- 或一个最值得读取的 expert 类型

### 6.4 推荐先做 soft routing

推荐初始版本：

$$
\alpha_t = softmax(W b_t)
$$

然后 evidence 为加权和：

$$
e_t = \alpha_t^{obj} e_t^{obj} + \alpha_t^{act} e_t^{act} + \alpha_t^{scene} e_t^{scene} + \alpha_t^{style} e_t^{style}
$$

soft routing 的优点：
- 更稳定
- 更容易训练
- 更容易先验证 controller 的价值

### 6.5 多步 belief update

使用 `T = 2` 或 `3` 步：

1. controller 读取当前 belief
2. 决定应该重点读哪些 expert
3. belief updater 更新 scores

记为：

$$
b_{t+1} = Update(b_t, e_t)
$$

### 6.6 训练与正则化

loss 仍然用最终分类 loss。

建议增加 routing regularization，例如：
- entropy penalty
- encourage sparse routing

目的：
- 防止 controller 学成所有 expert 平均分配

### 6.7 阶段对照组

至少比较：

1. Phase 1（无 controller，全 evidence）
2. Phase 2 soft routing
3. （可选）Phase 2 hard routing

### 6.8 成功信号

如果 soft routing 比无 routing 更强，则可以支持：

> agent-style evidence selection 是有效的，而不是简单增加更多模块。

---

## Phase 3：加入轻量 Graph Reasoning

### 7.1 目标

验证：

> **candidate–region–evidence 关系建模，是否能进一步帮助 hard ambiguity 消歧？**

### 7.2 图结构

不要一开始构建大图。建议只做一个轻量三层图：

- candidate intent nodes
- selected region nodes
- expert / evidence-type nodes

### 7.3 Region selection

不要把所有 patch tokens 都放进图。

建议先做 top-m region selection：
- 依据 candidate-conditioned attention
- 或 patch saliency

建议：
- `m = 8 / 12 / 16`

### 7.4 Graph update

只做一层即可：
- graph attention
- 或简化 message passing

输出 refined candidate representations，再映射成 final candidate scores。

### 7.5 阶段对照组

比较：

1. Phase 2 best
2. Phase 2 + graph reasoning

### 7.6 成功信号

- 若 graph 有明显提升，说明空间/关系推理有效
- 若没有明显提升，也不影响主线，因为 Phase 2 已足以形成方法主体

---

## 8. 主实验表设计

## 8.1 主表

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` |  |  |  |  |  |
| `SLR-C + v2 comparative + gate` |  |  |  |  |  |
| Phase 1: region-conditioned updater |  |  |  |  |  |
| Phase 2: agentic soft routing |  |  |  |  |  |
| Phase 3: + graph reasoning |  |  |  |  |  |

### 主表目标
证明该方向是否能从：
- `post-hoc verifier`
转向
- `region-grounded belief-update framework`
并带来更显著的整体增益。

---

## 8.2 消融 1：区域证据是否有用

| Evidence Setup | Macro | Hard |
| --- | ---: | ---: |
| global only |  |  |
| region object/activity + global scene/style |  |  |
| all global |  |  |
| all region (optional) |  |  |

### 目标
验证 region-grounded object/activity 是否真有必要，而不是所有证据都做成全局即可。

---

## 8.3 消融 2：controller 是否有用

| Routing | Macro | Micro | Hard |
| --- | ---: | ---: | ---: |
| no routing (all evidence) |  |  |  |
| soft routing |  |  |  |
| hard routing |  |  |  |

### 目标
验证“agent 决定读什么”是否真的比统一读所有 evidence 更有效。

---

## 8.4 消融 3：graph 是否有用

| Reasoning | Macro | Hard |
| --- | ---: | ---: |
| updater only |  |  |
| updater + graph |  |  |

### 目标
判断 graph reasoning 是否值得进入主线，还是只作为加分探索。

---

## 8.5 消融 4：belief update 步数

| T steps | Macro | Hard |
| --- | ---: | ---: |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

### 目标
判断 iterative belief update 是否真正成立。

---

## 9. 实现与训练建议

## 9.1 初始训练策略

不要一开始端到端全训。

建议先冻结：
- CLIP visual encoder
- （可选）CLIP text encoder

只训练：
- candidate-conditioned attention
- belief updater
- controller
- graph layer（若有）

这样更稳，也更容易判断新模块本身的贡献。

## 9.2 Region 表示建议

不要一开始用 detector-based regions。

优先：
- patch tokens

原因：
- 实现简单
- 与 CLIP backbone 更兼容
- 适合作为第一版 region grounding 方案

## 9.3 scene/style 不要过度区域化

初始保持：
- object/activity：区域级
- scene/style：全局级

这样更符合视觉语义，也避免第一版模型过重。

## 9.4 controller 优先 soft routing

不要一开始做：
- hard routing
- RL
- 离散采样策略

这些都应放在后续，只在 soft routing 已经验证有效时再考虑。

---

## 10. 成功信号与停止条件

## 10.1 成功信号

如果这条线成立，通常应看到：

### Phase 1
至少接近甚至超过 `v2 comparative + gate`

### Phase 2
soft routing 带来更稳的 `Macro / Hard` 提升

### Phase 3
若 graph 进一步提升 hardest ambiguous classes，则可作为增强模块保留；若提升不明显，也不影响主线成立。

## 10.2 停止条件

如果出现以下情况，则不建议继续深入：

1. Phase 1 就明显不如 `v2 comparative + gate`
2. region grounding 带来大量复杂度，但 Hard / Macro 没有明确收益
3. controller 学成平均分配，没有形成有意义的专家选择
4. graph 增加复杂度却无法稳定提升 hardest cases

---

## 11. 风险与注意事项

## 风险 1：controller 学成平均路由

解决建议：
- 加 entropy penalty
- 控制 expert 数量
- 先做 soft routing，再考虑更强约束

## 风险 2：graph 太重、数据不够

解决建议：
- 只做一层 graph update
- region 数量控制在小范围

## 风险 3：region-level evidence 太噪

解决建议：
- 第一版只让 object/activity 做区域级
- scene/style 保持全局级

## 风险 4：复杂度提升大于收益

因此必须坚持三阶段推进：
- 先证 region-conditioned updater
- 再证 controller
- 最后才考虑 graph

---

## 12. 推荐推进顺序

### Round 1
先做 Phase 1：Region-Conditioned Belief Update

这是必须先验证的核心版本，也是成本最低的一步。

### Round 2
如果 Phase 1 有希望，再做 Phase 2：Soft Routing Controller

### Round 3
只有在 Phase 2 已经有效时，才做 Phase 3：Light Graph Reasoning

---

## 13. 一句话总结

这个实验方向本质上是在验证：

> **能否把候选提出后的局部歧义消解，从 post-hoc verifier 升级为一个 region-grounded、multi-expert、agentic belief-update process，从而获得更显著的性能提升。**
