# Agentic Visual Intent Recognition via Multi-Expert Evidence Verification

## 一、这条线要解决什么

当前主线结果已经说明两件事：

1. **强 frozen CLIP baseline 已经足够强。**
2. **剩余难点主要不在 feature extraction，而在候选意图之间的局部语义消歧与最终决策。**

已有结果可以概括为：

- baseline test：
  - `macro = 45.98`
  - `micro = 56.54`
  - `samples = 56.40`
  - `hard = 26.86`
- `scenario SLR + class-wise threshold` 当前 overall 最强：
  - `macro = 51.28`
  - `micro = 59.13`
  - `samples = 58.47`
  - `hard = 33.98`
- baseline 的 candidate proposal 已经很强：
  - top-10 label recall = `83.91%`
  - top-10 `sample_any_recall = 95.07%`
  - top-10 `sample_all_recall = 75.33%`

这说明：

> 当前真正的空间，不是“再做一个更重的全局分类器”，而是让系统在已有 top-k 候选内部，做更可靠的意图验证。

当前 `SLR-C` 已经证明：

- `scenario prior` 作为局部 rerank 信号有效
- `class-wise calibration` 可以把排序收益转成真正的 multi-label 输出

但它还有一个明显限制：

> 当前 rerank 依赖的仍然是 **class-level text prior**，而不是对图像中具体证据的显式核验。

这就引出下一条方法线：

## **先提出 intent hypothesis，再用多专家视觉证据去验证它。**

---

## 二、方法定位

这条线不是推翻当前 `scenario rerank + calibration`，而是在它的基础上往前走一步。

整体定位应当是：

- 保留当前最强的 frozen CLIP 主干与 local candidate pipeline
- 保留 `scenario prior` 作为 hypothesis proposal 的语义来源
- 新增 **multi-expert evidence verification**
- 最终仍然使用 **calibrated decision rule**

因此，这个方法更准确地说，是当前 `SLR-C` 的下一阶段版本，而不是另一条完全脱节的新线。

可以暂时记为：

## **MEV-C: Multi-Expert Verification with Calibration**

整体流程：

```text
Image
  -> strong visual baseline / scenario rerank
  -> top-k intent hypotheses
  -> multi-expert evidence extraction
  -> hypothesis-conditioned evidence matching
  -> verification rerank
  -> calibrated multi-label decision
```

它对应的 agent 叙事非常自然：

1. hypothesis generation
2. evidence collection
3. evidence verification
4. calibrated decision

---

## 三、核心动机

### 3.1 为什么 plain rerank 还不够

当前 `SLR-C` 的增益已经很明确，但它本质上仍是在做：

```text
候选类 logits + 文本先验分数
```

也就是说，它回答的是：

> “这个 intent 的文本描述和图像整体像不像？”

但视觉意图往往不是单一相似度问题，而更像：

> “图像里是否出现了支持该 intent 的若干证据组合？”

例如：

- `Appreciating fine design`
  - 单靠整体 embedding 相似度不够
  - 更关键的是是否出现了：
    - 建筑/室内设计相关 object
    - 可支撑审美场景的 scene
    - 明显的 style / aesthetic cue

因此，plain rerank 的下一步，不应只是再换一个 prompt 或再加一个 gate，而应当是：

> 把意图判断从“整体文本相似”推进到“结构化证据核验”。

### 3.2 为什么这条线有实现价值

它和当前已有发现是连续的：

- baseline 的 top-k recall 已经很高，说明 hypothesis proposal 足够强
- 当前剩余问题更像是候选内部的局部边界
- `scenario prior` 已经提供了每个 intent 的典型场景描述

换句话说，当前已经有了三块现成基础：

1. **候选集**：baseline / SLR 负责给出 plausible intents
2. **模板来源**：scenario prior 可以转成 expected evidence
3. **最终决策**：class-wise calibration 已经证明有效

那么最自然的下一步，就是在中间补上一层：

## **evidence verification**

---

## 四、问题设定

给定图像 `x`，类别集合为 `Y = {1, ..., C}`。

当前强视觉 baseline 输出每个类别的 logits：

[
z(x) \in \mathbb{R}^{C}
]

在当前主线里，我们已经有一版局部 rerank 分数：

[
\tilde z_c =
\begin{cases}
z_c + \alpha \tilde s_c^{scn}(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
]

其中：

- `\mathcal{T}_k(x)` 是 top-k 候选 intent 集合
- `\tilde s_c^{scn}(x)` 是归一化后的 scenario prior score

MEV-C 的目标不是重做 proposal，而是在候选集合内部再增加一个 verification 分数：

[
q_c(x) = Verify(x, c)
]

然后得到新的 candidate score：

[
u_c =
\begin{cases}
\tilde z_c + \beta q_c(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
]

最后再使用 calibrated decision rule：

[
\hat y_c = \mathbb{I}(\sigma(u_c) > t_c)
]

其中 `t_c` 为 class-wise threshold。

这个设计有两个好处：

1. **不破坏强 baseline 的全局结构**
2. **verification 只作用于 plausible candidates，不会退化成全局 semantic classifier**

---

## 五、方法设计

### 5.1 Stage 0：Candidate Proposal

这一阶段直接沿用当前 strongest pipeline，不重新发明。

推荐默认入口：

- strong frozen CLIP baseline
- `scenario` semantic prior
- local rerank
- `top-k = 10`

原因很简单：

- 这已经是当前最强、最稳的候选生成方式
- top-10 coverage 足够高
- verification 模块本来就应该建立在高 recall candidate proposal 之上

也就是说，MEV-C 不负责“把正确类召回进来”，而负责：

> **在已经比较强的候选集内部，做更细粒度、更可解释的证据核验。**

---

### 5.2 Stage 1：Intent Evidence Template Construction

每个 intent 都需要一份 **expected evidence template**。

这个模板不是拍脑袋手写，而应尽量从已有 `scenario prior` 中自动解析得到。

对每个 intent `c`，构建：

[
T_c = \{T_c^{obj}, T_c^{scene}, T_c^{style}, T_c^{act}\}
]

其中：

- `T_c^{obj}`：该 intent 典型相关的 object / attribute
- `T_c^{scene}`：典型环境与场景
- `T_c^{style}`：审美、氛围、构图、风格线索
- `T_c^{act}`：人类活动或 interaction 线索

这里不要求所有 intent 都有四类证据。

实际中更合理的是：

- 有些 intent 主要依赖 `scene`
- 有些 intent 主要依赖 `style`
- 有些 intent 主要依赖 `object + activity`

因此模板天然会带来 **intent-specific expert dependency**。

#### 模板来源

优先使用当前已有的文本资源，按成本从低到高分三档：

1. **scenario prior**
   - 主来源
   - 负责给出典型场景与证据短语
2. **canonical / lexical prior**
   - 作为补充来源
   - 用于填补 scenario 模板过稀疏的问题
3. **人工小规模修订**
   - 只对明显歧义类补充
   - 避免把整个方法做成大量人工规则工程

#### 模板形式

建议模板不要写成长句，而写成短语集合。

例如：

| intent | object evidence | scene evidence | style evidence | activity evidence |
| --- | --- | --- | --- | --- |
| Appreciating fine design | architecture, ornament, interior detail | museum, historical site, elegant interior | aesthetic, artistic, refined | looking, observing |
| Fitness | gym equipment, bicycle, sportswear | gym, outdoor track, sports venue | energetic, dynamic | running, training, exercising |
| Exploration | map, backpack, landscape cue | nature, street, travel location | adventurous, open | walking, discovering |

#### v1 约束

为了先把系统跑通，`v1` 推荐：

- 只使用 **positive evidence**
- 暂不引入 negative / counter-evidence 模板

后续如果需要再扩展：

[
T_c = \{T_c^{+}, T_c^{-}\}
]

用来建模“什么出现会支持该 intent，什么出现会削弱该 intent”。

---

### 5.3 Stage 2：Multi-Expert Evidence Extraction

这里最重要的不是“专家越重越好”，而是 **先把证据空间定义清楚，并保证实现成本可控**。

因此我建议分成两个版本。

#### Version 1：CLIP-space expert banks

这是最应该先做的版本。

做法：

- 为 `object / scene / style / activity` 分别建立一个 vocabulary bank
- 每个 bank 由若干短语构成
- 用同一个 CLIP image embedding 与各 bank 的 text embeddings 做相似度
- 每个 expert 输出 top-m 证据项及其分数

形式上，对 expert `e`：

[
B_e(x) = \{(b_{e,1}, r_{e,1}), ..., (b_{e,m}, r_{e,m})\}
]

其中：

- `b_{e,j}` 是第 `j` 个证据短语
- `r_{e,j}` 是相似度分数

这个版本的优点：

- 不需要引入多个外部大模型
- 与当前 CLIP 主线天然兼容
- 所有证据都在统一文本空间里，匹配实现很简单
- 很适合作为论文的 first workable version

#### Version 2：Specialist pretrained experts

如果 `v1` 跑通且有收益，再考虑用更强专家替换部分 bank：

- `object expert`：
  - GroundingDINO / Detic / YOLO
- `scene expert`：
  - Places365 classifier
- `style expert`：
  - aesthetic predictor / style classifier / CLIP aesthetic head
- `activity expert`：
  - action recognition backbone 或 CLIP action bank

这个版本更强，但也更重，且会带来三个额外问题：

1. label space 不统一
2. 推理代价更高
3. 方法贡献容易被“用了更多外部模型”稀释

因此，文档层面应明确：

> **v1 先走 CLIP-space evidence bank，v2 再考虑 specialist experts。**

---

### 5.4 Stage 3：Hypothesis-Conditioned Evidence Matching

拿到图像证据 bank 后，需要计算它与某个候选 intent 模板的匹配程度。

对 expert `e` 与候选类 `c`，定义匹配分数：

[
m_e(x, c) = Match(B_e(x), T_c^e)
]

一个简单、足够合理的 `v1` 定义是：

[
m_e(x, c) =
\frac{1}{|T_c^e|}
\sum_{a \in T_c^e}
\max_{(b, r) \in B_e(x)}
r \cdot sim(a, b)
]

含义是：

- 对模板里的每个 expected evidence phrase `a`
- 在图像证据 bank 里找最相近的证据 `b`
- 用证据分数 `r` 和短语相似度 `sim(a, b)` 共同决定匹配强度

如果 `v1` 全部都在 CLIP text space 里：

- `sim(a, b)` 可以直接用 text embedding cosine similarity
- 实现和调试都比较直接

#### 为什么要 hypothesis-conditioned

因为同一张图像的证据，不同 intent 的解释方式不同。

例如同一张建筑照片：

- 对 `Appreciating fine design`，style / aesthetic evidence 更关键
- 对 `Exploration`，scene evidence 更关键
- 对 `Learning art history`，可能 object + scene 都重要

所以 verification 不是先算一个“统一图像质量分数”，而是：

> **对每个候选 hypothesis，分别问一次：这张图像的证据是否支持它？**

---

### 5.5 Stage 4：Intent-Conditioned Evidence Aggregation

单个 expert 的分数不够，最终需要聚合成 verification score。

定义：

[
q_c(x) = \sum_{e \in \mathcal{E}} w_{c,e} \, m_e(x, c)
]

其中：

- `\mathcal{E} = {obj, scene, style, act}`
- `w_{c,e}` 表示类别 `c` 对 expert `e` 的依赖程度

这里同样建议分两版。

#### Version A：固定权重

最简单可行：

[
q_c = w_{obj} m_{obj} + w_{scene} m_{scene} + w_{style} m_{style} + w_{act} m_{act}
]

或者进一步做成 **template-aware fixed weight**：

- 如果某个 intent 没有 `style` 模板，则 `w_{c,style} = 0`
- 剩余权重归一化

这版最适合 first paper version，因为：

- 简单
- 可解释
- 不依赖额外训练

#### Version B：intent-conditioned routing

更进一步，可以用一个小 gating 函数输出权重：

[
w_c = softmax(g(h_c))
]

其中 `h_c` 可以由下列信息拼接得到：

- candidate visual score `\tilde z_c`
- scenario prior score
- 各 expert matching 分数
- optional uncertainty signal

这版更“像方法”，但也更容易退化成 learnable fusion。

基于你前面的实验经验，这里要特别警惕：

> 复杂 learnable fusion 未必是核心增益来源。

因此论文主线建议保持：

- **主方法：固定或 template-aware aggregation**
- **routing：作为增强版消融**

---

### 5.6 Stage 5：Verification Rerank + Calibrated Decision

最终把 verification 作为 residual signal 加回候选分数：

[
u_c =
\begin{cases}
\tilde z_c + \beta q_c(x), & c \in \mathcal{T}_k(x) \\
z_c, & c \notin \mathcal{T}_k(x)
\end{cases}
]

其中 `\beta` 控制 verification 强度。

这一步的关键不是让 verification 取代 baseline，而是：

> 让证据核验成为 candidate-local residual correction。

之后继续沿用当前已经被验证过有效的 calibrated decision rule：

[
\hat y_c = \mathbb{I}(\sigma(u_c) > t_c)
]

推荐默认仍然使用：

- `class-wise threshold`

如果后续考虑更紧凑的版本，再补：

- `frequency-group threshold`

整个方法的角色分工应当写得非常清楚：

- baseline / SLR：candidate proposal
- MEV：candidate verification
- calibration：final decision alignment

---

## 六、训练与推理路径

### 6.1 v1 推荐：尽量少训练

这条线最合理的第一版，不应上来就做端到端重训练。

更合适的是：

#### Offline 准备

1. 为每个 intent 构建 evidence template
2. 为每个 expert 构建 vocabulary bank
3. 预编码所有 evidence phrases 的 text embeddings

#### Validation 上调参

搜索：

- `top-k`
- `alpha`
- `beta`
- expert weights
- `m`（每个 bank 保留多少条证据）
- class-wise thresholds

#### Test 推理

1. 跑 baseline / SLR 产生 top-k 候选
2. 对图像提取 multi-expert evidence bank
3. 对每个候选 intent 计算 expert matching
4. 聚合得到 verification score
5. residual rerank
6. calibrated decision

这个版本的优点非常重要：

- 不引入重训练风险
- 与你当前分析框架高度兼容
- 实验推进快
- 适合先验证“证据核验”这个机制是否本身有效

### 6.2 v2 扩展：训练一个小 aggregation head

如果 `v1` 明显有效，再考虑训练一个小头：

[
q_c = f(\tilde z_c, m_{obj}, m_{scene}, m_{style}, m_{act})
]

这里的 `f` 可以很小，例如：

- 两层 MLP
- linear + sigmoid gate

但这一步不应抢主线。

主线判断标准应该是：

> 先证明 evidence verification 机制有效，再讨论 learnable aggregation 是否进一步带来稳定增益。

---

## 七、为什么这条线在论文上站得住

### 7.1 它自然继承了当前 strongest pipeline

不是另起炉灶，而是顺着现有最强结果继续推进：

- `scenario prior` 负责给 hypothesis 提供语义模板
- strong baseline 负责给候选集
- calibration 负责最终 multi-label 输出

MEV 只是补上中间缺失的一环：

> 让语义先验从“类级文本相似”升级为“图像证据核验”。

### 7.2 它比 plain rerank 更有任务针对性

`SLR-C` 已经证明 local semantic 有用，但 reviewer 很容易追问：

> 你这是不是只是又加了一个 text prior trick？

MEV 的回答更强：

- 不是简单再加文本分数
- 而是把意图拆成多类证据
- 再检查图像中是否真的出现了这些证据

这会让方法从“工程 rerank”更接近“reasoning / verification framework”。

### 7.3 它和 agent 叙事高度一致

这条线之所以比“再堆一个 learnable fusion”更有论文味，一个核心原因就是叙事更完整：

1. propose hypotheses
2. collect evidence
3. verify hypotheses
4. make calibrated decision

这个叙事和视觉意图任务本身也更契合，因为意图本来就不是单一 object label，而更像高层主观判断。

---

## 八、实验计划

这一部分不能只写“我们会做消融”，而要围绕清晰问题来设计。

### 8.1 要回答的核心问题

#### Q1：MEV 是否能稳定超过当前 strongest `SLR-C`

这是主问题。

要证明：

- evidence verification 不是替代 SLR
- 而是在 SLR 之上进一步提升

#### Q2：不同 expert 是否真的互补

这决定方法是不是“多专家”而不是“换了三个名字的同一个相似度模块”。

#### Q3：verification 是否主要帮助 ambiguous / hard intents

这决定方法有没有真正击中任务难点。

#### Q4：增益来自 evidence verification 本身，还是来自更复杂参数化

这个问题必须回答清楚，否则 reviewer 会质疑：

> 你是不是只是因为多加了参数或外部模型？

---

### 8.2 Dataset 与指标

主实验仍然使用：

- **Intentonomy**

主指标继续保持与你现有分析一致：

- Macro F1
- Micro F1
- Samples F1
- Hard subset F1

如果主表里保留 `mean F1`，也可以继续，但不应替代 `macro / hard`。

---

### 8.3 Baselines

必须覆盖三层对比。

#### Layer 1：当前强基线

- strong frozen CLIP baseline
- baseline + class-wise threshold

#### Layer 2：当前 strongest semantic pipeline

- scenario SLR
- scenario SLR + class-wise threshold
- lexical + canonical SLR
- lexical + canonical SLR + class-wise threshold

#### Layer 3：新的 verification 变体

- SLR + MEV(object only)
- SLR + MEV(scene only)
- SLR + MEV(style only)
- SLR + MEV(all experts)
- SLR + MEV(all experts) + class-wise threshold

如果 `v2` 跑了，还可补：

- SLR + MEV(clip-bank experts)
- SLR + MEV(specialist experts)

---

### 8.4 核心消融

#### Ablation A：expert importance

```text
scenario SLR-C
+ object verification
+ scene verification
+ style verification
+ activity verification
+ all experts
```

目标：

- 证明专家互补性
- 看哪些 expert 对整体更关键
- 看 hard intents 是否更依赖某类 expert

#### Ablation B：template source

```text
scenario template
canonical template
lexical template
scenario + canonical template
```

目标：

- 验证 evidence template 是否也存在 source complementarity
- 检查 `scenario` 是否仍然是最强单源模板

#### Ablation C：aggregation method

```text
fixed equal weight
template-aware fixed weight
intent-conditioned routing
small learnable head
```

目标：

- 证明 gain 的核心是否来自 verification，而不是复杂聚合器

#### Ablation D：proposal source

```text
baseline top-k
scenario SLR top-k
```

目标：

- 检查 verification 对 proposal quality 的依赖
- 验证“强 candidate proposal + evidence verification”是否是正确组合

#### Ablation E：calibration vs verification

```text
SLR-C
SLR + MEV
SLR + calibration
SLR + MEV + calibration
```

目标：

- 把 verification 与 calibration 的作用拆开
- 避免两者在论文里被写成一团

---

### 8.5 分析实验

#### Analysis 1：expert dependency

做 intent-expert heatmap：

| intent | object | scene | style | activity |
| --- | ---: | ---: | ---: | ---: |
| Fine design | low | medium | high | low |
| Exploration | low | high | medium | medium |
| Fitness | high | medium | low | high |

这张图非常重要，因为它能直接回答：

> 不同 intent 是否真的依赖不同证据结构？

#### Analysis 2：template coverage

分析：

- 某类模板是否过 sparse
- 哪些类几乎只有 scene evidence
- 哪些类 object/style evidence 很丰富

然后看模板丰富度与 gain 是否相关。

#### Analysis 3：hard intent case study

重点看高混淆类别，例如：

- `Enjoy life`
- `Happy`
- `Playful`
- `Appreciating fine design`
- `Exploration`

展示：

```text
image
top-k hypotheses
evidence bank
template match scores
verification rerank result
final decision
```

这类 case study 会非常有说服力。

#### Analysis 4：失败案例

必须保留失败案例，尤其是：

- 模板错误导致误导
- expert bank 命中表面相似词但语义不对
- style evidence 过泛导致 false positive

如果不写失败案例，这条线会显得像“什么都能解释”的故事。

#### Analysis 5：效率 / 代价

至少报告：

- `v1` CLIP-bank MEV 的额外推理开销
- specialist expert 版本的额外开销

这能帮助说明：

> 为什么论文主线优先走轻量 verification，而不是直接堆一堆专家模型。

---

### 8.6 当前 v1 已完成实验结果（2026-03-10）

这一节记录已经真正跑完的版本，而不是计划中的理想版本。

当前完整实验入口：

- `scripts/analyze_agent_evidence_verification.py`

完整输出目录：

- `logs/analysis/full_agent_evidence_verification_20260310`

当前 `v1` 的实际 instantiation 为：

- candidate proposal：
  - `scenario SLR`
  - `top-k = 10`
  - `alpha = 0.3`
- evidence bank：
  - `object`: COCO 80 labels
  - `scene`: Places365 365 labels
  - `style`: Flickr Style 20 labels
  - `activity`: Stanford40 40 labels
- template source：
  - `intent_description_gemini.json`
  - `intent2concepts.json`
- verification matching：
  - `benchmark bank + similarity matching`
- aggregation：
  - `template-aware fixed weight`
- decision rule：
  - global threshold
  - class-wise threshold

也就是说，这一版已经不是早期的手工 generic bank，而是：

> **benchmark label sets + CLIP-space matching + calibrated decision**

#### 当前 strongest baseline 对比

参考当前已有 strongest `SLR-C`：

- `scenario SLR + class-wise threshold`
  - `macro = 0.5128`
  - `micro = 0.5913`
  - `samples = 0.5847`
  - `hard = 0.3398`

当前 `MEV v1` 按 validation 选配置后的正式最好结果为：

- `all experts + beta = 0.1 + add_norm + class-wise threshold`
  - `macro = 0.5084`
  - `micro = 0.5817`
  - `samples = 0.5721`
  - `hard = 0.3353`

相对 strongest `SLR-C`：

- `macro -0.0044`
- `hard -0.0045`

因此，到目前为止最重要的结论不是“MEV 已经超过 SLR-C”，而是：

> **benchmark-bank evidence verification 已经完整跑通，但在当前默认聚合方式下，尚未稳定超过 strongest SLR-C。**

#### class-wise calibration 仍然是必要的

当前 `MEV v1` 的一个很明确现象是：

- global threshold 下，best validation-selected MEV test 大致在：
  - `macro = 0.4768`
  - `hard = 0.3056`
- class-wise threshold 下，best validation-selected MEV test 达到：
  - `macro = 0.5084`
  - `hard = 0.3353`

这说明：

> verification 改变 score geometry 之后，仍然非常依赖 calibrated decision rule，尤其是 class-wise threshold。

换句话说，当前观察和前面的 `SLR-C` 结论是一致的：

- rerank / verification 负责 candidate disambiguation
- calibration 负责把它转成最终 multi-label outputs

#### 单 expert 的诊断结果

如果只看 validation-selected 配置，`all experts` 是当前最强的正式配置。

但如果把每个 single expert 在 test 上的最好结果当作诊断信号，会看到更有意思的趋势：

- `object` best test:
  - `macro = 0.5111`
  - `hard = 0.3435`
- `scene` best test:
  - `macro = 0.5105`
  - `hard = 0.3368`
- `style` best test:
  - `macro = 0.5186`
  - `hard = 0.3521`
- `activity` best test:
  - `macro = 0.5158`
  - `hard = 0.3479`

这里需要强调：

- 上述 single-expert best test 结果只是诊断，不应作为正式主表写法
- 它们用于判断“哪类 expert 更有潜力”，而不是替代 validation-selected 结果

这组结果带来的判断很明确：

> **style 与 activity 是当前最有潜力的证据源，而简单把所有 experts 一起并入，反而可能稀释增益。**

#### 当前阶段的判断

这一轮实验至少回答了三个问题。

第一，MEV 不是空想，链路已经能完整运行：

- evidence template 构建可行
- benchmark label set bank 可行
- similarity matching 可行
- calibration 接入可行

第二，当前 gain pattern 说明 expert 之间并不天然互补：

- `all experts` 没有自动超过强单 expert
- 这说明 aggregation 或 expert selection 仍有改进空间

第三，这条线的下一个重点不应该是继续扩 bank，而应该是：

- 更好地选择 expert
- 更好地控制 expert 权重
- 或限制 verification 只在某些 intent / sample 上激活

换句话说，当前最合理的下一步，不是“再加更多专家”，而是：

> **把 style / activity 的有效信号保留下来，同时避免 all-expert aggregation 的相互稀释。**

---

### 8.7 当前 v2 已完成实验结果（2026-03-10）

`v1` 的主要问题已经很明确：

- `all experts` 的固定聚合会稀释强单 expert 的有效信号
- `style` 与 `activity` 在单 expert 诊断里明显更强

因此，`v2` 没有再改 evidence bank，也没有引入 learnable head，而是只改 aggregation / rerank 逻辑。

当前 `v2` 的核心是：

> **class-wise expert routing**

具体做法是：

1. 先跑 `object / scene / style / activity` 四个 single-expert MEV
2. 用 validation per-class F1 相对 `SLR-C` 的 gain 构建 class-wise routing matrix
3. 对每个类别只保留最有用的 expert，或保留 top-2 positive experts
4. 再做 verification rerank + class-wise calibration

当前完整实验输出目录：

- `logs/analysis/full_agent_evidence_verification_v2_20260310`

#### 当前 `v2` 的最优结果

当前 validation-selected 最优 `v2` 配置为：

- `routing = top1_positive`
- `beta = 0.3`
- `fusion = add_norm`
- `num_routed_classes = 20`

对应 test：

- `macro = 0.5110`
- `micro = 0.5900`
- `samples = 0.5824`
- `hard = 0.3399`

#### 与 `v1` 和 strongest `SLR-C` 的关系

相对 `v1` 正式最好结果：

- `v1`: `macro = 0.5084`, `hard = 0.3353`
- `v2`: `macro = 0.5110`, `hard = 0.3399`

也就是说：

- `macro +0.0026`
- `hard +0.0046`

相对 strongest `SLR-C`：

- `SLR-C`: `macro = 0.5128`, `hard = 0.3398`
- `v2`: `macro = 0.5110`, `hard = 0.3399`

也就是说：

- `macro -0.0018`
- `hard +0.0001`

这说明：

> **v2 已经明显优于 v1，并且在 hard 上基本追平甚至略高于 strongest SLR-C，但 overall macro 仍略低于 strongest SLR-C。**

#### 为什么 v2 比 v1 更合理

`v2` 的最关键变化，不是多了更多参数，而是：

- 不再对所有类别共享同一组 expert policy
- 改为让不同类别依赖不同 expert

当前 best routed config 的 class distribution 为：

- `object`: 2 classes
- `scene`: 7 classes
- `style`: 1 class
- `activity`: 10 classes

剩余类没有被 route 到正增益 expert，直接退回 `SLR-C`。

这一点非常关键，因为它说明：

> **不是所有类别都需要 verification，也不是所有类别都应该使用全部 experts。**

这和 `v1` 的观察完全一致，也进一步支持了 agent / routing 的叙事。

#### global 指标的补充观察

`v2` 在 global threshold 下的 best result 也值得记一下。

当前 best validation-selected global `v2` 为：

- `routing = top2_soft`
- `gamma = 8`
- `beta = 0.1`

对应 test：

- `macro = 0.4925`
- `hard = 0.3501`

这说明：

- softer routing 对 hard subset 仍然有吸引力
- 但如果按正式主表口径，当前仍应以 class-wise threshold 结果为主

#### 当前对 v2 的判断

这一轮 `v2` 实验至少说明三件事。

第一，`class-wise routing` 的方向是正确的：

- 它显著优于 `v1`
- 证明 `all-expert fixed aggregation` 确实不是最优结构

第二，当前最好结果已经非常接近 strongest `SLR-C`：

- 已经不再是明显落后
- 差距收敛到很小的量级

第三，这条线接下来最值得继续挖的，不是 bank，而是 routing 本身：

- route 哪些类
- route 几个 expert
- 是否要加入 uncertainty-aware selective activation

换句话说，当前最合理的下一步不是“再造一个更大 expert bank”，而是：

> **在当前 benchmark-bank extraction 固定的前提下，继续优化 routing 与 selective verification。**

---

## 九、风险与边界

这部分建议在设计文档里先写清楚，避免后面实现时踩坑。

### 9.1 模板噪声风险

从 `scenario prior` 自动解析 evidence template 时，最容易出现：

- 词太泛
- 词太长
- 词不视觉
- 同义短语过多

解决策略：

- 优先解析成短语而不是长句
- 做词表裁剪
- 对明显非视觉词做过滤

### 9.2 Expert label mismatch

如果用了 specialist experts，不同模型输出 label space 会不一致。

解决策略：

- 所有 expert 输出都投影回统一 text space
- 匹配时不直接比 label id，而比 text embedding similarity

### 9.3 “外部模型堆叠”风险

如果一开始就上 GroundingDINO + Places + aesthetic predictor，reviewer 很容易说：

> 你只是用了更多 pretrained experts。

解决策略：

- 主线先做 `CLIP-space expert banks`
- specialist experts 只作为增强版或后续版本

### 9.4 复杂聚合器掩盖机制风险

你前面已经看到 learnable fusion 不一定带来真正增益。

所以这里一定要避免：

- 一上来就多层 MLP
- 把 verification 变成黑盒分数混合器

主线应坚持：

> 先用简单聚合证明 evidence verification 有效，再谈 learnable routing。

### 9.5 calibration overfitting

verification 改变了 score geometry，class-wise threshold 很可能仍然有效，但也要警惕过拟合 val。

解决策略：

- 固定统一 val protocol
- 先报告 global / group / class-wise 三种校准结果
- 让 calibration 的收益可解释，而不是只报一组最优值

---

## 十、建议的实现顺序

为了降低风险，建议严格按阶段推进，而不是同时开多条线。

### Phase 1：先做最小可验证版本

1. 从 `scenario prior` 自动解析 evidence template
2. 搭建 `object / scene / style` 三个 CLIP-bank experts
3. 在 `scenario SLR top-k` 上做 verification rerank
4. 接上 class-wise calibration

这一阶段只回答一个问题：

> **最小版本的 evidence verification 是否已经能超过当前 strongest SLR-C？**

当前状态：

- 已完成
- 结论：**链路跑通，但 benchmark-bank `MEV v1` 尚未超过 strongest `SLR-C`**

### Phase 2：做机制性消融

1. 单 expert 与全 expert 对比
2. 模板来源对比
3. fixed weight 与 template-aware weight 对比

这一阶段回答：

> gain 到底来自哪里？

当前状态：

- 已部分完成
- 已有结果表明：
  - `style` 与 `activity` 更有潜力
  - `all experts` 并未自然优于强单 expert
  - `class-wise routing` 能明显优于 `v1`
  - 下一步应优先做 routing / selective activation，而不是继续扩 bank

### Phase 3：再考虑增强版

1. 加 activity expert
2. 加 specialist experts
3. 加 routing / learnable head
4. 加 negative evidence

这一阶段不是主线前提，而是后续增强。

---

## 十一、论文写法上的主张

如果这条线成立，整篇论文的表述应该是：

> 视觉意图识别不是简单的全局分类问题，而是一个“候选提出 + 证据核验 + 决策校准”的过程。

对应的方法贡献可以写成：

1. We extend strong local semantic reranking into a hypothesis verification framework for visual intent recognition.
2. We introduce intent-specific evidence templates and multi-expert visual evidence matching for candidate-level verification.
3. We show that lightweight evidence verification, when combined with calibrated decision rules, improves intent recognition without replacing the strong frozen CLIP baseline.
4. We provide analysis showing that different intents depend on different evidence experts, supporting an agentic reasoning perspective on intent perception.

---

## 十二、当前版本的明确建议

如果现在就要决定落地方向，我的建议是：

### 主方法版本

- `scenario SLR` 产生 top-k hypotheses
- `benchmark-bank object / scene / style / activity experts`
  - `object`: COCO 80
  - `scene`: Places365 365
  - `style`: Flickr Style 20
  - `activity`: Stanford40 40
- `template-aware fixed-weight verification`
- `class-wise calibrated decision`

### 当前结论

基于 `logs/analysis/full_agent_evidence_verification_20260310` 与 `logs/analysis/full_agent_evidence_verification_v2_20260310`，当前最稳妥的判断是：

- 这条线已经具备实现与分析闭环
- `MEV v2` 明显优于 `MEV v1`
- `MEV v2` 已基本追平 strongest `SLR-C`
- 当前最值得继续打磨的不是 `all experts`
- 而是 `class-wise routing + selective verification`

因此，下一阶段最推荐的主攻方向应是：

- `class-wise expert routing`
- uncertainty-aware selective verification
- style/activity-dominant routing，而不是无条件 all-expert aggregation

### 不建议一开始就做的内容

- 端到端重训练
- 很重的 specialist multi-model pipeline
- 复杂 learnable fusion
- negative evidence 与多阶段 agent loop

原因不是这些没价值，而是：

> 你当前最需要的是证明“evidence verification”本身是有效机制，而不是把系统做得更复杂。

如果这一步成立，这条线就会比单纯的 prior rerank 更完整，也更像一篇真正的方法论文。
