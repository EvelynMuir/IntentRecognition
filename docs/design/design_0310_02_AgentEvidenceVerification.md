很好，这一步其实是把你现在 **prior + rerank 的成果升级为一个完整方法论文**。我会给你一个 **可以直接发展成论文 Method + Experiments 的草案**，尽量做到：

* 和你已有成果 **连续**（scenario prior / calibration / rerank 不浪费）
* 符合导师建议的 **agent / multi-expert reasoning**
* 方法 **不至于过重，能做得出来**
* 实验 **有明确证据链**

我会分成四部分：

1️⃣ 方法整体框架
2️⃣ 方法模块设计（每个模块具体怎么做）
3️⃣ 训练 / 推理流程
4️⃣ 实验计划（非常关键）

---

# 一、方法总体框架

论文可以命名为类似：

**Agentic Visual Intent Recognition via Multi-Expert Evidence Verification**

或稳一点：

**Multi-Expert Evidence Aggregation for Visual Intent Recognition**

---

## 核心思想

视觉意图不是单一视觉概念，而是由 **object / scene / style / human activity 等多种证据共同支持**。

现有方法通常：

```
image → feature → intent classifier
```

或者

```
image + text prior → similarity → intent
```

但缺乏：

**显式证据收集与验证过程。**

---

## 我们的方法：Intent Agent

整体流程：

```
Image
 ↓
Base VLM (CLIP)
 ↓
Intent hypotheses (top-k intents)

 ↓
Multi-Expert Perception
   ├ object expert
   ├ scene expert
   └ style expert

 ↓
Evidence Aggregation
 ↓
Hypothesis Verification
 ↓
Intent reranking
 ↓
Final intent prediction
```

核心思想：

> **先提出候选 intent，再通过多专家证据验证这些假设。**

这非常符合 agent 叙事：

* hypothesis generation
* evidence collection
* hypothesis verification

---

# 二、方法模块设计

下面是 Method section 的主体。

---

# 2.1 Base Intent Hypothesis Generation

首先使用 VLM 生成候选 intent。

使用：

**CLIP visual encoder**

得到图像特征：

[
v = f_v(I)
]

---

## Intent prior

使用你已经做好的三类 prior：

* lexical
* canonical
* scenario

我们最终使用 **scenario prior** 作为主 prior。

intent 文本 embedding：

[
t_i = f_t(p_i)
]

其中 (p_i) 是 scenario prompt。

---

## 初始 intent score

[
s_i = cosine(v, t_i)
]

---

## calibration

保留你已有方法：

[
s_i' = calibrate(s_i)
]

得到 calibrated score。

---

## top-k hypotheses

选取：

[
H = {i_1, i_2, ..., i_k}
]

作为 intent hypotheses。

---

# 2.2 Multi-Expert Perception

Agent 收集不同类型证据。

我们设计 **三个视觉专家**：

| expert        | 作用         |
| ------------- | ---------- |
| object expert | 识别关键物体     |
| scene expert  | 识别环境       |
| style expert  | 识别视觉风格 /氛围 |

这些专家 **可以直接使用 pretrained models**。

例如：

### object expert

例如：

```
Detic / GroundingDINO / YOLO
```

输出：

```
object labels + confidence
```

构成 object evidence：

[
E_o
]

---

### scene expert

例如：

```
Places365 classifier
```

输出：

[
E_s
]

---

### style expert

可以使用：

* CLIP aesthetic head
* style classifier
* aesthetic predictor

输出：

[
E_{st}
]

---

最终图像 evidence：

[
E = {E_o, E_s, E_{st}}
]

---

# 2.3 Intent Evidence Templates

关键创新点之一：

**每个 intent 对应一组 expected evidence。**

这些 evidence 来自：

**scenario prior**

例如：

```
Appreciating fine design
Scenario:
tourist looking at architecture
Gothic arches
historic building
```

可以解析出：

| evidence type | example                 |
| ------------- | ----------------------- |
| object        | building / architecture |
| scene         | historical site         |
| style         | aesthetic / artistic    |

形成 intent evidence template：

[
T_i = {T_i^o, T_i^s, T_i^{st}}
]

---

# 2.4 Evidence Matching

计算每个 expert evidence 与 intent template 的匹配度。

例如：

### object matching

[
m_o(i) = sim(E_o, T_i^o)
]

---

### scene matching

[
m_s(i) = sim(E_s, T_i^s)
]

---

### style matching

[
m_{st}(i) = sim(E_{st}, T_i^{st})
]

---

# 2.5 Evidence Aggregation

我们提出：

**Intent-Conditioned Evidence Aggregation**

最终 verification score：

[
S_i = s_i' + \sum_e w_e m_e(i)
]

其中：

* (s_i')：calibrated base score
* (m_e)：expert matching
* (w_e)：expert weight

可以有两种版本：

### version A（简单版）

固定权重：

[
S_i = s_i' + w_o m_o + w_s m_s + w_{st} m_{st}
]

---

### version B（更高级）

intent-conditioned routing：

[
w_e = g(i)
]

不同 intent 依赖不同 expert。

例如：

* scene-heavy intents
* object-heavy intents

---

# 2.6 Hypothesis Verification and Reranking

最终对候选 intents：

[
H = {i_1, i_2, ..., i_k}
]

计算 verification score：

[
S_i
]

重新排序：

```
rerank(H, S)
```

输出最终 intent。

---

# 三、训练 / 推理流程

为了简单：

**整个系统不需要复杂训练。**

可以采用：

### inference-time framework

步骤：

1️⃣ CLIP 计算 intent score
2️⃣ calibration
3️⃣ 生成 top-k intents
4️⃣ multi-expert evidence extraction
5️⃣ evidence matching
6️⃣ rerank

---

如果你愿意，也可以：

训练一个小 aggregation head：

[
S_i = f(s_i', m_o, m_s, m_{st})
]

但不是必须。

---

# 四、实验计划

这一部分非常关键。

如果实验设计不好，这篇论文就会变成“模块堆叠”。

所以实验必须围绕 **核心问题**：

> 多专家证据是否真的帮助 intent 识别？

---

# 4.1 Dataset

主实验：

**Intentonomy**

指标：

* Macro F1
* Micro F1
* mAP

---

# 4.2 Baselines

必须包括：

### 基础模型

```
CLIP zero-shot
```

---

### prior baseline

```
lexical prior
canonical prior
scenario prior
```

---

### 你的现有方法

```
scenario prior + calibration
scenario prior + calibration + rerank
```

---

### prior SOTA

例如：

```
Query2Label
LLM-prior methods
```

再加：

```
+ our verification
```

---

# 4.3 Ablation

必须有。

---

## 1 expert importance

```
baseline
+ object
+ scene
+ style
+ all experts
```

证明 **互补性**。

---

## 2 aggregation method

```
simple concat
sum score
intent-conditioned aggregation
```

证明方法设计不是 trivial。

---

## 3 prior comparison

```
lexical
canonical
scenario
```

验证你之前的发现。

---

## 4 calibration vs verification

```
baseline
+ calibration
+ verification
+ calibration + verification
```

说明两者作用不同。

---

# 4.4 Hard intent analysis

选取高混淆类别：

例如：

```
Enjoy life
Happy
Playful
```

展示：

* confusion matrix
* improvement

这是论文亮点。

---

# 4.5 Expert dependency analysis

展示：

不同 intent 依赖不同 expert。

例如：

| intent      | most useful expert |
| ----------- | ------------------ |
| Fine design | style              |
| Exploration | scene              |
| Fitness     | object             |

这会让方法 **更像 reasoning system**。

---

# 4.6 Case study

展示：

```
image
candidate intents
expert evidence
verification reasoning
final intent
```

非常适合 agent 叙事。

---

# 五、论文贡献可以写成

最终 contributions 可以写成：

1️⃣ We propose an **agentic visual intent recognition framework** that verifies intent hypotheses using multi-expert visual evidence.

2️⃣ We introduce **intent-conditioned evidence aggregation**, enabling different intents to rely on different perceptual experts.

3️⃣ We show that **scenario-level priors combined with evidence verification significantly improve intent recognition**, especially for ambiguous intent categories.

4️⃣ Extensive experiments demonstrate consistent improvements over prior-based baselines and prior SOTA.