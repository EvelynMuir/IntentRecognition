# Candidate-to-Evidence Verification for Intent Recognition

> 先由强基线提出候选意图，再从图像中识别显式视觉元素，并从训练数据中学习意图与元素的对应关系，最后基于已识别元素对候选意图进行显式证据验证与重排序。

---

# 1. 方法动机

现有的视觉意图识别通常直接将图像映射到意图标签，或者借助文本先验对意图做语义匹配。但这类方法缺少一个显式的“验证”过程：模型虽然能够提出一组高分候选意图，却未必真正检查图像中是否存在支持这些意图的可见证据。

一种自然思路是为每个意图构造证据模板，再用模板匹配图像证据。但模板若依赖 LLM 或手工定义，往往与数据集中的真实判别模式不一致。为此，我们提出从训练数据中直接学习 **intent–element relation**：
先将图像解析为 object、scene、style、activity 等可识别视觉元素，再统计或学习这些元素与意图之间的支持关系。测试时，系统不再依赖生成式模板，而是根据图像中实际识别到的元素，对候选意图做显式 evidence verification。

---

# 2. 整体框架

整体流程分为四步：

### Step 1. Candidate proposal

使用一个强视觉基线模型为所有意图类别打分，并选取 top-k 候选意图。

### Step 2. Element extraction

将图像映射到多个标准元素空间，识别 object / scene / style / activity 等显式视觉元素。

### Step 3. Intent–element relation learning

从训练集统计或学习每个意图与各类元素之间的关联强度，形成 data-driven evidence profile。

### Step 4. Evidence verification and reranking

在测试时，根据图像中激活的元素及其与候选意图的关联强度，计算验证分数，并作为残差项加回候选意图原始分数，完成局部重排序。

---

# 3. 符号定义

设训练集为：

[
\mathcal{D}={(x_n, y_n)}_{n=1}^N
]

其中 (x_n) 表示图像，(y_n \in {0,1}^C) 是多标签意图标注，(C) 为意图类别数。

对于一张图像 (x)，强基线输出所有意图类别的原始分数：

[
s^{base}(x) \in \mathbb{R}^C
]

取其中 top-k 候选意图集合：

[
\mathcal{H}(x)={c_1,\dots,c_k}
]

---

# 4. Candidate Proposal

这一部分保留现有的强基线。

我们采用冻结的 CLIP 视觉编码器提取图像表征：

[
v = f_v(x)
]

(cls + mean(patch) -> MLP，ckpt_path `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`)

同时，为每个意图类别构造文本先验（例如 lexical / canonical / scenario 等形式），得到文本嵌入：

[
t_c = f_t(p_c)
]

图像与意图类别的初始相似度为：

[
s_c^{base} = \mathrm{sim}(v, t_c)
]

进一步接入现有的 calibration 和类特异阈值方案，在这个强基线基础上，只对 top-k 候选意图做显式 evidence verification。

---

# 5. Element Extraction

我们不直接从图像特征端到端学习意图，而是先将图像解析到多个标准化元素空间中。设共有四类元素专家：

* object expert
* scene expert
* style expert
* activity expert

对应元素集合分别记为：

[
\mathcal{Z}^{obj},\ \mathcal{Z}^{scene},\ \mathcal{Z}^{style},\ \mathcal{Z}^{act}
]

例如：

* object: COCO 80 类
* scene: Places365 365 类
* style: Flickr Style 20 类
* activity: Stanford40 40 类

对于图像 (x)，每个专家输出该图像在相应元素空间中的响应分数：

[
e^{obj}(x)\in\mathbb{R}^{|\mathcal{Z}^{obj}|}
]
[
e^{scene}(x)\in\mathbb{R}^{|\mathcal{Z}^{scene}|}
]
[
e^{style}(x)\in\mathbb{R}^{|\mathcal{Z}^{style}|}
]
[
e^{act}(x)\in\mathbb{R}^{|\mathcal{Z}^{act}|}
]

最终拼接得到总的元素响应向量：

[
e(x)=[e^{obj}(x);e^{scene}(x);e^{style}(x);e^{act}(x)]
]

为了便于显式验证，我们进一步从每类专家中保留 top-m 个高响应元素，记为图像的激活元素集合：

[
\mathcal{A}(x)=\mathcal{A}^{obj}(x)\cup\mathcal{A}^{scene}(x)\cup\mathcal{A}^{style}(x)\cup\mathcal{A}^{act}(x)
]

这里每个激活元素 (z \in \mathcal{A}(x)) 伴随一个响应强度 (w_z(x))。

---

# 6. Intent–Element Relation Learning

这是核心模块。

目标是从训练数据中学习：

[
R(c,z)
]

表示元素 (z) 对意图 (c) 的支持强度。直观上，如果某个元素在带有意图 (c) 的图像中显著更常见或更强响应，那么它就应该成为该意图的支持证据。

---

## 6.1 最简单的统计式关系学习

对每个意图 (c) 和元素 (z)，定义正负样本条件下的平均响应：

[
\mu_{c,z}^{+} = \mathbb{E}[e_z(x)\mid y_c=1]
]
[
\mu_{c,z}^{-} = \mathbb{E}[e_z(x)\mid y_c=0]
]

则 intent–element relation 可以定义为：

[
R(c,z)=\mu_{c,z}^{+}-\mu_{c,z}^{-}
]

这个定义的含义是：如果元素 (z) 在正样本中比负样本中响应更高，那么它更可能支持意图 (c)。

---

## 6.2 判别性关系学习

由于一些元素可能在大量类别中都普遍出现，例如 person、outdoor 等，仅靠频率差异还不够。为了提高区分性，可以引入 hardest negative 或混淆类别约束。

设 (\mathcal{N}(c)) 表示与意图 (c) 最容易混淆的一组负类，则可定义更具判别性的关系：

[
R(c,z)=\mathbb{E}[e_z(x)\mid y_c=1] - \mathbb{E}[e_z(x)\mid y_{\tilde c}=1,\ \tilde c\in\mathcal{N}(c)]
]

这样学到的关系不只是“常见元素”，而是“能把该意图与近邻意图区分开的元素”。

---

## 6.3 稀疏 evidence profile

为减少噪声，我们不使用所有元素，而是对每个意图仅保留最有支持作用的 top-N 元素，形成该意图的数据驱动证据模板：

[
\mathcal{P}*c = \mathrm{TopN}{R(c,z)}*{z\in\mathcal{Z}}
]

其中 (\mathcal{Z}) 是所有元素的全集。
这样，每个意图都有一个显式的 evidence profile，例如：

* top supporting objects
* top supporting scenes
* top supporting styles
* top supporting activities

这一步很关键，因为它让最终验证过程保持显式、可解释，而不是变成黑箱小头。

---

# 7. Explicit Evidence Verification

对于测试图像 (x) 和候选意图 (c\in\mathcal{H}(x))，我们根据图像中激活的元素及其与该意图的关系强度，计算验证分数。

最基本的形式为：

[
V(c,x)=\sum_{z\in\mathcal{A}(x)} w_z(x),R(c,z)
]

其中：

* (\mathcal{A}(x)) 表示图像激活元素集合
* (w_z(x)) 表示图像对元素 (z) 的响应强度
* (R(c,z)) 表示元素 (z) 对意图 (c) 的支持强度

这个分数的直观解释是：

> 图像中真实出现了哪些元素，这些元素在训练数据中对候选意图有多强支持，把它们加权累积起来，就得到该意图的 evidence verification score。

---

## 7.1 分专家验证

为了保留不同专家的语义角色，我们可以分开计算四类验证分数：

[
V^{obj}(c,x)=\sum_{z\in\mathcal{A}^{obj}(x)} w_z(x),R^{obj}(c,z)
]

[
V^{scene}(c,x)=\sum_{z\in\mathcal{A}^{scene}(x)} w_z(x),R^{scene}(c,z)
]

[
V^{style}(c,x)=\sum_{z\in\mathcal{A}^{style}(x)} w_z(x),R^{style}(c,z)
]

[
V^{act}(c,x)=\sum_{z\in\mathcal{A}^{act}(x)} w_z(x),R^{act}(c,z)
]

最终总验证分数为：

[
V(c,x)=\beta_{obj}V^{obj}(c,x)+\beta_{scene}V^{scene}(c,x)+\beta_{style}V^{style}(c,x)+\beta_{act}V^{act}(c,x)
]

其中 (\beta) 可以是固定系数，也可以基于验证集为每个类别选择最优专家组合。

---

## 7.2 support–contradiction 版本

进一步增强区分性，可以为每个意图同时学习支持元素和反证元素：

[
R^+(c,z),\quad R^-(c,z)
]

则验证分数可以写成：

[
V(c,x)=\sum_{z\in\mathcal{A}(x)} w_z(x),R^+(c,z)
-\lambda\sum_{z\in\mathcal{A}(x)} w_z(x),R^-(c,z)
]

这能显式建模某些元素“支持某个意图”以及“与某个意图相矛盾”两种关系。

---

# 8. Candidate Reranking

最终，我们只在候选意图集合 (\mathcal{H}(x)) 内，用验证分数对原始分数做局部修正：

[
S(c,x)=S^{base}(c,x)+\alpha V(c,x),\quad c\in\mathcal{H}(x)
]

对于非候选类别，不进行改写。
这样可以避免 evidence verification 在全局范围内引入过多噪声，同时保持强基线的召回能力。

最后根据修正后的候选分数完成 reranking，并通过类特异阈值输出多标签结果。

# 9. 方法特点

这套方法的优点可以明确概括为三点：

### 1. 显式

不是让模型再隐式学一个小分类头，而是先识别元素，再验证这些元素是否支持候选意图。

### 2. Data-driven

意图与证据的关系来自训练数据，而不是依赖 LLM 或手工定义模板。

### 3. 可解释

每个意图都能对应到一组从数据中学出来的支持元素；每次预测也都能指出图像中哪些元素为该意图提供了证据。

---

# 10. 实验计划

## 10.1 主实验

数据集：Intentonomy
指标：

* Macro F1
* Micro F1
* Samples F1
* mean F1
* mAP
* hard

对比方法：

1. 强基线（你现有最好版本）
2. 强基线 + candidate rerank
3. 强基线 + LLM template verification
4. 强基线 + data-driven element verification
5. 强基线 + data-driven element verification + class-specific threshold

重点看 data-driven verification 是否稳定优于 template verification。

---

## 10.2 Relation learning 方式对比

比较不同 intent–element relation 定义：

1. 纯正样本频率
2. 正负差值
3. hardest-negative discriminative version
4. support-only vs support+contradiction

目的：验证“判别性 relation”是否优于“常见性 relation”。

---

## 10.3 Expert 分析

分别测试：

* object only
* scene only
* style only
* activity only
* all experts

看哪些 expert 真正有效，哪些只是加噪声。

---

## 10.4 稀疏性分析

比较每个意图保留的元素数目：

* top-5
* top-10
* top-20
* all elements

验证 sparse evidence profile 是否更适合 rerank。

---

## 10.5 候选验证分析

做几个关键诊断：

### verification gap

比较正确候选和错误候选的验证分数差异，检查 verification 是否真的有判别力。

### oracle top-k upper bound

如果 top-k 中包含正确类，理想 rerank 最多能提升多少。
这可以判断问题是在 candidate proposal 还是 verification 本身。

### correlation analysis

比较 verification score 和 base score 的相关性。
如果 data-driven verification 比 template verification 的相关性更低但效果更好，说明它确实提供了更独立的信息。

---

## 10.6 可解释性案例

展示几张典型图像，列出：

* 基线 top-k 候选
* 图像识别到的 top elements
* 某候选意图的数据驱动 evidence profile
* verification 后分数变化

---

# 12. 最小版本

先跑通这个 MVP：

### 版本 A

* 四类 expert 先都保留
* relation 用正负差值
* 每个 intent 保留 top-N supporting elements
* verification 用简单加权和
* 只在 top-k 候选里加残差

公式就是：

[
R(c,z)=\mu_{c,z}^{+}-\mu_{c,z}^{-}
]

[
V(c,x)=\sum_{z\in\mathcal{A}(x)} w_z(x),R(c,z)
]

[
S(c,x)=S^{base}(c,x)+\alpha V(c,x)
]
