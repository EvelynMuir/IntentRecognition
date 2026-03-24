# 一、方法主线

## 方法名

先用工作名：

**CAML-MP: Confusion-Aware Multi-Prototype Learning**

如果你想更论文一点，也可以叫：

**Confusion-Aware Multi-Prototype Contrastive Learning for Visual Intention Understanding**

---

# 二、方法设计

## 2.1 总体思路

保留你当前最稳的 baseline 主干，不改 backbone 主体，只加一个 **intent-centric embedding learning** 分支。

整体上：

* 主分支：原有 multi-label classifier
* 新分支：image-to-label prototype alignment
* 新分支：confusion-aware hard negative separation

最终损失：

[
\mathcal{L} = \mathcal{L}*{cls} + \lambda_p \mathcal{L}*{proto} + \lambda_h \mathcal{L}_{hard}
]

其中：

* (\mathcal{L}_{cls})：你当前最稳的分类损失（建议继续用 Asymmetric Loss / 现有 best）
* (\mathcal{L}_{proto})：图像与正标签 prototype 对齐
* (\mathcal{L}_{hard})：与 hardest confusing negatives 拉开 margin

---

## 2.2 Backbone 与特征

直接用你当前 best baseline 的视觉主干：

* CLIP ViT-L/14
* 用你现在最稳的 global feature 路径
* 不在第一阶段引入 patch token 复杂结构

输出图像表示：
[
h(x) \in \mathbb{R}^d
]

然后做 L2 normalize。

---

## 2.3 Label prototype 设计

### Phase 1：单 prototype

每个 intent 一个 prototype：
[
e_j \in \mathbb{R}^d,\quad j=1,\dots,C
]

### Phase 2：多 prototype

每个 intent 用 (M) 个 prototype：
[
e_{j,1}, \dots, e_{j,M}
]

原因很直接：
同一 intent 的视觉形态差异很大，你前面很多方法不 work，也可能是因为 **一个 label 对应的 visual mode 不止一个**。HLEG 也强调，同类 intention 的视觉内容本身是多样的。

### prototype 初始化

按顺序做三种：

1. random learnable
2. text-initialized（用 class text / description 初始化）
3. data-initialized（用训练集正样本 feature 均值或 k-means 中心初始化）

我建议先跑 1 和 3。

---

## 2.4 Prototype alignment loss

### 单 prototype 版本

对样本 (x)，正标签集合为 (P(x))，定义：

[
s_j = h(x)^\top e_j
]

然后做正标签对齐：

[
\mathcal{L}_{proto}
===================

-\frac{1}{|P(x)|}\sum_{j\in P(x)}
\log
\frac{\exp(s_j/\tau)}
{\sum_{k=1}^{C}\exp(s_k/\tau)}
]

这本质上是在说：
图像表示应该更靠近它的正 intent prototypes。

### 多 prototype 版本

对 label (j)，取最匹配的 prototype：

[
s_j = \max_{m=1,\dots,M} h(x)^\top e_{j,m}
]

然后用同样的 softmax alignment。
这样每个 label 可以有多个视觉 mode，不需要一个 prototype 硬吃所有样本。

---

## 2.5 Confusion-aware hard negative loss

这是这条方法最重要的部分之一。

### hard negative 候选来源

对每个正标签 (j)，构造 hardest negatives (N_j(x))。来源做消融：

1. **confusion prior**
   来自你已有 baseline / SLR-C 的 confusion matrix top-k 邻居
2. **hierarchy neighbor**
   层级邻近类
3. **top-scoring false positives**
   当前模型在该样本上打分最高但 GT 为负的类
4. **union**
   上述三者并集

我最建议先从 1 开始，因为这最贴你当前目标。

### hard negative margin

对每个正标签 (j) 和 hard negative (k)：

[
\mathcal{L}_{hard}
==================

\frac{1}{|P(x)|}\sum_{j\in P(x)}
\frac{1}{|N_j(x)|}\sum_{k\in N_j(x)}
\max(0, m - s_j + s_k)
]

其中：

* (m) 是 margin
* (s_j) 是正标签 prototype 相似度
* (s_k) 是 hard negative prototype 相似度

这一步不是后验 verifier，而是在训练时就要求：

> 对 hardest confusing negatives，正类必须留出 margin。

---

## 2.6 分类头与 prototype 分支的关系

分类头不要删。
预测最终仍然主要靠现有分类头，prototype 分支作为表征塑形约束。

也就是：

* 训练：分类头 + prototype / hard negative 双分支
* 测试：先用原分类头输出
  第一阶段不要把 prototype score 直接拿来替代分类分数

原因很简单：
你现在最需要的是稳定性，先让 prototype loss 改 feature space，而不是上来赌一个新 inference rule。

---

# 三、关键验证问题

---

## 验证 1：问题是不是“embedding 没拉开”

### 实验

比较：

1. baseline
2. baseline + prototype alignment
3. baseline + prototype alignment + hard negative margin

### 预期

如果 3 在 Hard 上明显优于 1/2，就说明：

> 训练期的表征拉开，比后验 decision conditioning 更关键。

---

## 验证 2：增益是否主要来自 confusion-aware negatives

### 实验

hard negatives 分别来自：

1. random negatives
2. confusion matrix
3. hierarchy neighbors
4. top-scoring false positives
5. union

### 预期

如果 confusion-aware 负类最好，论文故事就很顺：

> 不只是 contrastive，而是 **针对视觉意图歧义的 confusion-aware contrastive**。

---

## 验证 3：类内多模态是否是真问题

### 实验

比较：

1. single prototype / class
2. 2 prototypes / class
3. 4 prototypes / class
4. 8 prototypes / class

### 观察

* Macro
* Hard
* tail classes
* 类内相似度分布

### 预期

如果 multi-prototype 尤其提升 Hard / tail，就说明：

> intent 类内差异确实需要多 mode 建模。

---

## 验证 4：prototype 分支是 regularizer 还是 inference source

### 实验

比较：

1. 只训练时加 prototype loss，测试只用分类头
2. 测试时线性融合分类头和 prototype scores
3. 只用 prototype scores 推理

### 预期

大概率 1 或 2 最稳。
如果 3 很差，反而能说明 prototype 更适合作为 representation shaper，而不是直接 classifier。

---

# 四、实验清单

---

## 4.1 主对比

你至少需要这张主表：

* baseline best
* SLR-C best
* fixed bank best
* latent basis MVP
* baseline + prototype
* baseline + prototype + hard negative
* baseline + multi-prototype + hard negative

指标：

* Macro
* Micro
* Samples
* mAP
* Hard

重点盯：

* Hard
* Macro
* 和 SLR-C / fixed bank 的差距是否缩小或反超

---

## 4.2 消融表 A：hard negative 构造

固定 single prototype，比较：

* no hard negative
* random negative
* confusion negative
* hierarchy negative
* top-scoring FP negative
* confusion + FP

---

## 4.3 消融表 B：prototype 数量

固定最好 hard negative 策略，比较：

* 1 proto / class
* 2 proto / class
* 4 proto / class
* 8 proto / class

---

## 4.4 消融表 C：prototype 初始化

固定最好配置，比较：

* random init
* text init
* feature centroid init
* k-means init

---

## 4.5 消融表 D：loss 权重

比较：

* (\lambda_p) = 0.05 / 0.1 / 0.2 / 0.5
* (\lambda_h) = 0.05 / 0.1 / 0.2 / 0.5
* margin (m) = 0.1 / 0.2 / 0.3 / 0.5
* temperature (\tau) = 0.05 / 0.1 / 0.2

建议优先粗扫：

* (\lambda_p = 0.1)
* (\lambda_h = 0.1)
* (m = 0.2)
* (\tau = 0.1)

---

# 五、诊断实验

这些很重要，能解释为什么方法有效或无效。

---

## 5.1 confusion pair 分析

选你最容易混淆的 top-10 / top-20 label pairs，统计：

* 基线下 pairwise confusion
* 新方法下 pairwise confusion

你最想看到的是：

* Hard 总体提升
* 具体 confusion pairs 有实打实下降

---

## 5.2 embedding 可视化

对 hardest classes 做：

* t-SNE / UMAP of image features
* 可视化 prototype 位置

比较：

* baseline embedding
* prototype-trained embedding

重点看：

* 原本重叠的 hard classes 是否更分开
* 同类样本是否更聚拢

---

## 5.3 prototype usage 分析

多 prototype 版本下，统计：

* 每个 class 的 prototype 使用分布
* 是否有明显塌缩到单个 prototype
* 不同 prototype 对应的视觉 mode 是否不同

如果出现“每类 4 个 prototype 但基本只用 1 个”，那多 prototype 设计要重新评估。

---

## 5.4 calibration 分析

看：

* reliability diagram
* ECE / per-class confidence trend

虽然你的主目标不是 calibration，但如果 Hard 涨的同时 calibration 没崩，这是加分项。

---

# 六、实验顺序

我建议按这个顺序，别一上来全堆。

---

## Phase 1：最小可行版

### 模型

* baseline
* * 单 prototype / class
* * confusion-aware hard negative

### 目标

先回答最关键问题：

> training-time 的 prototype + hard negative，能不能让 Hard 至少超过 baseline，并逼近 SLR-C？

这是第一道门槛。

---

## Phase 2：hard negative 来源消融

如果 Phase 1 有信号，再做：

* confusion vs hierarchy vs FP

### 目标

找到最有效的 hard negative 构造方式，顺便把 novelty 立起来。

---

## Phase 3：多 prototype

如果 single prototype 有正信号，再上：

* 2 / 4 / 8 prototypes

### 目标

验证类内多样性建模是否进一步提升 Hard。

---

## Phase 4：诊断分析

补：

* confusion pairs
* t-SNE / UMAP
* prototype usage
* calibration

---

# 七、判停标准

这个很重要，避免你再被一条线拖太久。

## 可以继续深挖的信号

满足任一条都值得继续：

* Hard 超过 baseline 且接近或超过 SLR-C
* Macro 稳定涨 0.5 以上
* confusion pairs 明显改善
* 多随机种子下波动可接受

## 建议止损的信号

出现这些就该及时停：

* single prototype 完全不涨，Hard 无改善
* 加 hard negatives 后训练明显不稳
* 多 prototype 大面积塌缩
* Hard 和 Macro 都持续低于 SLR-C / fixed bank

---

# 八、你现在最该先跑的配置

我建议直接从这个最小版本开跑：

## MVP

* backbone：当前 best baseline
* prototype：1 / class
* init：feature centroid init
* hard negatives：confusion matrix top-3
* loss：

  * 原分类损失
  * (\lambda_p = 0.1)
  * (\lambda_h = 0.1)
  * (m = 0.2)
  * (\tau = 0.1)

### 首轮比较

* baseline
* baseline + prototype
* baseline + prototype + hard negative

只看：

* Macro
* Hard
* top confusion pairs

这一步有信号，再继续往 multi-prototype 扩。