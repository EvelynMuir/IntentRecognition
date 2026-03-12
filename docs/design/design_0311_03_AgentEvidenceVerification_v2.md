# 1. 加入“反证”比继续加“支持”更值得

你现在已经用了 `support_contradiction`，但我怀疑它还没被充分用起来。
视觉意图 top-k 候选之间，很多时候不是正确类支持不够，而是**错误类没被有效打掉**。

所以可以把 verification 明确写成：

[
V(c,x)=V_{support}(c,x)-\lambda V_{contradict}(c,x)
]

关键不是简单从 relation 里取正负值，而是把 contradiction 单独建模得更强一些：

### 可以怎么做

对每个 intent (c)，除了学 top supporting elements，还学：

* top contradicting objects
* top contradicting scenes
* top contradicting styles
* top contradicting activities

测试时如果某候选 intent 激活了这些“反证元素”，直接扣分。

### 为什么这可能有效

因为 top-k 候选往往都共享很多正证据，真正拉开差距的常常是：

* 某个场景不对
* 某种风格不对
* 某个活动和该 intent 矛盾

这个方向很适合你现在的 hard disambiguation。

---

# 2. 从“单类验证”升级成“候选对比验证”

这是我最推荐的升级。

你现在大概率还是对每个候选单独算：

[
V(c,x)
]

但真正需要解决的是：

> 在 top-k 候选里，为什么 A 比 B 更合理？

所以你可以把 verification 改成 **pairwise / comparative verification**。

---

## 一个简单版本

对 top-k 里的任意两个候选 (c_i, c_j)，定义：

[
\Delta V(c_i,c_j,x)=V(c_i,x)-V(c_j,x)
]

然后用这个差值来决定谁排前面。

---

## 更强一点的版本

relation 也做成相对式：

[
R_{pair}(c_i,c_j,z)=R(c_i,z)-R(c_j,z)
]

测试时只看图像激活元素对这两个候选的**相对支持差异**：

[
\Delta V(c_i,c_j,x)=\sum_{z\in A(x)} w_z(x),[R(c_i,z)-R(c_j,z)]
]

这特别适合你当前 setting，因为你真正想解决的是：

* Happy vs Playful
* Enjoy life vs Happy
* Fine design vs Beauty

而不是给某个类打一个绝对分。

### 为什么这个方向强

因为你现在已经证明：

* candidate recall 足够
* 问题在 candidate-local verification

那自然下一步就是做**candidate-local comparison**，不是继续做单类打分。

---

# 3. relation learning 从“全局负类”改成“混淆簇负类”

你现在有 `hard_negative_diff`，这已经很对了。
但还可以更进一步：不要只用统一 hardest negatives，而是做 **class-specific confusion neighborhood**。

---

## 做法

先用当前 strongest baseline 跑验证集或训练集，统计 confusion matrix。
对每个 intent (c)，找最常被混淆的几个类：

[
\mathcal{N}(c)=\text{top confusing classes of } c
]

然后 relation 改成：

[
R(c,z)=\mathbb{E}[e_z|y_c=1]-\mathbb{E}[e_z|y_{\tilde c}=1,\tilde c\in \mathcal{N}(c)]
]

这样你学到的是：

> 这个 element 不是“总体上支持 c”，而是“相对于 c 最容易被错分成的那些类，它更支持 c”。

### 为什么值钱

这会让 verifier 更像一个专门的“近邻判别器”，而不是泛化的辅助模块。
对 Hard subset 很可能继续涨。

---

# 4. 加一个“何时相信 verifier”的 gate

你现在 focused verification 已经能拉高 Macro / Hard，但没完全拉高 Micro / Samples。
这很像 verifier 在某些样本上很有用，在另一些样本上会轻微扰动本来就对的排序。

所以很值得加一个**轻量 gating**：

[
S^{final}(c,x)=S^{base}(c,x)+g(x,c)\cdot V(c,x)
]

其中 (g) 不一定要学网络，先规则化就行。

---

## 最值得试的 gate

### 方案 A：margin-aware

当 top-1 和 top-2 候选差距很小时，verifier 权重大；
差距很大时，verifier 权重小。

例如：

[
g(x)=\mathbf{1}[\text{margin}(x)<\tau]
]

或平滑一点：

[
g(x)=\exp(-\gamma \cdot \text{margin}(x))
]

### 方案 B：evidence-confidence-aware

如果图像激活元素太少、太弱，说明 verifier 不可靠，就减小权重。

### 方案 C：class-specific gate

某些类更吃 verifier，某些类不太需要。
可以在验证集上为每个类选一个 (\alpha_c)。

---

## 为什么这很可能有效

你现在 verifier 已经能提升 hardest cases，但可能也会轻微扰动 easy/common cases。
gate 的作用就是：

> **只让 verifier 在它擅长的时候出手。**

这通常能把 Macro/Hard 的收益尽量保住，同时把 Micro/Samples 拉回来一些。

---

# 我最推荐的组合

如果你问我“下一轮最该试哪两个”，我会选：

### 第一优先级

**候选对比验证（pairwise / comparative verification）**

### 第二优先级

**margin-aware gate**

因为这两步都高度贴合你现在的问题本质：

* 候选已经召回到了
* verifier 的任务是局部区分
* verifier 不是每次都该强介入

---

# 一个很实用的最小升级版

你可以做一个不太重、但很像方法升级的版本：

---

## Step 1

保留 `scenario SLR-C` 给 top-k 候选。

## Step 2

保留 focused discriminative profiles：

* sparse
* `hard_negative_diff`
* `support_contradiction`

## Step 3

对 top-k 候选两两计算 comparative evidence score：

[
\Delta V(c_i,c_j,x)=\sum_{z\in A(x)} w_z(x),[R(c_i,z)-R(c_j,z)]
]

## Step 4

用 margin-aware gate 决定 verification 强度：

[
g(x)=\exp(-\gamma \cdot \text{base margin}(x))
]

## Step 5

更新候选排序。

这个版本比你现在的“单类 residual”明显更针对当前瓶颈。

---

# 还有两个次优但也值得试的点

## A. expert-level pairwise voting

不要直接合成一个 verification 分数，而是让：

* object vote
* scene vote
* style vote
* activity vote

分别对 top-2 / top-3 候选投票，看谁更合理。
最后多数票或加权票决定 rerank。

这会更稳，也更可解释。

## B. 只对 hardest classes 启用 focused verifier

如果你已经知道哪些类最难、最混淆，可以局部启用 verifier。
这会非常符合你的 Hard gain 现象。