## 方案 A：Learnable affine fusion

### 形式

对每个候选类 (c \in \mathcal{T}_k(x))，有：

* baseline logit (z_c)
* text prior score (s_c)

学习一个非常小的 affine fusion：

[
z'_c = z_c + a_c , s_c + b_c
]

或者更稳一点：

[
z'_c = \lambda_c z_c + \mu_c s_c + b_c
]

但我建议先用第一种，简单很多：

[
z'_c = z_c + a_c s_c + b_c
]

其中：

* (a_c, b_c) 是可学习参数
* 只对 top-k 内启用
* top-k 外保持 (z'_c = z_c)

### 初始化

为了稳：
[
a_c = \alpha_0,\quad b_c=0
]
其中 (\alpha_0) 可以初始化成你当前最优 plain rerank 的 `0.3`。

这样训练一开始就接近你现在最强版本，而不是从零乱学。

---

## 方案 B：Per-candidate MLP fusion

这是第二步，可以更强一点，但风险也更高。

### 形式

对每个候选类：

[
\delta_c = f_\psi([z_c, s_c])
]

[
z'_c = z_c + \delta_c
]

其中 (f_\psi) 是一个很小的 MLP，比如：

* 输入 2 维
* hidden 8 或 16
* 输出 1 维

你也可以把额外特征拼进去：

[
[z_c, s_c, p_c, r_c]
]

其中：

* (p_c = \sigma(z_c))
* (r_c) 是该候选在 top-k 里的 rank


* 参数极小
* 强残差约束
* 只在 top-k 内
* 冻结 baseline

---

## 方案 C：Pairwise candidate fusion

这个更像真正的 reranker，但我建议放后面。

### 形式

不是单看 ((z_c, s_c))，而是看候选间相对信息：

[
\delta_c = f_\psi\Big(z_c, s_c, \text{mean}*{j\in T_k}s_j, \max*{j\neq c}s_j \Big)
]

然后：
[
z'_c = z_c + \delta_c
]

# 四、我最建议的第一版：从 SLR-v0 出发做 residual learnable fusion

你现在最强的是 SLR-v0。
所以最稳的升级方式不是推翻它，而是：

# **SLR-LF: SLR with Learnable Fusion**

---

## 版本 1：Class-wise residual fusion

[
z'_c =
\begin{cases}
z_c + a_c s_c + b_c, & c \in T_k(x) \
z_c, & \text{otherwise}
\end{cases}
]

训练时只学 (a_c, b_c)。

### 优点

* 和现有 SLR-v0 完全连续
* 最接近你当前 strongest heuristic
* 非常容易做 ablation：

  * fixed (\alpha) vs learnable (a_c)
  * no bias vs learnable (b_c)

### 这一步很可能就已经能回答你最担心的问题：

> 这不是 prompt engineering，而是一个可学习的局部语义融合模块。

---

## 版本 2：Shared tiny MLP fusion

[
z'*c =
\begin{cases}
z_c + f*\psi([z_c, s_c]), & c \in T_k(x) \
z_c, & \text{otherwise}
\end{cases}
]

其中同一个 (f_\psi) 共享给所有类。

### 为什么共享

这样参数量更小，也更像“通用融合机制”，而不是 class memorization。

### 可以加的最少特征

建议先试：

* (z_c)
* (s_c)

然后再试：

* (p_c = \sigma(z_c))
* normalized rank in top-k

不要一开始塞太多。

---

# 五、怎么训练，才不容易把 baseline 搞坏

这一步特别重要。

---

## 训练原则 1：冻结 baseline

先完全冻结：

* CLIP encoder
* baseline MLP head

只训练 fusion module。

这样你不会又掉回“大模块扰动强 baseline”的老坑。

---

## 训练原则 2：只在 top-k 内计算 fusion

候选外直接保留原 logits。
这样 fusion 只负责“局部判别”，不会全局乱改。

---

## 训练原则 3：identity / residual initialization

### 对 affine fusion

初始化：
[
a_c = 0.3,\quad b_c=0
]

也可以更保守：
[
a_c = 0,\quad b_c=0
]
然后 warm start from SLR-v0 用起来更慢。

我更建议从 **0.3** 开始，因为你已经知道这是强配置。

### 对 MLP fusion

最后一层权重初始化为 0，保证：
[
f_\psi(\cdot)\approx 0
]
于是初始时退化为 baseline。

如果你想退化为 SLR-v0，也可以设：
[
f_\psi([z,s]) \approx 0.3 s
]
但实现稍麻烦。先不必。

---

## 训练原则 4：正则化

为了避免 fusion 过拟合：

### 对 affine fusion

[
L_{\text{reg}} = \lambda_a \sum_c (a_c-\alpha_0)^2 + \lambda_b \sum_c b_c^2
]

### 对 MLP fusion

[
L_{\text{reg}} = \lambda |\delta|_2
]

或者直接约束平均改动幅度。

---
## 做一个很关键的实验

你后面方法实验里，一定要加：

### 固定 fusion 方法，比较不同 text source

比如：

* short
* detailed
* mixed

如果 fusion 在不同 source 上都有效，说明方法不是依赖某套精心 prompt 才成立。
这非常重要。

---

# 七、建议你现在做的具体实验顺序
按这个顺序来，不要乱。

---

## Exp 1：Learnable affine fusion

比较：

1. baseline
2. SLR-v0
3. SLR-LF-affine

其中：
[
z'_c = z_c + a_c s_c + b_c
]

固定：

* `topk=10`
* 先用当前最稳的 text source（我建议先 `detailed`，再补 `short` 和 `mixed`）

这是最优先。

---

## Exp 2：Shared tiny MLP fusion

比较：

1. SLR-v0
2. affine fusion
3. shared MLP fusion

如果 MLP 没明显优于 affine，那就停在 affine。
那样方法更干净。

---

## Exp 3：跨 text source 稳定性

在最好的 fusion 上测：

* short
* detailed
* mixed

这是为了摆脱 prompt engineering 印象。

---

## Exp 4：hard case 分析

看 learnable fusion 是否：

* 比 plain rerank 更少伤 `Attractive / EnjoyLife / Art`
* 仍保持对 `WorkILike / Harmony / EasyLife` 的收益

这会帮助你解释“learnable fusion 的价值”。
