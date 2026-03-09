# SUIL: Structure- and Uncertainty-aware Intent Learning for Multi-label Image Intent Recognition

## 1. 方法概览

SUIL 的核心目标是：在不破坏强 `frozen CLIP` baseline 的前提下，围绕 **监督不确定性** 与 **标签结构** 两个关键难点，设计一个统一且稳定的 intent recognition 框架。

本文档当前包含三个核心模块：

1. **Confidence-aware Binary Supervision**
2. **Hierarchy-aware Structure Regularization**
3. **Class-wise Decision Calibration**

整体思路是：

- 用 **binary target** 保持清晰的决策边界；
- 用 **soft score** 表示监督强度，而不是直接作为 soft target；
- 用 **hierarchy** 约束预测结构一致性；
- 用 **class-wise calibration** 将“类特异阈值”从测试技巧变成可学习决策层。

---

## 2. Confidence-aware Binary Supervision

### 2.1 核心思想

该模块的基本出发点是：

- **二值标签 `y`** 用来定义清晰的决策边界；
- **soft score `s`** 不直接作为 soft target，而是作为监督置信度；
- 置信度主要用于调节正类损失强度。

换言之，`y` 决定“是否是正类”，而 `s` 决定“这个正类有多可信”。

### 2.2 置信度映射

对于每个样本-类别对 `(i, c)`，定义置信度权重：

$$
w_{ic} =
\begin{cases}
g(s_{ic}), & y_{ic} = 1 \\
1, & y_{ic} = 0
\end{cases}
$$

其中，`g(\cdot)` 是单调递增映射。最简单的形式可以写为：

$$
g(s) = 1 + \lambda s
$$

也可以使用归一化形式：

$$
g(s) = \alpha + (1 - \alpha)s, \qquad \alpha \in (0, 1]
$$

如果原始 soft label 只取 `{1/3, 2/3, 1}`，还可以直接离散化定义：

$$
g(1/3) = w_1,\qquad
g(2/3) = w_2,\qquad
g(1) = w_3,\qquad
w_1 \le w_2 \le w_3
$$

这个离散版本非常实用，也更容易解释。

### 2.3 Confidence-aware ASL

baseline 已经使用 `ASL`，因此最稳的做法是只在 `ASL` 外层加权，而不改变整体损失框架。

先写标准 multi-label ASL 形式。对类别 `c`，定义：

$$
L_{ic}^{\text{ASL}}
=
- y_{ic}(1 - p_{ic})^{\gamma_+}\log p_{ic}
- (1 - y_{ic})\tilde p_{ic}^{\gamma_-}\log(1 - \tilde p_{ic})
$$

其中 negative probability shifting 后：

$$
\tilde p_{ic} = \max(p_{ic} - m, 0)
$$

这里：

- `\gamma_+`、`\gamma_-` 是 focusing 参数；
- `m` 是 negative margin。

在此基础上，定义 **confidence-aware binary supervision**：

$$
L_{\text{conf}}
=
\frac{1}{NC}
\sum_{i=1}^N \sum_{c=1}^C
w_{ic} L_{ic}^{\text{ASL}}
$$

如果想更保守，建议只给正项加权，而不改动负项。此时可写为：

$$
L_{ic}^{\text{conf-ASL}}
=
- w_{ic} y_{ic}(1 - p_{ic})^{\gamma_+}\log p_{ic}
- (1 - y_{ic})\tilde p_{ic}^{\gamma_-}\log(1 - \tilde p_{ic})
$$

最终损失为：

$$
L_{\text{conf}}
=
\frac{1}{NC}
\sum_{i=1}^N \sum_{c=1}^C
L_{ic}^{\text{conf-ASL}}
$$

当前更推荐这个版本，因为不确定性主要来自“正类到底有多可信”，而不是负类。

### 2.4 模块解释

这个定义表达的是：

- `y_{ic}` 决定类别 `c` 是否作为正类监督；
- `s_{ic}` 不改变监督方向，只改变监督强度；
- 标注一致性越高，模型越应该强力拟合该正类；
- 标注一致性较低的正类仍被保留为正类，但不会被过度拉扯。

这与“直接做 soft label regression”有本质区别。

---

## 3. Hierarchy-aware Structure Regularization

这里不建议直接上复杂图网络，而建议先做 **预测空间上的结构一致性约束**。这样更稳，也更容易和强 baseline 兼容。

假设 Intentonomy 的类别集合具有层次结构。记：

- 细粒度类别集合为 `\mathcal{C}`；
- 层次结构中每个类别 `c` 的父节点集合为 `\mathcal{P}(c)`。

如果是树结构，`\mathcal{P}(c)` 只有一个父节点；如果是 `DAG`，则可以有多个父节点。

### 3.1 从 fine logits 聚合 coarse logits

定义 coarse 节点 `u` 的概率由其子节点集合 `\text{Ch}(u)` 聚合而来。一个自然的 `noisy-or` 形式为：

$$
\hat p_u = 1 - \prod_{c \in \text{Ch}(u)} (1 - p_c)
$$

它的含义是：只要某个子 intent 成立，父 intent 也应具有较高概率。

如果觉得 `noisy-or` 太激进，也可以采用 `max` 近似：

$$
\hat p_u = \max_{c \in \text{Ch}(u)} p_c
$$

但论文里使用 `noisy-or` 会更正式，也更可微。

### 3.2 Parent-child consistency regularization

最稳的结构正则是要求：

> 子类概率不应明显高于父类概率。

对每条父子边 `(u, c)`，定义一致性损失：

$$
L_{\text{pc}}
=
\frac{1}{N|\mathcal{E}|}
\sum_{i=1}^N \sum_{(u,c)\in\mathcal{E}}
\max(0,\, p_{ic} - \hat p_{iu})
$$

其中，`\mathcal{E}` 是所有 parent-child 边的集合。

如果希望约束更柔和，可以引入 margin：

$$
L_{\text{pc}}
=
\frac{1}{N|\mathcal{E}|}
\sum_{i=1}^N \sum_{(u,c)\in\mathcal{E}}
\max(0,\, p_{ic} - \hat p_{iu} - \delta)
$$

其中，`\delta \ge 0` 是容忍边界。

这个形式的优点在于，它只惩罚“子类明显高于父类”的不合理预测。

### 3.3 Coarse-to-fine auxiliary supervision

如果可以从 fine label `y_i` 自动构造 coarse label，那么还可以加入 coarse level 监督。

对 coarse 节点 `u`，定义二值 coarse target：

$$
y_{iu}^{\text{coarse}}
=
\mathbb{I}\Big(\sum_{c \in \text{Ch}(u)} y_{ic} > 0\Big)
$$

然后对聚合出的 `\hat p_u` 做 `BCE` 或 `ASL`。若以 `BCE` 为例：

$$
L_{\text{coarse}}
=
\frac{1}{N|\mathcal{U}|}
\sum_{i=1}^N \sum_{u \in \mathcal{U}}
\left[
- y_{iu}^{\text{coarse}}\log \hat p_{iu}
- (1 - y_{iu}^{\text{coarse}})\log(1 - \hat p_{iu})
\right]
$$

其中，`\mathcal{U}` 是 coarse nodes 集合。

### 3.4 最终 hierarchy loss

可以先使用最小版本：

$$
L_{\text{hier}} = L_{\text{pc}}
$$

也可以使用增强版本：

$$
L_{\text{hier}} = L_{\text{coarse}} + \beta L_{\text{pc}}
$$

其中，`\beta` 用于控制一致性约束强度。

### 3.5 模块解释

这个模块的重点不是“让语义相似类更接近”，而是：

- 利用标签层次结构提供额外监督；
- 让预测满足 `coarse-to-fine` 的逻辑一致性；
- 避免模型输出结构上互相冲突的 intent 分布。

它比单纯的 semantic similarity regularization 更贴近任务本身。

---

## 4. Class-wise Decision Calibration

这一部分最好明确写成：

> 将 `per-class threshold / calibration` 从测试时技巧，转化为可学习的决策层。

最稳的做法是只在 logits 上做类特异校准，而不碰 backbone。

### 4.1 Class-wise affine calibration

对每个类别 `c`，定义校准后的 logit：

$$
\tilde z_{ic} = a_c z_{ic} + b_c
$$

其中：

- `a_c > 0` 是 class-wise temperature / scale；
- `b_c` 是 class-wise bias / threshold shift。

对应概率为：

$$
\tilde p_{ic} = \sigma(\tilde z_{ic})
$$

为了保证 `a_c > 0`，实际实现中可以参数化为：

$$
a_c = \text{softplus}(\rho_c)
$$

其中，`\rho_c` 是可学习参数。

如果想先做最简版本，只学习 bias 就足够：

$$
\tilde z_{ic} = z_{ic} + b_c
$$

这个形式已经非常接近“可学习类别阈值”，而且通常更稳。

### 4.2 用校准概率参与训练

将主分类损失中的 `p_{ic}` 全部替换为 `\tilde p_{ic}`，即主损失实际作用于校准后的输出：

$$
L_{\text{conf}}(\tilde p)
$$

而不是原始的 `p`。

这样 calibration 就不再是后处理，而是一个端到端学习到的决策修正模块。

### 4.3 与显式 threshold 的关系

推理阶段通常采用如下预测规则：

$$
\hat y_{ic} = \mathbb{I}(\tilde p_{ic} > 0.5)
$$

由于：

$$
\tilde p_{ic} = \sigma(z_{ic} + b_c)
$$

它等价于在原始 logit 空间使用类别相关阈值：

$$
\hat y_{ic} = \mathbb{I}(z_{ic} > -b_c)
$$

因此，`b_c` 实际上就是一个 **learnable class-specific threshold**。

这一点在论文中非常好用，因为它把“校准头”和“类别阈值学习”统一到同一套表述里。

### 4.4 可选的 calibration regularization

为了防止 calibration head 过拟合，可以加入一个轻量正则项。

若只使用 bias：

$$
L_{\text{cal-reg}} = \frac{1}{C}\sum_{c=1}^C b_c^2
$$

若使用 scale + bias：

$$
L_{\text{cal-reg}}
=
\frac{1}{C}\sum_{c=1}^C \left[(a_c - 1)^2 + b_c^2\right]
$$

这个正则项的含义是：默认让校准层接近 identity，只在确有必要时进行调整。

---

## 5. 总体训练目标

设主干输出 logits 为：

$$
z = f_{\theta}(x)
$$

校准后 logits 为 `\tilde z`，校准概率为 `\tilde p`。

总损失可以写为：

$$
L
=
L_{\text{conf}}(\tilde p)
+ \lambda_{\text{hier}} L_{\text{hier}}(\tilde p)
+ \lambda_{\text{cal}} L_{\text{cal-reg}}
$$

其中：

- `L_{\text{conf}}`：confidence-aware binary supervision
- `L_{\text{hier}}`：hierarchy-aware regularization
- `L_{\text{cal-reg}}`：calibration parameter regularization

如果采用增强版 hierarchy 设计，则：

$$
L_{\text{hier}} = L_{\text{coarse}} + \beta L_{\text{pc}}
$$

此时完整目标可写为：

$$
L
=
L_{\text{conf}}
+ \lambda_1 L_{\text{coarse}}
+ \lambda_2 L_{\text{pc}}
+ \lambda_3 L_{\text{cal-reg}}
$$

---

## 6. 推荐的第一版：最小稳定版

如果目标是先做一个最不容易炸、最适合验证方向是否有效的版本，建议从以下配置开始：

### 6.1 结构设定

- `confidence` 只加权正项；
- `hierarchy` 只使用 `parent-child consistency`；
- `calibration` 只学习 bias，不学习 scale。

### 6.2 形式化定义

校准层：

$$
\tilde z_{ic} = z_{ic} + b_c
$$

总损失：

$$
L
=
L_{\text{conf}}(\sigma(\tilde z))
+ \lambda_{\text{pc}} L_{\text{pc}}(\sigma(\tilde z))
+ \lambda_{\text{cal}} \frac{1}{C}\sum_c b_c^2
$$

### 6.3 为什么推荐这个版本

这个版本最适合作为第一版 SUIL，原因是：

- 改动非常克制，不会破坏强 baseline；
- 三个模块都能落地，但都采用最稳的实现形式；
- 便于做 ablation，能够清楚回答“收益到底来自 uncertainty、structure，还是 calibration”。

如果这一版有效，再逐步扩展到：

- 离散式 confidence mapping；
- `coarse-to-fine` 辅助监督；
- `scale + bias` 联合校准；
- 多层 frozen CLIP feature fusion。
