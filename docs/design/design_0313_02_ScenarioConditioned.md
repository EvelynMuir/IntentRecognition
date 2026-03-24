## 核心思想

传统 multi-label intent classifier 默认：

* 标签关系是全局固定的
* 所有样本共享一个 label graph / threshold / bias

但在视觉意图里，这很可能不对。

同一个标签关系在不同 scenario 下会变：

* 某些场景里，A 和 B 经常共现
* 换个场景，A 和 B 就变成互斥或弱相关
* 某些标签在某类 scenario 下应该更容易被激活
* 某些 hard class 的边界其实是 scenario-dependent 的

所以你要做的是：

## **对每张图先估计 scenario distribution，再让它条件化最终的 label relation 和 decision boundary。**

---

# 方法结构

## 1. Backbone

继续用你已有最稳的 CLIP baseline，不折腾 backbone。

输入图像后得到：

* global image feature (v)
* 或 patch-pooled feature，按你现有最优实现来

---

## 2. Scenario predictor

先预测一个 scenario 分布：

[
p(s \mid x) \in \mathbb{R}^{M}
]

这里 scenario 可以是：

### 方案 A：已有 scenario taxonomy

如果你已经有 SLR-C 里的 scenario 定义，就直接用。

### 方案 B：从标签共现 / 视觉聚类中挖 scenario prototypes

如果现成 scenario 不够强，再扩展。

但建议先别扩，先用你已有 SLR-C 最稳的 scenario。

---

## 3. Scenario-conditioned label relation

这是核心。

传统做法是一个固定 label relation matrix：
[
R \in \mathbb{R}^{C \times C}
]

你现在改成：

[
R(x) = \sum_{m=1}^{M} p(s_m \mid x), R_m
]

其中：

* (R_m) 是第 (m) 个 scenario 专属的 label relation matrix
* 每张图根据自己的 scenario 分布，动态生成 relation graph

这一步非常关键。它意味着：

* label co-occurrence 不是固定的
* label exclusion 不是固定的
* hard label pair 的相对边界不是固定的

---

## 4. Scenario-conditioned classifier correction

baseline logits 记为：
[
z \in \mathbb{R}^{C}
]

然后用 scenario-conditioned relation 去修正：

[
z' = z + \alpha , R(x), \sigma(z)
]

或者更稳定一点：

[
z' = z + \alpha , g(x)
]

其中 (g(x)) 是基于 scenario-conditioned graph propagation 的 residual correction。

你可以把它理解成：

* baseline 先给初始判断
* scenario 告诉模型“在这种语境下，哪些标签该一起上调，哪些该互相抑制”

---

## 5. Scenario-conditioned threshold / calibration

这一步我觉得很值得做，而且实现不重。

因为很多 hard 类问题不一定是 representation 不行，而是**阈值不对**。

所以别让所有类都用固定 threshold，而是：

[
\tau_j(x) = \tau_j^{base} + \Delta \tau_j(s)
]

也就是：

* 每个 label 有基础阈值
* 再根据 scenario 做偏移

最终预测：
[
\hat y_j = \mathbf{1}[z'_j > \tau_j(x)]
]

这很适合你的任务，因为 intent 本来就是 subjectively activated 的，scenario-dependent threshold 很合理。

---

# 这个方法为什么比你之前的东西更不一样

因为它不是：

* 又一个 bank
* 又一个 verifier
* 又一个 factor module

而是明确在改：

## **标签空间本身的条件结构**

你的论文故事会变成：

> 现有方法默认标签依赖是全局固定的，但视觉意图中的标签关系高度依赖语境。我们提出用 scenario 作为中间上下文变量，动态调制标签关系、阈值与校准，从而提升细粒度和歧义类别识别。

这个故事明显更完整，也更像你目前结果自然推出来的方向。

---

# 应该做哪些验证

这个方向的验证会比 latent basis 清楚很多。

## 验证 1：scenario 真的是有效中间变量吗？

比较：

* baseline
* baseline + scenario auxiliary prediction
* baseline + scenario-conditioned bias
* baseline + scenario-conditioned relation
* baseline + scenario-conditioned relation + threshold

如果只有 auxiliary scenario 没啥用，但 condition 进去后明显涨，就说明：

> scenario 的价值不在“多一个任务”，而在“重写 label decision”。

---

## 验证 2：固定 label graph 和条件化 graph 谁更好？

比较：

* static relation matrix
* scenario-conditioned relation matrix

看：

* Hard
* confusion-heavy classes
* per-class calibration

你很可能会看到 static graph 有点用，但不如 conditional graph。

---

## 验证 3：增益来自 relation 还是 threshold？

比较：

* relation only
* threshold only
* relation + threshold

这个实验很关键，因为它告诉你：

* 是标签传播贡献大
* 还是 calibration 贡献大
* 还是二者互补

---

## 验证 4：哪些类最受益？

把类别按下面拆：

* 高混淆类
* 高共现类
* tail 类
* hierarchy 相邻类

如果 scenario-conditioned 模型主要提升这些类，故事就成立了。

---

# 实验清单

## 主对比

* baseline
* fixed bank best
* SLR-C
* SLR-C + static label graph
* SLR-C + scenario-conditioned label graph
* SLR-C + scenario-conditioned threshold
* SLR-C + relation + threshold

## 消融

* scenario 数量
* scenario 分布用 hard assignment 还是 soft assignment
* relation matrix 是否对称
* 只建 positive correlation / 加 negative suppression
* threshold 是否 class-specific
* relation 用一层传播还是两层

## 诊断

* 不同 scenario 下 label relation heatmap
* 不同 scenario 下 hardest label pairs 的边界变化
* calibration curve
* top confusion pairs 的修正前后变化

---

# 我最想让你先做的一个小实验

为了避免再走大弯路，我建议先做一个**很轻量但判别力很强的 sanity check**：

## 在你当前最强 SLR-C 上，只加“scenario-conditioned class bias / threshold”

先别上完整 graph。

做：

[
z'_j = z_j + b_j(s)
]

或

[
\tau_j(x) = \tau_j^{base} + \Delta \tau_j(s)
]

也就是：

* scenario 只用来调每个类的 bias 或 threshold
* 不建 label graph
* 不加复杂传播

### 这个实验的意义

如果这一步都能涨，说明：

> scenario 确实应该进入 decision layer

那你再继续做 conditional relation graph 就很合理。

如果这一步完全不涨，再考虑 scenario 是否真的该做主线。

---

# 现在的明确建议

基于你刚说的“结构化 expert 其实也差不多试过”，我现在的建议会很明确：

## **停止继续在 feature decomposition 上内卷。**

把主线转成：

### **Scenario-conditioned decision learning**

也就是让 scenario 去条件化：

* class bias
* threshold
* label relation
* calibration

这条线最符合你目前所有正负结果给出的信号。

如果你愿意，我下一条直接把这个方向写成一份新的、可开做的**实验计划 + 方法设计 + ablation 表**。
