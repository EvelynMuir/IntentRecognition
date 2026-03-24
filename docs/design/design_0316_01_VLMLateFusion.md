# 实验计划：VLM Late Fusion

## 1. 目标

前一轮 VLM 方向的主要问题不是视觉 Backbone 弱，而是：

1. 把长文本当作 teacher embedding 做全局对齐，信号很容易被 projector 吞掉
2. `caption / rationale / step1 / step1+step2` 在全局对齐框架下很难真正改写分类边界
3. `Step 3` 的反事实信息没有稳定转化为 hardest confusion 的抑制

因此，本轮不再继续做 embedding alignment，而是改成一个更直接的思路：

> 不改视觉 Backbone，不加 teacher-side 对齐损失，让文本只在最终 logit 层做 late fusion residual。

---

## 2. 方法

### 2.1 基础结构

保持视觉主干与 baseline 完全一致：

[
v_i = \text{Backbone}(x_i)
]

基础分类 logits 仍然来自已有 baseline：

[
\text{Logits}_{base} = W_{cls} v_i
]

这里的 `Backbone` 与 `Logits_base` 在实现上直接复用缓存：

- train / val / test 的 `*_clip.npz` 作为图像特征
- train / val / test 的 `*_base.npz` 作为基础 logits

也就是说，这一轮不再重训视觉模型，而是在 cached feature space 里验证：

> 文本 residual 本身能否作为一个有效的后期修正器。

### 2.2 文本残差

将视觉特征投影到文本空间：

[
q_i = v_i W_{proj}
]

然后分别与三种文本特征做余弦相似度：

- `T_vis`：视觉证据文本（对应 Step 1）
- `T_ctx`：上下文推理文本（对应 Step 2）
- `T_neg`：反事实排斥文本（对应 Step 3）

定义：

[
S_{pos} = \text{sim}(q_i, T_{vis})
]

[
S_{ctx} = \text{sim}(q_i, T_{ctx})
]

[
S_{neg} = \text{sim}(q_i, T_{neg})
]

最终 late fusion logits：

[
\text{Logits}_{final}
=
\text{Logits}_{base}
 \alpha_1 S_{pos}
 \alpha_2 S_{ctx}
- \alpha_3 S_{neg}
]

其中：

- `alpha_1 / alpha_2 / alpha_3` 为可学习标量
- `S_pos / S_ctx` 作为全局奖励
- `S_neg` 只对 confuse class 做惩罚更合理

---

## 3. 文本来源

### 3.1 train

train split 使用离线生成的 rationale：

- `y_true`：使用 GT 多标签拼接
- `y_confuse`：使用 baseline top-1 false positive

### 3.2 val / test

为了避免 GT 泄漏，val/test 的文本构造不使用 GT，而是：

- `label_source = baseline_pred`
- `y_true`：用 baseline 预测标签集合构造
- `y_confuse`：用最高分但不在预测标签集合里的类构造

这让 late fusion 更接近真实 inference：

> 文本 residual 不是 oracle correction，而是建立在 baseline 当前判断上的弱指导。

---

## 4. 特征编码

这轮不再使用 `CLIP-text` 作为主文本编码器，原因是：

1. `CLIP-text` 只接收 `77` tokens
2. 长 rationale 会被截断
3. `Step1 / Step2` 即使文本不同，进入 CLIP text 空间后仍过于相似

因此改用：

> `BAAI/bge-large-en-v1.5`

并显式保留：

- `step1_features`
- `step2_features`
- `step3_features`
- `pos_features = step1 + step2`
- `neg_features = step3`

---

## 5. 实验对比

### 5.1 主对比

至少比较：

1. baseline
2. baseline + `T_pos only`
3. baseline + late fusion (`T_vis + T_ctx - T_neg`)
4. strongest reference `SLR-C`

### 5.2 关注指标

- Macro
- Micro
- Samples
- mAP
- Hard

重点仍然看：

1. Hard 是否能稳定高于 baseline
2. Macro 是否有正式正增益
3. 与 `SLR-C` 的差距是否缩小

### 5.3 关键诊断

1. learned `alpha_1 / alpha_2 / alpha_3`
2. `Step1 only` vs `Step1+Step2`
3. `caption` vs `rationale`
4. `baseline_pred` 文本是否过噪

---

## 6. 判停标准

这条线应继续的信号：

1. late fusion 在 Hard 上明显优于 baseline
2. Macro 有正式正增益
3. `alpha` 学到的权重有清晰方向，例如：
   - `alpha_pos > 0`
   - `alpha_ctx > 0`
   - `alpha_neg > 0`

应止损的信号：

1. late fusion 仍然低于 baseline
2. Hard 持续下降
3. learned `alpha_ctx` 接近 `0` 或转负，说明 Step 2 不提供稳定帮助
4. `alpha_neg` 增大但 hardest confusion 仍不改善

---

## 7. 最小可执行版本

当前仓库里这条线的最小版本定义为：

1. 复用 cached image features 和 cached base logits
2. 复用 train rationale 与 val/test baseline-pred rationale
3. 使用 `BGE` 编码三段文本
4. 使用一个轻量 `W_proj + alpha` 做 late fusion residual
5. 先跑 full train / val / test，再决定是否继续扩展

一句话总结：

> `VLMLateFusion` 不是再让 teacher 改写视觉表征，而是直接让文本在 decision layer 给 baseline logits 加减分，从而验证文本 residual 是否比全局对齐更有效。  
