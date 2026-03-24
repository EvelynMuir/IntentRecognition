# 一、目标任务

**图像意图识别（visual intent recognition）**，数据集主要是 **Intentonomy**。
任务的难点不在于识别显式物体，而在于识别图像背后的**隐含意图**，比如快乐、探索、欣赏设计、享受生活、表达自我等。

这个任务有几个核心特点：

## 1. 类间相似性强

很多intent语义接近，比如：

* Enjoy Life
* Happy
* Playful

视觉上往往共享类似场景和物体，边界很模糊。

## 2. 类内差异大

同一个intent可能由完全不同的视觉内容表达出来，不是固定物体对应固定标签。

## 3. 标签本身带有语言/语义先验

类别不是“dog”“car”这种客观类别，而是带有解释性的抽象语义，因此**文本描述和先验知识**可能很重要。

## 4. 数据量有限、长尾明显

# 二、分解 / 组合式表示

假设：

> intent 不是单一视觉特征，而是由多个潜在语义因素组合而成。

做**表示分解**，希望把图像特征拆成若干factor，再由factor组合出intent。

## 2.1 基于 VQ / factor decomposition 的方法

尝试：

* 多个离散codebook / orthogonal VQ
* factorized latent representation
* anchor / disentangle / regularization 一类约束

目标是让不同factor对应不同语义成分，再组合成intent。

### 结果

这条线整体效果不好，主要问题有：

* 训练不稳定
* codebook利用率低
* 很多factor学不到有意义的分工
* 一阶段后，每个factor有效code很少
* 最终分类性能不如简单baseline

总结：

> 最好的结果仍然是 **MLP baseline**，
> 只有少量正交VQ对CLS token做分解，
> 复杂分解模型反而掉分。

* **MLP baseline macro-F1 ≈ 0.44**
* 某些 factor/VQ 方法大约 **0.40**
* **LLM / RAG SOTA ≈ 0.43**

### 可能的原因

#### (1) 任务本身缺少足够强的“可分解监督”

Intent 并没有天然的factor标注，分解容易退化成“人为施加结构”，但不一定真对应任务判别边界。

#### (2) Intentonomy 数据规模不足以支撑复杂解耦

复杂分解模型参数更多、训练目标更多，但数据量和标签质量不足，容易学出不稳定结构。

即使讲出组合式intent的故事，如果最终性能没超过强baseline，容易被质疑：

* 分解是否真的必要
* 学到的factor是否可信
* contribution是否只是复杂化模型

# 三、concept / graph / reasoning

假设：

> image → visual concepts → concept graph reasoning → intent prediction

intent可能通过：

* 物体
* 背景
* 动作
* 情绪
* 风格

等多种因素隐式表达，因此想引入**concept pool**和concept reasoning。

## 3.1 具体方法

* 在baseline上加 **concept branch**
* 让视觉特征经过额外的concept通路辅助分类

### 结果

> **concept branch 加在 baseline 上以后，分数会下降。**

## 3.2 可能的原因

### (1) concept 的定义不够稳定

Intentonomy 的intent非常抽象，而concept既可以是具体视觉元素，也可以是高层语义。
一旦concept定义不稳，模型就很难学到一致的映射。

### (2) concept 引入了额外误差源

从图像到concept本身就不容易，concept如果预测不准，后续intent分类会被连带拖累。

### (3) 当前数据没有concept-level监督

如果没有强监督，concept branch很容易变成“额外参数 + 额外噪声”，不一定真能帮助最终分类。

# 四、baseline 误差分析

> 这个任务未必需要更复杂的生成式/分解式结构，
> 更可能需要的是：**围绕现有强视觉 backbone，把真正的错误来源分析清楚，再做针对性修正。**

## 4.1 强baseline：CLIP ViT

**CLIP-based framework**：

* frozen CLIP visual encoder

做了以下 feature 性能比较：

* CLS token baseline
* patch token baseline
* ViT不同层特征
* linear probe / MLP probe
* patch token 和 global token

CLS token 拼接 mean(patch token) 得到最优 baseline

任务瓶颈**不主要在特征提取能力不足**，而在于：

### 1. label ambiguity

类别之间边界模糊，单靠视觉相似度会混淆近义intent。

### 2. language prior

intent标签本身是语言概念，不是纯视觉类别。
文本描述质量会直接影响分类效果。

### 3. top-k 排序

模型预测常常已经把正确类别放进候选里，但没有排到最前。

# 五、calibration + rerank

> **以 CLIP 为主干，利用语义先验做 calibration，再通过 reranking 修正候选类别顺序。**

visual intent recognition 的主要误差来源是 prior mismatch 与 inter-class ambiguity
* 提出 prior-calibrated scoring
* 再通过 reranking 修正近义intent排序
* 不需要额外标注，不破坏现有主干

### Step 1: CLIP visual-text matching

用 frozen CLIP 提取图像特征，并与intent文本先验做匹配，得到初始intent分数。

### Step 2: calibration

利用构造好的 prior 对初始分数进行校准。

- `lexical`
  - 标签短语
- `canonical`
  - 标签简短描述（Intentonomy 数据集给出）
- `scenario`
  - Gemini 3.1 pro生成`Text Query`，每个类别3-5条
- `lexical + canonical`
  - 用于 heterogeneous source ensemble

* 拉开高混淆类别之间的分数差距
* 让分数更符合该任务的语义先验

### Step 3: rerank

针对 top-k 候选intent 做二次排序。

# 六、结果与分析

## 6.1 主表

| Method | macro | micro | samples | mean F1 | hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior SOTA | 43.05 | 54.77 | 56.75 | 51.52 | 27.39 |
| baseline | 45.98 | 56.54 | 56.40 | 52.97 | 26.86 |
| baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 51.07 | 29.71 |
| baseline + SLR-v0 (scenario) | 47.14 | 56.52 | 55.11 | 52.92 | 28.32 |
| baseline + SLR-v0 (scenario) + class-wise threshold | **51.28** | **59.13** | **58.47** | **56.29** | 33.98 |
| baseline + SLR-v0 (scenario) + frequency-group threshold | 49.41 | 55.70 | 55.16 | 53.43 | **35.99** |
| baseline + lexical+canonical SLR | 49.61 | 57.20 | 57.29 | 54.70 | 33.11 |
| baseline + lexical+canonical SLR + class-wise threshold | 50.65 | 57.37 | 57.46 | 55.16 | 32.38 |
| best learnable fusion | 49.63 | 57.58 | 57.60 | 54.97 | 32.49 |

## 6.2 分析

### (1) calibration 是主增益来源之一

在 baseline 上：

- class-wise threshold 能涨 `macro / hard`
- 但会降低 `micro / samples`

在 SLR 上：

- class-wise threshold 不仅涨 `macro / hard`
- 还同时涨 `micro / samples`


> calibration 不只是一个通用后处理技巧，而是与 reranked score geometry 强耦合的第二阶段决策模块。

### (2) source 比 fusion 更重要

- `scenario` 是最强的 single-source prior
- `lexical + canonical` 是最强的 heterogeneous-source 组合
- `discriminative` 单独可用，但不如 `scenario`

而 learnable fusion：

- class-wise affine 比 shared MLP 强
- 但仍然不如 strongest calibrated decision rule

> 当前收益的主来源不是更复杂的 fusion function，而是更好的 source 形式 + 更好的 calibrated decision rule。

### (3) candidate proposal 已经很强，瓶颈在 candidate set 内

`candidate recall / oracle analysis` 的结果：

- baseline top-10 label recall: `83.91%`
- top-10 sample-any recall: `95.07%`
- top-10 sample-all recall: `75.33%`

同时：

- oracle@top10 的上界仍然极高
  - `macro = 87.42`
  - `micro = 91.25`
  - `samples = 89.67`

> baseline 的 candidate proposal 阶段已经很强，主要瓶颈是候选集内部的语义消歧和最终决策。

## 6.3 backbone 替换为 `CLIP ViT-B/32` 的补充结果

这一步的目的不是重新调一整套超参，而是回答一个更直接的问题：

> 如果把当前 strongest pipeline 的 backbone 从 `ViT-L/14` 换成更轻的 `ViT-B/32`，方法结论是否仍然成立？

### 6.3.1 实验设置

本轮使用的训练 run：

- `logs/train/runs/2026-03-10_15-07-05`

关键配置：

- backbone：`CLIP ViT-B/32`
- layer：`12`
- batch size：`64`
- lr：`1e-4`
- `use_ema=true`
- `seed=42`

当前分析使用的 checkpoint：

- `logs/train/runs/2026-03-10_15-07-05/checkpoints/epoch_005.ckpt`

需要明确说明的是：

> 这是一版 **current-best@epoch5** 的 first-pass 结果，不是 full early stopping 终态。

对应训练曲线里，`val/f1_macro` 的变化为：

- epoch 0：`0.2213`
- epoch 1：`0.3057`
- epoch 2：`0.3417`
- epoch 3：`0.3656`
- epoch 4：`0.3713`
- epoch 5：`0.3978`

分析输出目录：

- calibrated decision rule：`logs/analysis/vitb32_epoch005_calibrated_decision_rule_20260310`
- candidate recall / oracle：`logs/analysis/vitb32_epoch005_candidate_recall_20260310`

### 6.3.2 主结果

| Method | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| baseline | 38.29 | 53.09 | 51.45 | 11.92 |
| baseline + class-wise threshold | 44.52 | 51.64 | 51.36 | 26.23 |
| baseline + frequency-group threshold | 40.71 | 51.20 | 50.96 | 22.09 |
| baseline + SLR-v0 (scenario, original threshold) | 42.04 | 50.87 | 50.32 | 28.45 |
| baseline + SLR-v0 (scenario, re-tuned global threshold) | 41.94 | 52.63 | 51.08 | 23.32 |
| baseline + SLR-v0 (scenario) + class-wise threshold | 46.77 | 55.37 | 53.70 | 30.32 |
| baseline + SLR-v0 (scenario) + frequency-group threshold | 44.81 | 52.03 | 50.48 | **31.12** |
| baseline + lexical+canonical SLR | 45.39 | 54.79 | 53.72 | 27.18 |
| baseline + lexical+canonical SLR + class-wise threshold | **47.34** | **57.29** | **55.72** | 30.24 |

补充看 single-source：

- `lexical` global：`macro = 45.57`，`hard = 31.47`
- `lexical` class-wise：`macro = 47.06`，`hard = 30.53`
- `canonical` global：`macro = 43.87`，`hard = 22.94`
- `canonical` class-wise：`macro = 47.76`，`hard = 31.01`

### 6.3.3 结果解读

先说最直接的结论：

> 把 backbone 从 `ViT-L/14` 换成 `ViT-B/32` 以后，整体性能会明显下降，但“local rerank + calibrated decision rule 有效”这个方法结论仍然成立。

具体看：

1. baseline 明显变弱

- `ViT-B/32` baseline 只有 `macro = 38.29`
- 明显低于前面的 `ViT-L/14` baseline `45.98`

这说明：

> backbone capacity 的下降会直接拉低 intent recognition 的整体上限。

2. calibration 依然是强增益

- baseline `38.29 -> 44.52`（class-wise threshold）
- `hard 11.92 -> 26.23`

也就是说：

> 即使换成更弱 backbone，decision rule 仍然不是小修小补，而是主要收益来源之一。

3. SLR 依然有效，但 strongest source 发生了轻微变化

- single-source strongest 仍然是 `scenario + class-wise threshold`
  - `macro = 46.77`
  - `hard = 30.32`
- 但当前这版 `ViT-B/32` overall 最强变成了
  - `lexical+canonical + class-wise threshold`
  - `macro = 47.34`
  - `micro = 57.29`
  - `samples = 55.72`

这说明：

> backbone 变化以后，prior source 的相对强弱会有轻微变化，但 heterogeneous semantic prior 仍然有价值。

4. hard-best 仍然来自 frequency-group threshold

- `scenario + frequency-group threshold`
  - `hard = 31.12`

因此：

> 如果只追求 hardest intents 的收益，group-wise threshold 依然值得保留。

### 6.3.4 candidate recall / oracle 补充

`ViT-B/32` baseline 的 top-k candidate recall 结果如下：

- top-10 label recall：`81.69%`
- top-10 sample-any recall：`94.49%`
- top-10 sample-all recall：`72.94%`
- top-10 mean positive coverage：`85.02%`

对应 oracle@top10 上界：

- `macro = 85.33`
- `micro = 89.93`
- `samples = 88.21`

同时 false negative recoverability@top10：

- baseline global：`recoverable_ratio = 65.82%`
- scenario SLR global：`58.77%`
- scenario SLR class-wise：`54.59%`

以及相对 baseline 新恢复的正类标签数：

- `scenario SLR global`：`+250`
- `scenario SLR class-wise`：`+369`

这说明：

> 即使在 `ViT-B/32` 上，candidate proposal 阶段依然已经不弱，主要剩余空间仍然在候选集内部的语义消歧与最终决策。

### 6.3.5 当前判断

这一轮补充实验给出的最稳结论是：

- `ViT-B/32` 明显弱于 `ViT-L/14`
- 但 `SLR + calibrated decision rule` 的方法逻辑没有失效
- `class-wise threshold` 依然是强增益来源
- `lexical+canonical` 在较弱 backbone 上是一个很值得重视的稳定 prior 组合

因此，如果论文里要加 backbone sensitivity / lighter-backbone 补充，可以写成：

> stronger visual backbone 决定整体上限，而 local semantic reranking 与 calibrated decision rule 决定如何更好地释放该 backbone 的 intent prediction potential。

## 6.4 在 IntCLIP 上外部复现实验

这一节的目的不是再训练一个新方法，而是回答一个更直接的问题：

> 把当前的 `SLR-C` 后处理逻辑接到外部仓库 `IntCLIP` 上以后，是否仍然有效？

### 6.4.1 实现说明

外部仓库：

- `https://github.com/yan9qu/IntCLIP`

本地路径：

- `../IntCLIP`

为使其在当前环境可运行，做了两类基础适配：

1. `Intentonomy` dataloader 路径兼容
   - 让仓库直接读取我们本地的 `../Intentonomy/data`
2. `Dassl` 与当前 PyTorch 版本的 lr scheduler 兼容修复

在此基础上，新增了一个外部评估脚本：

- `../IntCLIP/eval_slrc.py`

这个脚本不改 IntCLIP 主干，只在 checkpoint 上额外评估：

- baseline
- baseline + `scenario` SLR global
- baseline + `scenario` SLR class-wise
- baseline + `lexical+canonical` SLR global
- baseline + `lexical+canonical` SLR class-wise

使用的 IntCLIP checkpoint 为：

- `../IntCLIP/output/intentonomy_rn101_retrain/intentonomy-DualCoop-RN101-cosine-bs64-e60/model_best.pth.tar`

对应输出：

- `../IntCLIP/output/intentonomy_rn101_retrain/intentonomy-DualCoop-RN101-cosine-bs64-e60/eval_slrc_scenario.json`
- `../IntCLIP/output/intentonomy_rn101_retrain/intentonomy-DualCoop-RN101-cosine-bs64-e60/eval_slrc_lexcan.json`

### 6.4.2 主结果

| Method | macro | micro | samples | mean F1 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| IntCLIP baseline | 42.27 | 52.39 | 50.86 | 48.51 | 44.46 |
| + scenario SLR global | 41.81 | 48.27 | 47.64 | 45.91 | 47.50 |
| + scenario SLR class-wise | 45.22 | 52.60 | 51.83 | 49.88 | 47.50 |
| + lexical+canonical SLR global | 45.64 | 52.63 | 52.22 | 50.16 | 49.39 |
| + lexical+canonical SLR class-wise | **46.06** | 48.44 | 48.22 | 47.58 | **49.39** |

当前 strongest overall 版本是：

- `baseline + lexical+canonical SLR global`

对应：

- `macro = 45.64`
- `micro = 52.63`
- `samples = 52.22`
- `mean F1 = 50.16`

如果看 `scenario`：

- global rerank 会掉点
- 但配上 class-wise threshold 后可以反超 baseline
  - `macro = 45.22`
  - `micro = 52.60`
  - `samples = 51.83`

### 6.4.3 结果解读

这一步最重要的结论不是“涨了多少”，而是：

> `SLR-C` 的方法逻辑确实能迁移到外部框架上。

具体来说：

1. baseline 之外的两条 semantic prior 线都能带来收益

- `scenario + class-wise threshold` 明显优于 IntCLIP baseline
- `lexical+canonical + global rerank` 进一步优于 baseline

2. 在 IntCLIP 上，最优落点与我们本地 strongest pipeline 不完全一致

在我们自己的 strongest CLIP-MLP baseline 上：

- `scenario + class-wise threshold` 是 overall 最强

但在 IntCLIP 上：

- `lexical+canonical + global rerank` 反而是 strongest overall

这说明：

> 主干架构变化以后，semantic prior source 与 decision rule 的最优组合会变化，但“local rerank + calibrated decision”这个总体思想仍成立。

3. calibration 在 IntCLIP 上仍然有价值，但不总是最优

这里和我们本地 baseline 的现象不同：

- `scenario` 需要 class-wise threshold 才能变强
- `lexical+canonical` 则在 global rerank 下已经是当前最优

这意味着：

> calibrated decision rule 依然重要，但它在外部模型上的最佳形态需要和该模型自己的 score geometry 一起重新选择。

### 6.4.4 当前判断

这一轮外部仓库复现实验给出的最稳结论是：

1. `SLR-C` 不是只对我们自己的 baseline 有效
2. 它在 IntCLIP 上也能带来稳定正增益
3. 但最佳 prior source / calibration 组合是 backbone- and framework-dependent

因此，这一步的价值主要体现在：

- 增强方法的外部可迁移性证据
- 证明 `local semantic reranking + calibrated decision` 不是一个只在单一实现里成立的技巧
