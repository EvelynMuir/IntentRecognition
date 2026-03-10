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
- 但会伤 `micro / samples`

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

### (4) 方法更偏向提升抽象 intent

把 intent 主观粗分成：

- `abstract`
    - CreativeUnique
    - CuriousAdventurousExcitingLife
    - EasyLife
    - EnjoyLife
    - FineDesignLearnArt-Culture
    - GoodParentEmoCloseChild
    - Happy
    - HardWorking
    - Harmony
    - Health
    - InLove
    - InspirOthrs
    - ManagableMakePlan
    - PassionAbSmthing
    - ShareFeelings
    - SocialLifeFriendship
    - SuccInOccupHavGdJob
    - WorkILike
- `concrete`
    - Attractive
    - BeatCompete
    - Communicate
    - FineDesignLearnArt-Arch
    - FineDesignLearnArt-Art
    - InLoveAnimal
    - NatBeauty
    - Playful
    - TchOthrs
    - ThngsInOrdr

baseline 上：

- abstract mean F1 = `41.93`
- concrete mean F1 = `53.26`

而 `scenario SLR + class-wise threshold`：

- abstract mean F1 = `48.69`
- concrete mean F1 = `55.93`

相对 baseline：

- abstract `+6.76`
- concrete `+2.67`

> 当前方法提升的核心更偏向 high-level、abstract intent perception，而不是普通的具体视觉类别识别。