## 方案设计：基于分歧先验与特权信息的鲁棒蒸馏 (Noise-Robust Privileged Distillation)

**核心动机 (Motivation)：** Intentonomy 数据集中的 Hard 样本不仅来源于视觉可判别性不足，更来源于人类标注者在缺乏上下文时的主观臆断和错误。强制视觉模型拟合这些带有高分歧（如 1/3 专家同意）的硬标签，会导致严重的噪声过拟合（Noise Memorization）和梯度冲突。我们提出引入 VLM 作为“常识判别器”，通过动态蒸馏来阻断这种噪声传播。

### Phase 1: 训练多模态“上帝视角”教师网络 (Oracle Teacher)

不要让视觉特征去痛苦地对齐文本，而是让它们在特征层面直接结合，培养一个性能天花板极高的 Teacher。

* **输入特征：** * 离线提取的图像视觉特征 $v_i$（来自固定的 Vision Backbone）。
* 离线提取的 VLM Rationale 文本特征 $T_{rationale}$（来自 BGE 或 Sentence-BERT，包含视觉事实、上下文逻辑和反事实排斥）。


* **网络结构：** 一个极其轻量的多模态融合头（MLP）。将 $v_i$ 与 $T_{rationale}$ 拼接后，输出多标签的分类 Logits。
* **训练目标：** 使用标准的二元交叉熵损失（Optimized Asymmetric Loss）拟合 Binarized 真实标签。
* **物理意义：** Teacher 凭借 VLM 提供的强大常识（特权信息），能够轻易绕过视觉上的模棱两可，学到最接近真实意图的 Logit 分布。它是我们后续用来“纠错”的标尺。

### Phase 2: 人类分歧引导的动态蒸馏 (Uncertainty-Guided Student Distillation)

这是论文的核心 Novelty。训练真正的最终视觉模型（Student）时，输入**仅包含图像**。我们利用数据集自带的“专家一致性”作为置信度门控。

* **定义置信度权重 $\omega_i$：** 根据人工标注的分歧度（1/3, 2/3, 1），定义当前样本的置信度 $\omega_i \in [0, 1]$。例如，全票通过时 $\omega_i = 1.0$，分歧极大时 $\omega_i = 0.33$。
* **动态鲁棒损失函数 (Dynamic Robust Loss)：**

$$\mathcal{L}_{total} = \omega_i \cdot \text{Optimized Asymmetric}(Logits_{stu}, Y_{binarized}) + (1 - \omega_i) \cdot \lambda \cdot \text{KL\_Div}(P_{stu}, P_{tea})$$



*(其中 $P_{stu}$ 和 $P_{tea}$ 是经过带温度系数 Sigmoid/Softmax 平滑后的概率分布。)*
* **机制解析：**
* **当专家达成共识 ($\omega_i = 1$)：** 模型主要听从 Binarized 标签（Optimized Asymmetric），确保基础的 Macro 性能和 Easy 样本的绝对正确率。
* **当专家分歧极大/标注大概率错误 ($\omega_i \to 0.33$)：** 模型自动降低对可疑硬标签的信任，转而通过 KL 散度去学习 Teacher 凭借 VLM 常识推导出的“软分布”。这彻底斩断了模型死记硬背错误标签的路径。



---

## 核心实验清单 (Experiment Checklist)

为了让这套故事逻辑无懈可击，你需要按顺序完成以下实验，每一步都在向审稿人证明你的逻辑。

### 1. 验证 Teacher 的“纠错”上限 (Oracle Upper Bound)

这步是后续所有工作的基础，证明你的 BGE 特征确实含有降维打击的常识。

* [ ] **Train Oracle Teacher：** 跑通 Phase 1，记录 Teacher 模型的 Macro、Micro 和 Hard 指标。
* [ ] **预期结果：** Teacher 的指标应该**大幅超越**纯视觉 Baseline。
* [ ] **定性分析 (The "Aha" Moment)：** 挑选几张你肉眼发现“人工明显标错了”的图片，将它们输入给训练好的 Teacher。截图并记录 Teacher 的输出概率。**如果 Teacher 给出的高概率意图比人工硬标签更合理，务必把它保留下来作为论文的图表！** 这将是整篇论文最具说服力的证据。

### 2. 主实验与消融验证 (Main Results & Ablation)

证明你的动态蒸馏机制不仅保底，而且精准解决了 Hard 样本。

* [ ] **Baseline：** 纯视觉 Backbone + Optimized Asymmetric + Binarized Label。
* [ ] **Ablation 1 (Standard KD)：** 视觉 Backbone + 固定权重的 KL 蒸馏（不使用 $\omega_i$ 门控，强制所有样本以固定比例向 Teacher 学习）。观察是否会因为矫枉过正而影响 Easy 样本。
* [ ] **Ours (Dynamic Gated KD)：** 跑通 Phase 2 完整的动态损失函数。
* [ ] **预期结果：** Ours 方案在保证 Micro/Samples 等全局准确率不掉（或微涨）的前提下，显著拉升 Hard 和 Macro 指标。

### 3. 数据集切片深度分析 (Slice Analysis on Noisy Data)

为了彻底坐实“噪声鲁棒”的 Story，我们需要把测试集切开来看。

* [ ] **按专家分歧度分组测试：** 将测试集按照 `Agreement = 1`，`Agreement = 2/3`，`Agreement = 1/3` 分为三个子集。
* [ ] **对比 Baseline 与 Ours 在各子集的表现：**
* [ ] **预期结果：** 在 `Agreement = 1` 的纯净子集上，Ours 与 Baseline 表现相当（保底成功）；但在 `Agreement = 1/3` 的极度嘈杂/困难子集上，Ours 的指标应实现对 Baseline 的碾压。这完美证明了动态门控真正起到了“去伪存真”的作用。

## 统一口径
1. backbone: CLIP ViT-L/14
2. python环境使用conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda