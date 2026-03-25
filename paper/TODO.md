### 4.1 Experimental Setup (实验设置)
* **Datasets:** 重点介绍 Intentonomy，说明它有丰富的 $\omega_i$ 标注（0, 0.33, 0.66, 1）。
* **Evaluation Metrics:** 列出 Macro, Micro, Samples, mAP。**必须专门用一两句话强调 `Hard` 指标的定义和重要性**。
* **Implementation Details:** 交代视觉 Backbone（默认网络和 ResNet101）、ASL 的参数、Teacher 的文本编码器（BGE）、VLM 的型号（Qwen-2.5-VL）、SLR-C 的 $K$ 值设定（Top-10）。强调 UTD 在推理期是 **Zero Extra Cost（零额外开销）**。

### 4.2 Comparison with State-of-the-Arts
* **对比方法：** 把你之前提到的 PIP-Net, CPAD, HLEG, LabCR, IntCLIP, IntentMLM 全部列上去。
* **分级展示 (The Grand Flex)：** 在表格的最下方，分两行展示你的方法：
  * `Ours (UTD only)`: 强调在**不增加任何推理开销**的前提下，已经全面超越了那些笨重的多模态大模型。
  * `Ours (UTD + SLR-C)`: 祭出你那组无敌的数据（**Samples 60.30, Hard 36.43**），告诉审稿人什么是这个数据集的绝对天花板。

### 4.3 Ablation Studies (消融实验 —— 全文的灵魂)

* **Ablation 1: The Magic of Counterfactual Rationale (VLM 推理链消融)**
  * *实验设计：* 对比 `step1_only` vs `step1_step2` vs `Full (1+2+3)`。
  * *故事线：* 这段是整篇论文的学术高光！必须详细阐述我们之前讨论的现象：仅引入上下文（Step 2）会引发语义发散，导致大盘指标跳水但 Hard 提升；而引入反事实消歧（Step 3）则像一把剪刀，完美收束了边界，实现了全局指标的大满贯。
* **Ablation 2: Why Gated Distillation? (门控机制与容量代沟)**
  * *实验设计：* 对比 `Baseline` vs `Standard KD` vs `Dynamic Gated KD`。
  * *故事线：* 结合 ResNet101 的那组数据。论证 Standard KD 存在“师生容量代沟（Capacity Gap）”，会破坏 Easy 样本的特征流形；而 $\omega_i$ 门控实现了“按需喂饭”，在保护客观大盘的同时，精准纠偏主观噪声。
* **Ablation 3: Modality Collapse Avoidance (模态消融)**
  * *实验设计：* 对比 `Image+Text Teacher` 和 `Text-Only Oracle Teacher`。
  * *故事线：* 证明在主观任务中，强行跨模态特征融合是毒药。纯文本的特征流形才能提供最纯净的逻辑暗知识。
* **Ablation 4: Heterogeneous Priors in SLR-C (异构先验消融)**
  * *实验设计：* 在局部重排中，对比 `Lexical`, `Canonical`, `Scenario`, 以及 `Ensemble`。
  * *故事线：* 证明 Scenario 效果最好，因为它能将抽象心理动机具象化为视觉元素，完美缝合了模态鸿沟。

### 4.4 Robustness and Generalization (鲁棒性与泛化性)
证明你的方法不是在 Intentonomy 上过拟合的 Trick。
* **Backbone Independence (跨骨干网络)：** 放上 ResNet101 的数据，证明即便是极弱的视觉网络，UTD 依然能带来极其稳定的巨幅提升。
* **Cross-Dataset Generalization (EMOTIC 实验)：**
  * *重点提示：* 在这里坦诚地讨论 EMOTIC 上的“模态断层”现象（即情绪识别需要极微观的视觉特征，VLM 容易过度自信）。展示通过调节 KD 温度 $\tau$ 和权重 $\lambda$ 后，模型依然能取得超越 Baseline 的收益。这会显得你极其客观、严谨。

### 4.5 Qualitative Analysis (定性可视化)
用一张极具视觉冲击力的图（Figure 3/4），彻底征服审稿人。
* **选两张经典的 Hard 图片：**
  * 左边放图片和真实 Soft Labels。
  * 中间放 Baseline 的错误预测（False Positives）。
  * 右边放 VLM 生成的三段式 Rationale（重点高亮 Step 3 里的那句反事实推论）。
  * 最右边放 UTD + SLR-C 纠正后的最终精准预测。
* **目的：** 让审稿人直观地感受到：“哇，这个文本 Teacher 真的是在教模型做逻辑推理，而不是瞎猜。”