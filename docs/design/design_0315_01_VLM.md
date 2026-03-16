## 核心实验方案 (Methodology)

### 1. 离线多模态数据生成 (Rationale Generation)

这是整个方法的信息源头。为了保证训练效率，我们**不**在训练时动态调用大模型，而是提前构建一个“推理特征库”。

* **VLM 选择：** 使用开源且性能强劲的模型（Qwen2.5-VL-7B-Instruct，使用vllm框架，模型权重在/home/evelynmuir/lambda/hf-models下有）。
* **Prompt 工程设计 (至关重要)：** 我们需要构建一个“四段式” Prompt，强迫模型吐出结构化的高质量推理：
1. *Fact (客观事实)：* 描述图中的人物、动作和核心物体。
2. *Context (环境上下文)：* 描述背景氛围（如：灯光暗淡的室内、喧闹的街道）。
3. *Reasoning (意图推理)：* 结合 1 和 2，推断人物的心理意图。
4. *Contrast (反事实对比)：* **（Novelty所在）** 明确指出“为什么不是 [易混淆的近邻意图]”。

```
System Prompt
"""
You are an expert human behavioral analyst and computer vision specialist. Your task is to analyze an image and provide a highly logical, structured reasoning chain (Rationale) that explains the underlying human intent. You must strictly follow the requested 3-step format and avoid any conversational filler.
"""

User Prompt
# 假设 y_true 是一个字符串，如 "Socializing, Entertainment, and Eating"
# 假设 y_confuse 是一个假阳性标签，如 "Working"

prompt = f"""
Given this image, the ground-truth human intents are strictly defined as the concurrent occurrence of: **[{y_true}]**.

Please carefully observe the image and generate a structured reasoning report strictly following these 3 steps:

**Step 1: Visual Evidence**
Describe the explicit physical actions, key objects, body language, or facial expressions in the image that strongly support the presence of **these concurrent intents ([{y_true}])**. Be specific about what you see and how the visual cues correspond to these multiple intents.

**Step 2: Contextual Bridging**
Explain how the background, environment, or the relationship between the subjects logically connects with the visual cues from Step 1 to reveal why these multiple psychological motivations **([{y_true}]) naturally co-exist** in this specific scene.

**Step 3: Counterfactual Disambiguation**
In a purely visual context, a machine learning model might easily misclassify this image as also containing the intent of **[{y_confuse}]**. Point out the specific visual clues that are definitively *missing*, or the contradictory details that are *present*, which prove the intent of **[{y_confuse}] is absolutely NOT happening** here.
"""
```

{y_true} 的构造： 将该样本的所有正类标签用逗号和 "and" 连接起来。

{y_confuse} 的构造： 易混淆标签是 Baseline 预测分最高的那一个“假阳性 (False Positive)”标签。


* **特征预提取：** 使用一个预训练的强大 Text Encoder（CLIP-Text），将生成的每一段 Rationale 编码为固定维度的文本向量 $t_i$。将其保存为 `.pt` 或 `.npy` 格式，与图像数据集对齐。

### 2. 网络结构设计 (Architecture)

推理阶段我们只需要视觉网络，因此这套架构采用“Teacher-Student”非对称设计。

* **文本分支 (Teacher - 冻结)：** 仅在训练阶段提供监督信号。输入提前提取好的文本推理向量 $t_i$。
* **视觉分支 (Student - 可训练)：** 你的主干网络（如 ViT-B/16 或 Swin-Transformer）。图像 $x_i$ 通过视觉 Backbone 提取出全局视觉特征 $v_{cls}$。
* **对齐投影头 (Alignment Head)：** 在视觉特征后接一个两层的 MLP（带 ReLU），将视觉特征 $v_{cls}$ 映射到与文本特征相同的维度，得到投影特征 $v_i$。

### 3. 损失函数定义 (Loss Design)

为了解决 Hard 样本和标签边界模糊问题，我们需要任务损失与蒸馏损失双管齐下。

* **任务分类损失 (Task Loss)：** Optimized Asymmetric Loss.


* **隐式推理对齐损失 (InfoNCE Alignment Loss)：**
不使用简单的 MSE，而是使用对比学习损失。让当前图像的视觉特征 $v_i$ 靠近对应的文本推理特征 $t_i$，并远离 Batch 内其他图像的文本推理特征。

$$\mathcal{L}_{align} = -\log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)}$$



*(其中 $\text{sim}(\cdot, \cdot)$ 为余弦相似度，$\tau$ 为温度系数。)*
* **总损失：**

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{align}$$



---

## 核心实验清单 (Experiment Checklist)

为了让 Reviewer 信服，你需要按照以下顺序推进实验，逐步证明你方法的有效性。

### Phase 1: 主实验与基线对比 (Main Results)

* [ ] **Baseline 复现：** 仅使用视觉 Backbone + $\mathcal{L}_{cls}$，记录 Overall、Macro 和 Hard Intent 的准确率。（这是你进步的标尺）。
* [ ] **SOTA 对比：** 对比 Intentonomy 榜单上或你之前跑过的其他结构先验方法（如 SLR-C）。
* [ ] **Proposed Method 测试：** 跑通完整的 $\mathcal{L}_{total}$ 框架，重点观察 Macro 和 Hard 指标是否如预期般显著拉升。

### Phase 2: 核心消融实验 (Ablation Studies)

这一步是防守 Reviewer 攻击的关键。你需要证明是“推理蒸馏”起了作用，而不是别的 Trick。

* [ ] **Ablation 1: 损失权重敏感度：** 调节 $\lambda \in \{0.1, 0.5, 1.0, 2.0\}$，证明对齐损失 $\mathcal{L}_{align}$ 的增益，并找到最佳平衡点。
* [ ] **Ablation 2: 对齐方式的对比：** 替换 $\mathcal{L}_{align}$ 的计算方式。对比 InfoNCE (对比学习)、MSE (直接均方误差) 和 Cosine (仅余弦相似度)。预期结果：InfoNCE 效果最好，因为它能在特征空间拉开近邻类的距离。
* [ ] **Ablation 3: 知识来源的质量 (极为关键)：**
* *Setting A (无常识)：* 把对齐目标的文本换成“简单的类别标签名”（如 "This is an image of socializing"）。
* *Setting B (浅层描述)：* 把文本换成 VLM 生成的简单图像描述（Caption，无推理过程）。
* *Setting C (Ours)：* 使用完整的 Rationale（包含推理与反事实对比）。
* *(预期：Setting C 远好于 A 和 B，完美证明“推理过程”比“简单描述”更解决问题。)*



### Phase 3: 定性与可视化分析 (Qualitative Analysis)

数据指标能证明你的方法有效，但良好的可视化能让你的故事更丰满。

* [ ] **t-SNE 特征分布可视化：** 提取 Baseline 和你方法的视觉特征（特别是 Hard 样本密集的那些近邻类别），做 t-SNE 降维对比。预期你的方法能展现出更清晰的类间边界，证明“语义混淆”被缓解。
* [ ] **Grad-CAM 注意力热图对比：** 找几张极度模糊的 Hard 样本图。对比 Baseline 和你方法的 Attention Map。预期你的方法（因为学习了文本的上下文常识）能将注意力从“单一动作/物体”扩散到“全局上下文”，从而做出正确判断。
* [ ] **Bad Case 分析：** 诚实地挑几个使用了你的方法依然分错的样本。探讨是不是 VLM 生成的 Rationale 本身就出现了幻觉（Hallucination），体现你思考的深度。