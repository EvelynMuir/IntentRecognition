### 软标签引导的监督对比学习 (Soft-SupCon)

目前你的蒸馏只发生在**逻辑层（Logit Level）**，Student 的视觉特征空间（Feature Space）依然是在被 Loss 自由放养。我们之前放弃特征对齐，是因为跨模态（图与文）强行对齐会导致崩盘。但是，**在同一模态（图与图）之间做对比学习**是极其稳定且有收益的！

* **核心痛点：** 意图分类的难点在于类间差异小（比如“休闲”和“社交”长得很像）。传统的对比学习用 Hard Label 区分正负样本，这在多标签且有噪声的数据上会把原本应该靠近的样本推开。
* **新颖解法：** 利用你的 Text-Only Teacher 给出高质量软分布（Soft Distribution $P_{tea}$），在 Student 的视觉特征 $v_i$ 之间做一个**语义感知的对比学习（Semantic-Aware Contrastive Learning）**。
* **具体做法：**
在每个 Batch 内，计算任意两张图片 Teacher 软标签的相似度：$S_{i,j} = \text{sim}(P_{tea}^{(i)}, P_{tea}^{(j)})$。
利用这个相似度作为连续的权重，来指导 Student 视觉特征的对比损失：

$$\mathcal{L}_{Soft-SupCon} = - \sum_{i \neq j} S_{i,j} \log \frac{\exp(v_i \cdot v_j / \tau)}{\sum_{k \neq i} \exp(v_i \cdot v_k / \tau)}$$