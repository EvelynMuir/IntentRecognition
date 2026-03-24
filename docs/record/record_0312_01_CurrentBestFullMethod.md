# 总述：当前最优完整方法

## 0. 文档目的

这份文档不是某一次单独实验的日志，而是对当前整条方法线的**完整技术总结**。

目标是把下面四个阶段串成一个完整 pipeline：

1. `baseline`
2. `SLR-C`
3. `focused verifier`
4. `v2 best method`

并明确回答：

> 当前最推荐的完整方法到底是什么？  
> 每个模块具体怎么做？  
> 为什么最后会收敛到现在这个版本？

---

## 1. 一句话结论

当前最推荐的完整方法是：

> **Frozen CLIP ViT-L/14 baseline + scenario SLR-C + data-driven comparative evidence verification v2 (`hard_negative_diff + pairwise compare + margin-aware gate + add fusion`) + class-wise threshold**

如果只看最终推荐的 validation-best 完整系统，它的 test 指标为：

- Macro F1: `51.73`
- Micro F1: `59.22`
- Samples F1: `58.19`
- mAP: `53.78`
- Hard: `35.83`

和 `scenario SLR-C` 相比：

- Macro: `+0.45`
- Micro: `+0.09`
- Samples: `-0.28`
- mAP: `+0.12`
- Hard: `+1.86`

因此，当前最合理的总判断是：

1. 如果要一个**最强的整体方法**，选 v2
2. 如果特别强调 `samples F1` 且想保持方法最简单，选纯 `SLR-C`
3. 如果目标是继续推 `macro / hard`，就继续沿着 v2 走

---

## 2. 方法演化脉络

### 2.1 baseline：强视觉分类器

最早期的关键发现不是“复杂结构更强”，而是：

> **一个足够强的 frozen CLIP visual baseline 已经比复杂 decomposition / concept / reasoning 方案更有效。**

这个 baseline 的作用是：

- 提供一个强而稳定的候选意图打分器
- 把问题从“如何重新设计大模型”收缩成“如何修正 candidate-local 排序和最终决策”

### 2.2 SLR-C：把语言先验和决策规则接上去

接着发现：

- baseline 的候选召回已经很高
- 主要问题在近义 intent 之间的局部消歧
- 文本先验能帮忙修正候选内部排序
- class-wise threshold 对 reranked score space 非常关键

于是得到 `SLR-C`：

> **semantic local rerank + class-wise calibration**

这一步把系统从单纯视觉分类器升级成：

- 强视觉 backbone
- 候选级语义先验重排
- 类别级 calibrated decision rule

### 2.3 focused verifier：把 evidence verification 接进来

然后又发现：

- 只做文本 rerank 还不够“显式”
- top-k 候选里很多时候都能被相似的正证据支持
- 真正需要的是从图像中找显式 evidence，再用它区分近邻候选

于是又做了 data-driven evidence verification：

- benchmark evidence bank
- training-set relation learning
- sparse evidence profile
- residual rerank

第一轮 full-space 搜索说明：

- 如果把所有 relation family 混在一起按单个 validation macro 选模
- 容易选到 `positive_mean` 这种 commonness-heavy 配置
- 结果会被“选坏”

后来 focused 搜索只保留判别式 relation 后，增益才真正显现。

### 2.4 v2：从单类 residual 走向 comparative verification

再往前一步的关键判断是：

> 当前问题不是“给每个候选一个更好的绝对分数”，而是“在 top-k 候选里，A 为什么比 B 更合理”。  
> 这本质上是一个 **pairwise / comparative** 问题。

所以 v2 的升级点是：

1. 从单类 verification 改成 pairwise comparative verification
2. 加入 margin-aware gate，只在 verifier 真该出手时增强它
3. 尝试 confusion-neighborhood negatives，但这一轮没超过普通 hard negatives

---

## 3. 当前推荐方法的完整结构

当前推荐方法可以写成：

```text
Image
  -> Frozen CLIP ViT-L/14 visual classifier
  -> baseline logits
  -> scenario text prior rerank (SLR)
  -> class-wise thresholds (SLR-C)
  -> top-k candidate intents
  -> multi-expert evidence extraction
  -> data-driven relation learning
  -> pairwise comparative verification
  -> margin-aware gate
  -> residual fusion
  -> class-wise thresholds
  -> final multi-label prediction
```

下面按模块展开。

---

## 4. 模块一：Visual Baseline

### 4.1 Backbone

视觉主干使用：

- CLIP `ViT-L/14`
- frozen visual encoder
- 图像尺寸：`224`

具体实现配置见：

- `configs/model/intentonomy_clip_vit_layer_cls_patch_mean.yaml`

### 4.2 Feature 形式

当前 strongest baseline 不是只用 CLS token，而是：

- 取 CLIP 最后一层（`layer_idx = 24`）
- 抽取 `CLS token`
- 对 patch tokens 取 mean
- 把 `CLS + mean(patch)` 拼接
- 送入 MLP classifier 做多标签分类

这一步的关键意义是：

> backbone 已经足够强，后续方法不是为了替代它，而是为了修正它的 candidate-local mistakes。

### 4.3 Training

baseline 本身是训练得到的，训练配置核心为：

- frozen backbone
- MLP head
- `AsymmetricLossOptimized`
- optimizer: `AdamW`
- scheduler: cosine annealing

当前分析和后处理主要基于 checkpoint：

- `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

### 4.4 Baseline 输出

对于图像 `x`，baseline 输出：

[
z^{base}(x)\in\mathbb{R}^{C}
]

其中 `C = 28` 是 Intentonomy 的类别数。

对应 test baseline 指标：

- global threshold:
  - Macro `45.98`
  - Micro `56.54`
  - Samples `56.40`
  - mAP `50.29`
  - Hard `26.86`

---

## 5. 模块二：SLR-C

`SLR-C` 是当前整条 pipeline 的真正 base system。

### 5.1 SLR 的核心思想

`SLR` 全称可以理解为：

> **Semantic Local Rerank**

做法不是重新训练新分类器，而是：

1. 让视觉 baseline 先给出全类别 logits
2. 用文本先验给每个类别额外提供一个 semantic prior score
3. 只在 baseline 的 top-k 候选内进行局部重排

也就是说，SLR 做的是：

> candidate-local semantic correction  
> 而不是 full-space score replacement

### 5.2 文本先验的来源

当前已经试过多个 source：

- `lexical`
- `canonical`
- `scenario`
- `discriminative`
- `lexical + canonical`

其中当前 strongest single-source 是：

- `scenario`

### 5.3 `scenario` 具体是什么

`scenario` 来自：

- `intent_description_gemini.json`

其中每个 intent 类别由 Gemini 生成若干条 `Text Query`，表示该意图的典型视觉场景。

它不是短标签，也不是一句 canonical definition，而是：

> 更接近“这个 intent 在图像里通常会怎么被看见”

### 5.4 SLR 的具体计算

用 CLIP text encoder 编码 scenario text queries，得到 intent 的文本先验分数：

$s^{prior}(x)\in\mathbb{R}^{C}$

然后只对 baseline 的 top-k 候选做局部修正：

$$
\tilde z_c =
\begin{cases}
z_c^{base} + \alpha \cdot \tilde s^{prior}_c, & c \in \mathcal{T}_k(x) \\
z_c^{base}, & c \notin \mathcal{T}_k(x)
\end{cases}
$$

其中：

- `top-k = 10`
- `alpha = 0.3`
- prior fusion 使用 `add_norm`
  - 即先对 prior scores 做 per-sample z-score normalization
  - 再加回 baseline logits

### 5.5 为什么只在 top-k 内 rerank

因为 candidate recall 已经足够高：

- top-10 label recall: `85.36%`
- top-10 sample-any recall: `95.72%`
- top-10 sample-all recall: `77.14%`

所以问题不在全局召回，而在：

> top-k 内部谁更像正确类

### 5.6 C：class-wise threshold

`SLR-C` 的 `C` 表示：

> **class-wise calibration / class-wise threshold**

具体做法：

- 在 validation set 上为每个类别独立搜索阈值
- 搜索网格：`0.05, 0.06, ..., 0.95`
- 每个类都选择能让该类 F1 最优的阈值

实现见：

- `src/utils/decision_rule_calibration.py`
  - `search_classwise_thresholds`

最终预测规则：

[
\hat y_c = \mathbb{I}(\sigma(\tilde z_c) > t_c)
]

### 5.7 `SLR-C` 的地位

这一步非常关键。

`SLR-C` 不是一个小后处理技巧，而是：

> 整个系统中负责 proposal + calibrated decision 的核心基座

它的 test 指标是：

- Macro `51.28`
- Micro `59.13`
- Samples `58.47`
- mAP `53.66`
- Hard `33.98`

---

## 6. 模块三：Focused Verifier（v1 focused）

这一步的目标是把 `SLR-C` 再往上推。

### 6.1 输入

`SLR-C` 提供：

- top-k candidate intents
- candidate-local baseline geometry

Focused verifier 再引入：

- image evidence
- training-set learned intent-element relations

### 6.2 Evidence extraction

图像不会直接拿去和 intent 做第二次文本匹配，而是先被解析到四个 benchmark evidence spaces：

- `object`: COCO 80
- `scene`: Places365 365
- `style`: Flickr Style 20
- `activity`: Stanford40 40

对每张图像：

1. 用 frozen CLIP image feature 编码图像
2. 与各 expert bank 的 text embeddings 计算相似度
3. 得到四个 expert 的 evidence score matrix

这一步的意义是：

> 把“抽象 intent 判断”拆成“显式视觉元素响应”

### 6.3 Relation learning

Focused verifier 真正有效的部分不是 `positive_mean`，而是判别式 relation。

当前最有效的两个 relation family 是：

1. `hard_negative_diff`
2. `support_contradiction`

其中 focused v1 最常用、也最稳的一支是：

[
R(c,z)=\mu_c^+(z)-\mu_{hard(c)}^+(z)
]

也就是：

- 正样本下该元素对 intent `c` 的平均响应
- 减去 hardest negative intents 下该元素的平均响应

### 6.4 Sparse profile

不是保留所有元素，而是只保留最有信息量的 top-N evidence：

- 实验上 `top-10 / top-20` 都明显优于 `all`
- focused v1 的强配置通常落在 `top-20`

这一步非常重要，因为它把 verifier 从“噪声辅助项”变成了：

> candidate-local sparse discriminator

### 6.5 Focused verifier 的最终形式

在 focused v1 里，最强的一支可以概括成：

- expert subset：`all`
- relation：`hard_negative_diff`
- sparse profile：`top-20`
- activation top-m：`5`
- rerank beta：`0.08 ~ 0.2`

最终对 top-k 候选做 residual rerank：

[
z_c^{focus}= \tilde z_c + \beta \, V(c,x)
]

然后继续用 class-wise threshold 做输出。

### 6.6 Focused verifier 的代表结果

focused v1 validation-best 结果：

- Macro `51.35`
- Micro `59.06`
- Samples `58.12`
- mAP `53.58`
- Hard `35.24`

相比 `SLR-C`：

- Macro 稍涨
- Hard 明显涨
- Samples 略掉

这说明 verifier 确实有效，但还没有完全抓住问题本质。

---

## 7. 模块四：v2 Best Method

这一步是当前推荐方法的最终版本。

### 7.1 为什么 v1 还不够

v1 仍然是在做：

> 给每个候选一个绝对 verification residual

但当前真正要解决的是：

> 在 top-k 里，`c_i` 为什么比 `c_j` 更合理？

所以 v2 的核心变化是：

> 从单类 residual 变成 pairwise comparative verification

### 7.2 Pairwise comparative verification

对于 top-k 候选中的任意两个类别 `c_i, c_j`，构造 pairwise relation：

[
R_{pair}(c_i,c_j,z)=R(c_i,z)-R(c_j,z)
]

再只用图像中激活的 evidence 元素计算相对支持差异：

[
\Delta V(c_i,c_j,x)=\sum_{z\in A(x)} w_z(x)\,[R(c_i,z)-R(c_j,z)]
]

最后把候选在所有 pairwise comparison 中的 margin 做平均，得到 comparative verification score。

实现上，这一步对应：

- `build_pairwise_relation_profiles`
- `compute_pairwise_comparative_scores`

### 7.3 Margin-aware gate

v2 不让 verifier 永远同样强地介入，而是先看 `SLR-C` 自己有多确定。

先取 baseline top-1 / top-2 候选 margin：

[
m(x)=z_{top1}(x)-z_{top2}(x)
]

再构造 gate：

[
g(x)=\exp(-\gamma \, m(x))
]

含义是：

- margin 小：候选接近，verifier 更该出手
- margin 大：baseline 已经很自信，verifier 少动

### 7.4 为什么 v2 用 `add`，不用 `add_norm`

这个点非常关键。

一开始也试了：

- pairwise compare
- margin gate
- `add_norm`

但发现所有 `gate_gamma` 几乎等价。

原因是：

1. gate 先做样本级缩放
2. `add_norm` 再做 per-sample z-score
3. gate 的缩放被 normalization 抵消

所以当前 v2 的正确形式是：

[
z_c^{v2}= \tilde z_c + \beta \, g(x)\, V^{pair}(c,x)
]

并且 fusion 必须用：

- `add`

而不是 `add_norm`

### 7.5 当前最优的 v2 配置

当前 validation-best v2 complete method：

- Stage 0: `scenario SLR-C`
- evidence experts: `all`
- relation: `hard_negative_diff`
- sparse profile: `top-20`
- pairwise profile: `top-5`
- activation top-m: `5`
- gate gamma: `2.0`
- fusion: `add`
- beta: `0.01`

### 7.6 confusion-neighborhood negatives 为什么没进入最终版本

v2 还实现了：

- `confusion_hard_negative_diff`

它用 top-k confusion 统计构造 class-specific neighborhood negatives。

但当前最小稳定实验里，它没有超过普通 `hard_negative_diff`。

所以当前正式推荐版本里：

- **保留 pairwise compare**
- **保留 margin-aware gate**
- **不把 confusion-neighborhood negatives 作为默认模块**

---

## 8. 当前最优完整 pipeline：逐步推理过程

这里给出完整 inference pipeline。

### Step 1. 图像进入视觉 baseline

输入图像 `x`：

- resize 到 `224`
- 用 frozen CLIP `ViT-L/14` 编码
- 取最后一层 `CLS + mean(patch)`
- 用 MLP 得到 baseline logits

输出：

[
z^{base}(x)
]

### Step 2. 计算 scenario semantic prior

对每个 intent：

- 从 `intent_description_gemini.json` 取 `Text Query`
- 用 CLIP text encoder 编码
- 与图像特征计算语义相似度

输出：

[
s^{scenario}(x)
]

### Step 3. 做 SLR top-k rerank

只在 baseline top-10 候选上做：

[
\tilde z = \text{SLR}(z^{base}, s^{scenario})
]

当前实现：

- `top-k = 10`
- `alpha = 0.3`
- `mode = add_norm`

### Step 4. 用 validation 学 class-wise thresholds

在 validation set 上搜索：

- 每一类各自的最佳阈值 `t_c`

得到 `SLR-C`。

### Step 5. 抽取图像 evidence

对同一张图像，再计算：

- object bank scores
- scene bank scores
- style bank scores
- activity bank scores

### Step 6. 用训练集学 data-driven relations

从训练集统计：

- 哪些 element 相对 hardest negatives 更支持某个 intent

保留 sparse top-20 profile。

### Step 7. 在 top-k 候选里做 pairwise comparative verification

对 top-k 中的每一对候选 `c_i, c_j`：

- 只比较两者之间的相对证据差异
- 聚合得到每个候选的 comparative verification score

### Step 8. 用 margin-aware gate 控制 verifier 强度

根据 `SLR-C` 自己的 top-1 / top-2 margin 算 gate：

- margin 小，gate 大
- margin 大，gate 小

### Step 9. 把 verifier 残差加回候选 logits

只在 top-k 内做：

[
z^{final}= \tilde z + \beta \, g(x)\, V^{pair}(x)
]

其中当前默认：

- `beta = 0.01`
- `gamma = 2`

### Step 10. 再次用 class-wise thresholds 输出标签

最终做：

[
\hat y_c = \mathbb{I}(\sigma(z_c^{final}) > t_c)
]

输出多标签预测。

---

## 9. 当前最优方法与各阶段结果对比

| Stage | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 45.98 | 56.54 | 56.40 | 50.29 | 26.86 |
| Baseline + class-wise threshold | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR` | 47.14 | 56.52 | 55.11 | 53.66 | 28.32 |
| `scenario SLR-C` | 51.28 | 59.13 | **58.47** | 53.66 | 33.98 |
| focused verifier (v1) | 51.35 | 59.06 | 58.12 | 53.58 | 35.24 |
| v2 best method | **51.73** | **59.22** | 58.19 | **53.78** | **35.83** |

这个表说明三件事：

1. `SLR-C` 是真正的性能拐点  
2. v1 verifier 证明了 evidence verification 有价值  
3. v2 把 verifier 从“有价值”推进到了“当前推荐版本”  

---

## 10. 为什么当前方法有效

### 10.1 baseline 解决了“强视觉表示”

当前问题已经不是缺 backbone capacity。

### 10.2 SLR-C 解决了“语义先验 + 最终决策”

它把 abstract intent label space 的语言先验和 calibrated threshold 接到了视觉分类器上。

### 10.3 verifier 解决了“candidate-local ambiguity”

这是目前最关键的剩余误差来源。

### 10.4 v2 把 verifier 进一步变成“候选对比器”

这正好贴合当前任务的真实瓶颈：

- `Happy vs Playful`
- `EnjoyLife vs Happy`
- `FineDesign vs FineDesign-Art`

不是绝对分类问题，而是近邻候选比较问题。

### 10.5 gate 让 verifier 不会过度干扰 easy cases

虽然目前 samples 还略低于纯 `SLR-C`，但 gate 至少已经把 verifier 的使用方式从“总是介入”改成了“有条件介入”。

---

## 11. 当前正式推荐的使用方式

### 11.1 如果要默认方法

推荐：

> `scenario SLR-C + v2 comparative verifier (add-gated)`

因为这是当前最强的整体方法。

### 11.2 如果要最稳妥、最简单的方法

推荐：

> 纯 `scenario SLR-C`

因为：

- 结构最简单
- 已经很强
- samples 最稳

### 11.3 如果要继续做研究推进

最值得继续的不是重回全空间大搜索，而是：

1. 把 `hard_negative_diff + pairwise compare + add-gate` 固化成正式主方法
2. 单独重做 `support_contradiction` 的稳定实验
3. 继续研究更好的 gate，而不是继续扩 evidence bank

---

## 12. 代码与产物映射

### 12.1 主要代码

- baseline model:
  - `configs/model/intentonomy_clip_vit_layer_cls_patch_mean.yaml`
  - `src/models/intentonomy_clip_vit_layer_cls_patch_mean_module.py`
- SLR / prior analysis:
  - `scripts/analyze_text_prior_boundary.py`
  - `scripts/analyze_calibrated_decision_rule.py`
- v1 data-driven verifier:
  - `scripts/analyze_data_driven_agent_evidence_verification.py`
- v2 verifier:
  - `scripts/analyze_agent_evidence_verification_v2.py`
- evidence utilities:
  - `src/utils/evidence_verification.py`
- threshold calibration:
  - `src/utils/decision_rule_calibration.py`

### 12.2 关键实验产物

- calibration / SLR:
  - `docs/record/record_0310_CalibrationReranking.md`
- v1 verifier:
  - `docs/record/record_0311_02_AgentEvidenceVerification_v1.md`
- v2 verifier:
  - `docs/record/record_0311_03_AgentEvidenceVerification_v2.md`

---

## 13. 最终总结

当前最优完整方法不是单个“模型结构创新”，而是一条逐层收敛出来的 pipeline：

1. 用 frozen CLIP `ViT-L/14` 建立强视觉 baseline
2. 用 `scenario` semantic prior 做 local rerank
3. 用 class-wise threshold 做 calibrated decision，得到 `SLR-C`
4. 再用 data-driven evidence verification 专门处理 top-k 候选内部的细粒度消歧
5. 最后用 pairwise compare 和 margin-aware gate，把 verifier 变成一个更有针对性的局部比较模块

所以，当前最准确的命名不是“某个单点 trick”，而是：

> **A strong calibrated candidate-proposal system (`SLR-C`) augmented by a pairwise, data-driven, gated evidence verifier.**
