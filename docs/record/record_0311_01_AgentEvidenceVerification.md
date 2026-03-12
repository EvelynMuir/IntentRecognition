# 0311 答辩版大纲：Agent Evidence Verification

## 第 1 页：问题不是特征不够强，而是候选 intent 缺少显式验证

- Intentonomy 是高层、抽象、主观语义任务，单一 object / scene 特征不足以直接决定 intent。
- 当前 strongest `SLR-C` 已经说明 frozen CLIP baseline 很强，剩余瓶颈主要在候选内部的语义消歧与最终决策。
- 因此新的问题不是“再做更强分类器”，而是“能否对 top-k intent hypotheses 做显式证据核验”。

## 第 2 页：当前 strongest pipeline 已经把正确类召回进候选集

- `scenario SLR + class-wise threshold` 当前 overall 最强：`macro 51.28 / micro 59.13 / samples 58.47 / hard 33.98`。
- baseline top-10 candidate recall 已经很高：`label recall 83.91% / sample-any 95.07% / sample-all 75.33%`。
- 这说明真正的问题是 candidate set 内部“谁更对”，而不是 candidate proposal 本身太弱。

## 第 3 页：核心想法是把 intent 识别改写成 hypothesis verification

- 先用 strongest `SLR-C` 生成 top-k intent hypotheses，而不是重新设计全局分类器。
- 再为每个 hypothesis 收集 `object / scene / style / activity` 四类证据，并检查这些证据是否支持该 hypothesis。
- 最后把 verification score 作为 residual signal 回注到候选排序，再交给 calibrated decision rule 输出最终多标签结果。

## 第 4 页：方法流程是 propose, collect, verify, decide

- `Image -> scenario SLR -> top-k hypotheses -> evidence extraction -> evidence matching -> verification rerank -> calibrated decision`。
- 这个流程对应自然的 agent 叙事：`hypothesis generation -> evidence collection -> evidence verification -> calibrated decision`。
- 方法定位是对当前 `SLR-C` 的增强，而不是替换 strongest baseline。

## 第 5 页：候选提出阶段完全沿用 strongest SLR-C

- 输入是 baseline logits 与 scenario prior，输出是 top-k candidate intents。
- `top-k` 只决定“哪些类别允许被 verification 修正”，不是 matching 本身的输入。
- 这样可以保留 strongest candidate proposal，同时把新方法的增益集中在 candidate-local correction 上。

## 第 6 页：每个 intent 先被拆成一个 evidence template

- 对每个 intent 构建 `object / scene / style / activity` 四类 expected evidence 短语。
- template phrase 是短语级证据，例如 `museum / historic site / aesthetic / looking`，不是完整句子，也不是 bank label。
- 模板来源是 `intent_description_gemini.json` 的 `Visual Elements` 与 `intent2concepts.json`，作用是定义“什么证据支持该 intent”。

## 第 7 页：证据抽取使用标准 benchmark label sets

- `object`: COCO 80，`scene`: Places365 365，`style`: Flickr Style 20，`activity`: Stanford40 40。
- 图像先被编码成 CLIP image feature，再与每个 expert bank 的 text embeddings 计算相似度，得到 evidence score matrix。
- 这样做的好处是 label space 公开、稳定、可解释，并且不依赖额外重型 detector pipeline。

## 第 8 页：Hypothesis-Conditioned Matching 显式使用图像特征

- matching 不是“只看 top-k candidates”，而是先用图像特征对 benchmark bank 产生 evidence scores。
- 再把 template phrases 和 bank labels 编码到 CLIP text space，计算 `template-bank similarity`。
- 最终用“图像对 bank 的响应”结合“template 到 bank 的相似度”得到 class-specific support。

## 第 9 页：v1 与 v2 的区别在 aggregation，而不在 extraction

- `v1` 使用 template-aware fixed aggregation，问题是所有类别共享同一 expert policy，容易稀释强单 expert 信号。
- `v2` 改成 class-wise expert routing，用 validation per-class gain 决定每个类别依赖哪个 expert。
- 因此 v2 的核心不是“更多 expert”，而是“不同 intent 依赖不同证据结构”。

## 第 10 页：最终输出仍然依赖 calibration，而不是只靠 rerank

- verification score 只作为 residual signal 加回 `SLR` top-k logits，不单独取代 baseline 分类器。
- 当前结果表明 verification 之后仍然明显依赖 class-wise threshold，才能把排序收益转成真正的 multi-label outputs。
- 所以方法分工很清楚：`SLR` 负责 proposal，`MEV` 负责 verification，`calibration` 负责 final decision。

## 第 11 页：主结果说明 v1 跑通了，v2 更进一步

- `prior SOTA`: `43.05 / 54.77 / 56.75 / 27.39`，`baseline`: `45.98 / 56.54 / 56.40 / 26.86`，`SLR-C`: `51.28 / 59.13 / 58.47 / 33.98`。
- `MEV v1`: `50.84 / 58.17 / 57.21 / 33.53`，说明 benchmark-bank evidence verification 已完整跑通，但尚未正式超过 strongest `SLR-C`。
- `MEV v2`: `51.10 / 59.00 / 58.24 / 33.99`，明显优于 v1，并在 hard 上基本追平甚至略高于 `SLR-C`。

## 第 12 页：ablation 表明问题不在 expert 数量，而在 expert 选择

- 单 expert 诊断结果里，`style` 最强：`macro 51.86 / hard 35.21`，`activity` 次强：`macro 51.58 / hard 34.79`。
- 相比之下，`all experts` 的 `v1` 正式最好结果只有 `macro 50.84 / hard 33.53`，说明固定 all-expert aggregation 会稀释强信号。
- 因此当前最关键的 ablation 结论是：gain 的核心来自正确的 expert selection / routing，而不是无条件多专家堆叠。

## 第 13 页：analysis 表明 v2 的 route 方式是合理的

- `v2` 最优配置为 `top1_positive + beta 0.3 + add_norm`，共有 `20` 个类别被 route，`8` 个类别直接退回 `SLR-C`。
- 最佳 routed 分布为：`object 2 / scene 7 / style 1 / activity 10`，说明并不是所有类别都需要 verification。
- 这说明当前最合理的解释是：verification 更像 selective local disambiguation，而不是全局统一修正。

## 第 14 页：最终结论是方法线成立，但下一步要继续打磨 routing

- strongest overall 仍然是 `SLR-C`，但 `MEV v2` 已经把差距缩到很小，并在 hard 上基本持平。
- 因此这条线已经从“设计假设”变成“可持续优化的方法线”，主方向是正确的。
- 下一步最值得继续做的是 `class-wise routing`、`uncertainty-aware selective verification` 和 `style/activity-dominant routing`，而不是继续无条件扩展 expert bank。

## 汇报时要特别避免的误解

- 不要把方法讲成“用了更多专家模型所以更强”，当前实现的重点其实是 benchmark bank + routing。
- 不要把 matching 讲成“只在 top-k 上做文本对齐”，它明确使用了图像特征与 bank evidence scores。
- 不要把方法讲成严格意义上的 `training-free`，更准确的说法是：不额外训练新 verification 模块，但依赖已有 checkpoint 和 validation-time calibration / search。
