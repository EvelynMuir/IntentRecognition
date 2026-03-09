# 2026-03-09 Text Prior / Retrieval Boundary Analysis

## 1. 目标

这份文档将原先的 4 个 TODO 收敛为一套统一实验协议，用来回答一个核心问题：

> 当前 strongest baseline 的 hard gap，究竟更像是 `semantic prior / neighbor reference` 不足，还是纯 `decision boundary` 问题？

这里的分析不改训练，只围绕固定 baseline checkpoint 做后验诊断与轻量融合。

---

## 2. 固定口径

### 2.1 Baseline 定义

凡是本文里的 `baseline`，都固定指向：

`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

对应 run:

- `logs/train/runs/2026-03-03_16-34-30`
- 配置：`experiment=intentonomy_clip_vit_layer_cls_patch_mean`
- Backbone：`frozen CLIP ViT-L/14`
- Feature：第 `24` 层 `CLS + mean patch`
- Head：`2-layer MLP`

当前已知基线结果：

| Split | macro F1 | micro F1 | samples F1 | easy | medium | hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.4785 | 0.5682 | 0.5660 | 0.7549 | 0.5207 | 0.3322 |
| test | 0.4638 | 0.5676 | 0.5670 | 0.7576 | 0.5263 | 0.2819 |

补充说明：

- 上表是历史 standalone eval / record 中的 baseline 结果。
- 本文后续所有 `text prior / retrieval` 对比，统一以同一脚本协议下的 baseline 为准：
  - `val` 选全局阈值
  - `test` 直接复用该 `val` 阈值
- 在这个统一协议下，本次全量分析里的 baseline 是：
  - `test macro = 0.4598`
  - `test hard = 0.2686`
- 因此后文所有 gain 都相对于这组“同协议 baseline”计算，而不是相对于上表 standalone eval 数字。

### 2.2 评估协议

- 所有新分析都以 `val` 选阈值。
- `test` 统一复用 `val` 上选出的阈值，不在 `test` 上重新 search。
- 统一报告：
  - `macro F1`
  - `micro F1`
  - `samples F1`
  - `mAP`
  - `easy / medium / hard`
- `hard` 划分沿用 `src/utils/metrics.py` 里的 `SUBSET2IDS["hard"]`。

### 2.3 统一脚本入口

已新增：

- `scripts/analyze_text_prior_boundary.py`

该脚本一次性覆盖：

1. 纯文本零样本 / 相似度诊断
2. baseline top-k rerank
3. retrieval prior upper bound
4. hard-case 局部导出

默认输出到：

- `logs/analysis/<timestamp>_text_prior_boundary/`

主要产物：

- `summary.json`
- `rerank_leaderboard.csv`
- `retrieval_leaderboard.csv`
- `hard_cases_baseline_wrong_llm_right.json`
- `hard_cases_baseline_right_llm_wrong.json`

---

## 3. TODO 1：纯文本零样本 / 相似度诊断

### 3.1 要回答的问题

LLM descriptions 是否真的比 label name 更能区分 hard classes？

### 3.2 文本源定义

脚本固定对比 3 种 text source：

1. `short`
   - annotation 里的短 label name
2. `detailed`
   - `INTENTONOMY_DESCRIPTIONS`
3. `llm`
   - `intent_description_gemini.json`
   - 每个 intent 会聚合：
     - `intent`
     - `Core Difference`
     - `Visual Elements`
     - `Text Query`

其中：

- `short / detailed` 用 prompt:
  - `A photo that expresses the intent of {}.`
- `llm` 直接用原始描述文本，不再额外套 prompt

### 3.3 打分方式

- 图像侧：标准 CLIP image embedding
- 文本侧：标准 CLIP text embedding
- 核心分数就是 image-text `cosine similarity`
- 实现上会再乘以 CLIP 自带 `logit_scale`，得到 `text_logit`

这里要特别说明：

- TODO 1 的本意是“直接用 similarity 做分类或 rerank”
- 也就是说，`similarity / text_logit` 本身就是 class score
- 不应该把它理解成又训练了一个新的分类器

因此，TODO 1 里有两种合法用法：

1. `text-only classification`
   - 直接把 similarity 当作每个类的分数
   - 再做 top-k、threshold 或 ranking evaluation
2. `baseline rerank`
   - 在 baseline 给出的候选类内部，用 similarity 重新排序

本文脚本里之所以还会计算 `sigmoid(text_logit)`，只是为了和仓库现有 multi-label F1 评估口径对齐。
但从诊断角度看，`similarity ranking` 才是主信号，`thresholded F1` 只是辅助视角。

### 3.4 重点看什么

- `similarity ranking` 本身是否变好
- `mAP / per-class AP`
- `overall`
- `hard subset`
- `test_top_confusions`
- `test_top_hard_confusions`
- `llm - short` / `detailed - short` 的 per-class gain

### 3.5 判据

只要满足以下任一条件，就说明 text prior 路线值得继续：

- `llm` 在 `hard` 上显著优于 `short`
- `llm` 在若干 hard classes 上 per-class F1 提升明显
- `llm` 能缓解若干典型 confusion pairs

不要求 zero-shot overall 直接超过 baseline。

更准确地说：

- 如果 `mAP / ranking` 改善，但 thresholded F1 没改善，不能直接判定这条路无效
- 这更可能说明 text prior 有排序价值，但没有被当前全局阈值充分利用

---

## 4. TODO 2：baseline (cls + mean patch) top-k rerank

### 4.1 要回答的问题

如果 baseline 本身已经把候选空间压到 `top-5 / top-10`，text prior 能不能只在局部排序上帮到 hard classes？

### 4.2 做法

对 baseline logits `z`，先取 top-k 候选类，再只对候选类做 logit-space 融合：

1. `add`

```text
z' = z + alpha * s_text
```

2. `mix`

```text
z' = (1 - alpha) * z + alpha * s_text
```

3. `add_norm`

```text
z' = z + alpha * norm(s_text)
```

其中：

- `s_text` 使用上一步得到的 `text_logit`
- `norm(s_text)` 是按样本对类别维做 z-score
- 非 top-k 类别保持原 baseline logits 不变

### 4.3 当前搜索空间

- `topk in {5, 10}`
- `mode in {add, mix, add_norm}`
- `alpha in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}`
- 默认 rerank source：`llm`

### 4.4 重点看什么

- `val_macro` 最优配置
- `val_hard` 最优配置
- `test_macro / hard / easy / medium`
- 相对 baseline 的增益
- 哪些类收益最大

### 4.5 判据

如果出现下面任一现象，就说明问题更像是“候选已对，但局部排序/语义校正不足”：

- top-k rerank 能涨 `hard`
- `hard` 提升同时 overall 不明显掉
- gain 主要集中在抽象语义类，而不是视觉显著类

---

## 5. TODO 3：retrieval prior upper bound

### 5.1 要回答的问题

如果 hard gap 主要来自“缺少语义邻居 / prototype reference”，那么一个非常简单的 retrieval prior 应该已经能救一部分 hard。

### 5.2 做法

使用 train split 的 CLIP image features 做 kNN 检索，构造两种 prior：

1. `binary_vote`
   - 邻居标签二值投票均值
2. `soft_distribution`
   - 邻居 soft label 分布均值

先评估 prior-only，再与 baseline logits 做小权重融合：

```text
z' = z + beta * logit(prior)
```

其中 `prior` 会先做 safe-logit clipping。

### 5.3 当前搜索空间

- `k in {5, 10, 20}`
- `beta in {0.05, 0.1, 0.2, 0.3}`
- `prior in {binary_vote, soft_distribution}`

### 5.4 重点看什么

- retrieval prior 单独能否在 `hard` 上优于 text-only
- retrieval + baseline 是否优于 baseline
- `soft_distribution` 是否明显强于 `binary_vote`

### 5.5 判据

如果 retrieval prior 单独或融合后就能明显救 `hard`，说明：

- hard gap 确实和“邻居语义参照不足”有关
- 问题不只是 boundary calibration
- 后续可以考虑把 retrieval 做成更正式的 memory / prototype / neighbor branch

如果 retrieval 几乎不救 hard，而 rerank 能救，则更像是 boundary / semantic reweighting 问题。

---

## 6. TODO 4：hard-case 局部分析

### 6.1 要回答的问题

LLM prior 到底在补什么？

- 抽象语义映射
- 类边界解释
- 场景原型
- 还是单纯文本偏见

### 6.2 导出规则

脚本会导出两组 test hard cases：

1. `baseline_wrong_llm_right`
   - LLM rerank 的 sample-F1 高于 baseline
   - 且恢复了 baseline 漏掉的真值标签
2. `baseline_right_llm_wrong`
   - LLM rerank 的 sample-F1 低于 baseline
   - 且丢掉了 baseline 原本命中的真值，或引入了新的 false positive

每条记录包含：

- `image_id`
- `image_path`
- `ground_truth_labels`
- `baseline_pred_labels`
- `llm_pred_labels`
- `recovered_labels`
- `dropped_true_labels`
- `new_false_positive_labels`
- `baseline_top5`
- `llm_rerank_top5`
- `llm_text_only_top5`

### 6.3 重点看什么

人工看样本时，优先回答这几个问题：

1. LLM 是否恢复了 baseline 原本排在边界附近的抽象类？
2. LLM 是否依赖了某种高层语义原型，而不是局部视觉细节？
3. LLM 出错时，是否把“语义相近但标注更窄/更宽”的类推高？
4. 错误是否集中在描述文本带有强偏置词的类别？

---

## 7. 推荐运行命令

全量运行：

```bash
python scripts/analyze_text_prior_boundary.py \
  --run-dir logs/train/runs/2026-03-03_16-34-30 \
  --ckpt-path logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt \
  --rerank-source llm \
  --num-workers 4
```

快速 smoke test：

```bash
python scripts/analyze_text_prior_boundary.py \
  --run-dir logs/train/runs/2026-03-03_16-34-30 \
  --ckpt-path logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt \
  --num-workers 0 \
  --max-samples 64
```

---

## 8. 事前判据回顾

下面这些是跑实验前预先定义的解释框架；实际结果与结论见第 9-11 节。

### 8.1 如果 `llm zero-shot > short zero-shot`，且 rerank 能救 hard

说明：

- 文本先验确实包含 baseline 没显式利用的抽象语义
- 下一步可以往 `text prior calibration / lightweight semantic adapter` 走

### 8.2 如果 retrieval prior 明显比 text prior 更有效

说明：

- hard gap 更像是缺少近邻参照
- 后续更值得投入 retrieval / memory / prototype 路线

### 8.3 如果两者都不明显有效

说明：

- strongest baseline 的 hard gap 可能更像标注噪声、label ambiguity 或阈值/校准尾部问题
- 后续更适合继续做 `boundary-aware` 或 `uncertainty-aware` 路线

---

## 9. 全量结果（2026-03-09）

全量实验已完成，输出目录固定为：

- `logs/analysis/full_text_prior_boundary_20260309`

核心结果文件：

- `summary.json`
- `rerank_leaderboard.csv`
- `retrieval_leaderboard.csv`
- `hard_cases_baseline_wrong_llm_right.json`
- `hard_cases_baseline_right_llm_wrong.json`

### 9.1 TODO 1 结果：text-only zero-shot 基本无效

如果只看当前全局阈值下的 thresholded F1，3 种文本源几乎完全一样：

| Source | val macro | val hard | test macro | test hard | test mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 0.1186 | 0.0708 | 0.1218 | 0.0696 | 18.48 |
| detailed | 0.1186 | 0.0708 | 0.1218 | 0.0696 | 24.78 |
| llm | 0.1186 | 0.0708 | 0.1218 | 0.0696 | 13.45 |

直接结论：

- `llm zero-shot` 并没有优于 `short`
- `detailed` 的 ranking 质量略高于 `short`（`mAP` 更高），但仍然无法转化为 thresholded F1 提升
- `llm` 反而在 `mAP` 上更差，说明它作为“全局 CLIP classifier”并不可靠

因此，TODO 1 的答案是：

> LLM descriptions 不适合作为纯 text-only 分类器，但这并不排除它在局部 rerank 场景下有价值。

换句话说，TODO 1 的设计思路本身是对的：

- 先算 similarity
- 再用 similarity 做分类或 rerank

真正的实验结论不是“similarity 设计错了”，而是：

- `text-only similarity classification` 不够强
- `similarity-based local rerank` 反而很有效

### 9.2 TODO 2 结果：top-k rerank 明显有效

按 `val_macro` 选出的最佳 rerank 配置为：

- `topk = 10`
- `mode = add_norm`
- `alpha = 0.3`

对应 test 结果：

| Method | macro | micro | samples | mAP | easy | medium | hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.4598 | 0.5654 | 0.5640 | 50.29 | 0.7640 | 0.5265 | 0.2686 |
| baseline + llm rerank | 0.4843 | 0.5773 | 0.5715 | 53.78 | 0.7591 | 0.5602 | 0.2879 |

相对 baseline 的增益：

- `macro +0.0245`
- `micro +0.0120`
- `samples +0.0076`
- `mAP +3.49`
- `hard +0.0193`

最强信号是：

- 最优配置稳定落在 `add_norm`
- `topk = 5` 和 `topk = 10` 都有效
- 增益不仅体现在 `macro`，`hard` 和 `medium` 也一起上涨

收益最大的类：

- `WorkILike +0.182`
- `BeatCompete +0.157`
- `FineDesignLearnArt-Culture +0.094`
- `EasyLife +0.093`
- `Playful +0.080`
- `Harmony +0.077`
- `CuriousAdventurousExcitingLife +0.064`
- `SuccInOccupHavGdJob +0.054`
- `InspirOthrs +0.049`

受损较明显的类：

- `FineDesignLearnArt-Art -0.132`
- `Communicate -0.090`
- `PassionAbSmthing -0.054`
- `Attractive -0.050`
- `InLoveAnimal -0.033`
- `EnjoyLife -0.020`

这说明：

- LLM prior 的正向作用主要发生在抽象类、语义边界模糊类上
- 它对原型更明确、视觉风格更稳定的类别反而可能产生语义过泛化

### 9.3 TODO 3 结果：retrieval prior 只有弱增益

按 `val_macro` 选最优配置：

- `binary_vote`
- `k = 20`
- `beta = 0.05`

对应 test：

- `macro = 0.4609`
- `hard = 0.2652`

相对 baseline：

- `macro +0.0011`
- `hard -0.0033`

如果改成按 `val_hard` 选配置，最优是：

- `binary_vote`
- `k = 20`
- `beta = 0.2`

对应 test：

- `macro = 0.4672`
- `micro = 0.5747`
- `samples = 0.5820`
- `hard = 0.2815`

相对 baseline：

- `macro +0.0074`
- `hard +0.0129`

这个结果说明：

- retrieval prior 不是完全没用
- 但它的增益明显弱于 top-k text rerank
- 它更像次要补充，而不是当前 hard gap 的主解释

### 9.4 TODO 4 结果：LLM prior 在补“高层语义氛围”，也会带来正向文本偏置

注意：下面统计基于导出的 `top-40 improved` 和 `top-40 degraded` hard cases，而不是全量全集。

在导出的 improved cases 中：

- `37 / 40` 是“恢复真值且没有引入新 false positive”
- 被救回最多的标签是：
  - `EasyLife (8)`
  - `Playful (8)`
  - `WorkILike (6)`
  - `Happy (4)`
  - `SocialLifeFriendship (3)`
  - `GoodParentEmoCloseChild (3)`

这说明 LLM prior 确实能补：

- 生活状态类
- 情绪氛围类
- 职业满足感 / 社会关系这类高层语义

在导出的 degraded cases 中：

- `11 / 40` 会直接把预测压成空
- `10 / 40` 主要表现为新增 false positive
- 最常被误伤的标签是：
  - `Attractive (10)`
  - `FineDesignLearnArt-Art (7)`
  - `EnjoyLife (7)`
- 最常见的新 false positive 是：
  - `Playful (13)`

这说明它的典型风险是：

- 把图像往更“轻松 / 快乐 / 活跃”的通用语义上拉
- 对 `Attractive / FineDesignLearnArt-Art / EnjoyLife` 这类已有较强视觉原型或更细粒度定义的类造成挤压

---

## 10. 最终判断

这次实验最明确的结论是：

> 当前 strongest baseline 的 hard gap，更像是“局部边界与语义消歧问题”，而不是“缺少邻居 reference”的主问题。

原因有三点：

1. `text-only zero-shot` 不行，说明 LLM descriptions 不能直接替代分类器。
2. `top-k rerank` 明显有效，说明 baseline 往往已经把正确类放进候选集，问题出在局部排序与边界校正。
3. `retrieval prior` 只有弱增益，说明“缺邻居”不是当前 hard gap 的主导瓶颈。

更精确地说，这批结果支持以下表述：

- `LLM text prior` 更像一个 `local semantic calibrator`
- 它适合在 baseline 候选集内部做重排序
- 它不适合直接充当全局 zero-shot classifier

---

## 11. 下一步建议

基于本次结果，后续优先级建议如下：

1. 优先继续做 `lightweight semantic rerank / boundary calibration`
   - 方向是对 baseline logits 做局部语义校正，而不是重做大模型分类器
2. 如果要继续用文本先验，优先限制在 `top-k candidate refinement`
   - 不建议直接做纯 text-only 分类
3. retrieval 可以保留，但不应作为当前主线
   - 更适合后续作为辅助 prior，而不是主要 novelty
4. 新方法需要显式抑制 `Playful / EasyLife / Happy` 这类正向泛化偏置
   - 否则很容易通过“语义变宽”伤害 `Attractive / FineDesignLearnArt-Art / EnjoyLife`

---

## 12. 当前状态

原始 4 个 TODO 已全部完成，包括：

- 统一设计协议
- 统一分析脚本
- 全量实验运行
- 结果解读与方向判断

因此，这个问题当前不再是“要不要做”，而是“下一步把 rerank / calibration 做成什么训练式方法”。
