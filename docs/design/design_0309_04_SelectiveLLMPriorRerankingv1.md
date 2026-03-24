# 2026-03-09 Semantic Local Reranking v1

## 1. 目标

这份文档记录在 `Selective LLM-Prior Reranking` 之外做的第二轮推理期尝试，核心问题是：

> 不去先学 gate，而是先把 `text prior` 和 `local rerank` 本身做得更强，是否能进一步超过当前 plain rerank？

这里的尝试全部不改训练，只在固定 baseline 上做推理期实验。

---

## 2. 固定口径

### 2.1 Baseline

固定 baseline checkpoint：

- `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

统一协议下 baseline test：

- `macro = 0.4598`
- `hard = 0.2686`

### 2.2 Plain Rerank 基准

固定 `SLR-v0` 作为比较基准：

- `source = llm`
- `topk = 10`
- `mode = add_norm`
- `alpha = 0.3`

对应 test：

- `macro = 0.4843`
- `micro = 0.5773`
- `samples = 0.5715`
- `mAP = 53.78`
- `hard = 0.2879`

后文所有 gain，默认都相对这组 plain rerank 计算。

### 2.3 输出目录

本轮实验结果目录：

- `logs/analysis/full_semantic_local_rerank_20260309`

核心文件：

- `summary.json`
- `semantic_local_rerank_leaderboard.csv`

---

## 3. 实现内容

这轮实现没有新开训练分支，而是在现有分析脚本里新增了一个 `semantic_local_rerank` 结果块。

实现位置：

- `scripts/analyze_text_prior_boundary.py`
- `src/utils/text_prior_analysis.py`

具体新增三类变体：

1. `prompt source` 消融
   - `short`
   - `detailed`
   - `llm`
   - `mixed`
2. `multi-prompt aggregation`
   - `average`
   - `max`
   - `top2_avg`
   - `logsumexp`
3. `comparative local rerank`
   - `none`
   - `topk_center`
   - `topk_margin`

其中：

- `mixed` 表示将 prompt-wrapped 的 `short + detailed` 和原始 `llm` 文本混合
- comparative rerank 只在 `top-k` 内改变 prior 的相对值，不改候选外类别

---

## 4. 方向 1：把 text prior 本身做得更强

### 4.1 Prompt 类型消融

这一步对应的问题是：

> 在相同 local rerank 框架下，哪种 text source 最有价值？

按各 source 在 validation 上的最佳配置选出结果后，test 表现如下：

| Source | Best config | test macro | test hard | 相对 SLR-v0 |
| --- | --- | ---: | ---: | --- |
| `short` | `average + none` | 0.4947 | 0.3276 | `macro +0.0104`, `hard +0.0397` |
| `detailed` | `average + none` | 0.4891 | 0.3198 | `macro +0.0048`, `hard +0.0319` |
| `llm` | `average + topk_center` | 0.4837 | 0.2956 | `macro -0.0006`, `hard +0.0077` |
| `mixed` | `average + topk_center` | 0.4928 | 0.3124 | `macro +0.0085`, `hard +0.0245` |

### 4.2 结果解读

这一步最重要的发现不是 `llm` 更强，而是：

- `short` 和 `detailed` 明显强于当前 `llm`
- `mixed` 也有效，但在按 val 选配置时没有压过 `short`

这和第一轮 plain rerank 的结论不完全一样。之前的 plain rerank 是：

- 固定用 `llm`
- 结果已经能明显涨点

而现在加入 prompt source 消融后，结果反而说明：

> 当前最强的 local rerank 文本源，不一定是 LLM 描述本身。

更具体地说：

- `short` 在 test 上给出当前最强的 hard 提升之一
- `detailed` 在 validation 上最稳，按 `val_macro` 选出来的 overall 最优配置就是它
- `llm` 的优势没有在这轮 source 消融里体现出来，说明 raw LLM descriptions 仍然存在过泛化或噪声问题

### 4.3 一个非常关键的补充观察

虽然按 `val_macro` 规则选出的全局最优配置是：

- `detailed + average + none`

其 test 为：

- `macro = 0.4891`
- `hard = 0.3198`

但在全部实验组合里，test 上实际最强的一个配置是：

- `mixed + average + none`

它的 test 为：

- `macro = 0.5061`
- `hard = 0.3529`

这个结果不能直接当正式最佳配置写进结论，因为它不是按 validation 规则选出来的；但它非常值得重视，因为它说明：

> mixed prompt 本身可能是当前最有潜力的方向，只是现有 validation 选择规则还没有把它稳定挑出来。

这通常意味着两种可能：

1. mixed prompt 的收益更依赖数据子集，稳定性还不够。
2. 当前 val selection 方式对这类 prompt 组合不够敏感。

不管是哪一种，`mixed prompt` 都值得继续做，而不是略过。

---

## 5. 方向 1B：多 prompt 聚合方式

### 5.1 实验设计

这一部分的目标是回答：

> 当一个类有多条 prompt 时，`average / max / top2_avg / logsumexp` 哪种聚合更好？

### 5.2 结果

这轮实验没有看到明确的聚合优势。

按 `best_by_aggregation` 统计：

- `average`
- `max`
- `top2_avg`
- `logsumexp`

在当前最优组合上给出的结果几乎一致，尤其在 `detailed` source 上完全重合。

### 5.3 解释

这里的原因并不神秘：

- 对 `short` 和 `detailed`，本身每类有效 prompt 数很少，aggregation 几乎没有自由度
- 对 `llm / mixed`，aggregation 也没有表现出稳定优于 `average` 的趋势

因此，本轮可以得出一个保守但明确的结论：

> 目前 aggregation 不是主要矛盾。

换句话说：

- 问题不在“同一组 prompt 怎样聚合”
- 而更在“到底应该给每个类用什么 prompt source / prompt set”

所以后续优先级应当是：

1. 继续改 prompt source / prompt composition
2. 暂时不要把太多精力花在 aggregation trick 上

---

## 6. 方向 1C：区分性 prompt

这一部分还没有正式实现，但从本轮结果看，优先级明显上升了。

原因是：

- 当前 `llm` source 并不占优
- `short / detailed / mixed` 反而更稳
- 说明“文本信息太多”不等于“文本信息更可判别”

而 hard-case 里已经反复出现这些受损类：

- `Attractive`
- `EnjoyLife`
- `FineDesignLearnArt-Art`
- `Playful` 过泛化

因此，下一步最合理的 prompt 改造方向不是盲目加更多描述，而是加入边界信息。

例如：

- `Playful`
  - 强调 active fun / teasing / childlike play
  - 明确排除 mere relaxation / comfortable life / generic happiness
- `Attractive`
  - 强调 appearance-driven visual cues
  - 明确区别于 generic happiness / social warmth
- `FineDesignLearnArt-Art`
  - 强调 artwork appreciation 本身
  - 明确区别于 broader culture / architecture / tourism

也就是说，这一轮结果其实在给一个很明确的信号：

> 接下来更值得做的是 `discriminative prompt engineering`，而不是继续堆 prompt 数量。

---

## 7. 方向 2：候选内相对比较 rerank

### 7.1 实现版本

这一方向做了两种 comparative prior：

1. `topk_center`

```text
s_tilde(c) = s_hat(c) - mean_{j in T_k}(s_hat(j))
```

2. `topk_margin`

```text
s_tilde(c) = s_hat(c) - max_{j != c, j in T_k}(s_hat(j))
```

二者都只在 `top-k` 候选内定义，候选外保持不变。

### 7.2 结果

按 `best_by_comparative_mode`：

| Comparative mode | Best config | test macro | test hard | 相对 SLR-v0 |
| --- | --- | ---: | ---: | --- |
| `none` | `detailed + average` | 0.4891 | 0.3198 | `macro +0.0048`, `hard +0.0319` |
| `topk_center` | `detailed + average` | 0.4859 | 0.3095 | `macro +0.0016`, `hard +0.0216` |
| `topk_margin` | `short + average` | 0.4871 | 0.3197 | `macro +0.0028`, `hard +0.0318` |

### 7.3 结果解读

这一步最重要的结论是：

> comparative rerank 没有稳定优于 plain local rerank。

更具体地说：

- `topk_center` 有时会稍微提升 `hard`，但 overall 优势不明显
- `topk_margin` 的 hard 表现接近 `none`，但也没有形成明显超越
- 当前最好结果仍然来自 `comparative = none`

所以这一方向目前的判断是：

- 它不是无效
- 但也不是当前最有潜力的主线

如果后续继续做 comparative，前提应该是：

- 先把 prompt source 做强
- 再回来看 relative comparison 是否还能提供额外信息

不建议现在把它升格为主要方法创新点。

---

## 8. 方向 3：very-light learnable fusion

这一部分在 `v1` 设计里被提出，但本轮没有实现。

当前不优先做它，原因有三点：

1. 现在已经有更低风险、且更有信号的方向：
   - prompt source
   - mixed prompt
   - discriminative prompt
2. 一旦引入 learnable fusion，就不再是纯推理期分析，会变成新的训练分支。
3. 目前还没有证据说明“fusion 形式不够好”是主瓶颈。

因此，这一部分保留为后续候选方向，但不作为下一步优先尝试。

---

## 9. 当前最重要的结论

这轮 `v1` 实验最重要的结论可以压缩成 4 句：

1. `prompt source` 比 `aggregation` 和 `comparative mode` 更重要。
2. 当前最稳的 source 不是 `llm`，而是 `short / detailed / mixed`。
3. `mixed prompt` 很有潜力，甚至在 test 上出现了目前最强结果，但还需要更稳的 validation 选择与多 seed 验证。
4. `comparative rerank` 没有形成稳定优势，暂时不是主线。

---

## 10. 下一步建议

基于这轮结果，后续优先级建议如下。

### 10.1 第一优先级

继续做 `prompt source / prompt composition`

具体是：

1. 把 `mixed prompt` 做得更规范
   - 明确哪些 prompt 该进混合池
   - 不要只是简单堆文本
2. 加 `discriminative / boundary prompt`
   - 专门修 `Attractive / EnjoyLife / FineDesignLearnArt-Art / Playful`
3. 对 `short / detailed / mixed` 做多 seed 稳定性验证

### 10.2 第二优先级

重新审视 validation 选型规则

因为当前已经观察到：

- 按 `val_macro` 选出来的是 `detailed + average + none`
- 但 test 上实际最强的一组是 `mixed + average + none`

这意味着当前 `val` 规则未必最适合 semantic local rerank。

至少可以补看：

- `val hard`
- `macro + lambda * hard`
- 多 seed 平均排序

### 10.3 暂缓项

暂缓以下方向作为主线：

1. `comparative rerank`
2. `aggregation trick`
3. `learnable fusion`

它们现在都没有表现出比 prompt source 更强的价值密度。

---

## 11. 当前状态

这份 `v1` 文档不再是脑暴草稿，而是已有实现和结果支撑的实验记录。

已完成：

- semantic local rerank 实现
- source / aggregation / comparative 三类变体
- smoke test
- 全量实验

对应输出：

- `logs/analysis/full_semantic_local_rerank_20260309/summary.json`
- `logs/analysis/full_semantic_local_rerank_20260309/semantic_local_rerank_leaderboard.csv`

当前最值得继续推进的，不是更复杂的 gate 或 fusion，而是：

> 把 prompt 本身做成更有边界感、更适合 local rerank 的 semantic prior。
