# 实验记录：Agent Evidence Verification v2

## 0. 文档信息

- 日期：`2026-03-12`
- 对应设计：`docs/design/design_0311_03_AgentEvidenceVerification_v2.md`
- 任务：在 `scenario SLR-C` 基础上继续尝试更强的 candidate-local verification
- 本次重点：
  - pairwise / comparative verification
  - margin-aware gate
  - confusion-neighborhood negatives

---

## 1. 本次实现内容

相对 v1，本次真正落地了三件新东西：

1. **pairwise comparative verification**
   - 不再只给单个候选打绝对 verification residual
   - 改成对 top-k 候选两两比较
   - 比较时只看 pair-specific discriminative evidence profile

2. **margin-aware gate**
   - 用 baseline top-1 / top-2 margin 控制 verification 强度
   - 目标是只在“候选本来就接近”的样本上让 verifier 强介入

3. **confusion-neighborhood negatives**
   - 不再只用 prototype similarity 的 hard negatives
   - 额外实现了基于 `scenario SLR` top-k confusion 统计得到的 class-specific confusion neighborhood

对应代码：

- `src/utils/evidence_verification.py`
  - `build_confusion_neighborhoods`
  - `build_pairwise_relation_profiles`
  - `compute_pairwise_comparative_scores`
  - `build_margin_aware_gate`
- `scripts/analyze_agent_evidence_verification_v2.py`
  - v2 comparative verification 分析脚本
- `tests/test_evidence_verification_v2.py`
  - v2 核心逻辑单测

---

## 2. 实验设置与稳定性说明

### 2.1 为什么这次没有跑完整大网格

这次尝试过程中出现了多次 OOM 与系统重启。

因此我没有继续扩 full-space search，而是收缩到一个**最小但足够回答 v2 核心问题**的实验：

- `expert subset = all`
- `relation variant`：
  - `hard_negative_diff`
  - `confusion_hard_negative_diff`
- `profile_topn = 20`
- `activation_topm = 5`
- `pair_profile_topn = 5 / 10`
- `beta = 0.08 / 0.1 / 0.15 / 0.2`
- `gate_gamma`
  - `0 / 1 / 2 / 4` for `add_norm`
  - `0 / 1 / 2 / 4 / 8` for `add`

目标很明确：

> 先回答 design v2 最核心的问题：  
> **pairwise comparative verification + margin-aware gate**，能否在 `scenario SLR-C` 基础上进一步超过 v1 focused verifier。

### 2.2 产物目录

本次有两组关键产物：

1. `add_norm` sanity run
   - `logs/analysis/min_agent_evidence_verification_v2_comparative_20260311`
2. `add` gate-effective run
   - `logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312`

---

## 3. 先说结论

本次 v2 的结论很清楚：

1. **pairwise comparative verification 是有价值的**
2. **margin-aware gate 只有在 `fusion=add` 下才真正起作用**
3. **`confusion_hard_negative_diff` 这轮没有超过普通 `hard_negative_diff`**
4. 在当前最小稳定实验里，v2 已经比 v1 focused 再往前走了一步

最强的 v2 validation-best 结果来自：

- relation：`hard_negative_diff`
- pair profile：`top-5`
- gate gamma：`2.0`
- beta：`0.01`
- fusion：`add`

它的 test class-wise 结果为：

- macro：`51.73`
- micro：`59.22`
- samples：`58.19`
- mAP：`53.78`
- hard：`35.83`

---

## 4. 关键结果对比

### 4.1 与 SLR-C / v1 focused / v2 add_norm 的主对比

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` | 51.28 | 59.13 | **58.47** | 53.66 | 33.98 |
| v1 focused verifier | 51.35 | 59.06 | 58.12 | 53.58 | 35.24 |
| v2 comparative + gate (`add_norm`) | 50.41 | 58.22 | 57.51 | 53.65 | 34.68 |
| v2 comparative + gate (`add`) | **51.73** | **59.22** | 58.19 | **53.78** | **35.83** |

### 4.2 结果解读

相对 `scenario SLR-C`：

- macro：`+0.45`
- micro：`+0.09`
- samples：`-0.28`
- mAP：`+0.12`
- hard：`+1.86`

相对 v1 focused verifier：

- macro：`+0.38`
- micro：`+0.16`
- samples：`+0.07`
- mAP：`+0.20`
- hard：`+0.59`

因此本次最重要的结论是：

> **v2 的 pairwise comparative verification 在 `scenario SLR-C` 上确实比 v1 focused verifier 更强。**

虽然 samples 还没超过纯 `SLR-C`，但 macro / micro / mAP / hard 四项都已经更好。

---

## 5. v2 内部分析

### 5.1 `add_norm` 为什么不对

`add_norm` 那组实验一开始看起来很奇怪：

- 所有 `gate_gamma` 的结果几乎完全一样
- 说明 gate 没有真正影响最终 rerank

原因在于：

1. gate 先把 verification score 按样本整体缩放
2. `add_norm` 再对每个样本做 z-score normalization
3. 于是这个整体缩放被抵消掉了

结论：

> **margin-aware gate 不能和当前的 `add_norm` 直接搭配。**

所以 design v2 里 gate 的正确实验方式是：

> `fusion=add`，让 gate 直接控制 residual 强度

### 5.2 `add` 下 gate 变得有效

`fusion=add` 后，best validation-best row 是：

- relation：`hard_negative_diff`
- pair_profile_topn：`5`
- gate_gamma：`2`
- beta：`0.01`

而 best test row 则来自：

- relation：`hard_negative_diff`
- pair_profile_topn：`10`
- gate_gamma：`1`
- beta：`0.01`

它的 test 指标为：

- macro：`52.61`
- hard：`36.01`

这个结果不能作为正式主结果，因为不是 validation-best。  
但它很有价值，因为它说明：

> pairwise compare + moderate gate 确实还有上探空间。

### 5.3 confusion neighborhood 没赢

这次 `confusion_hard_negative_diff` 的 best row 为：

- macro：`50.99`
- hard：`33.55`

明显弱于普通 `hard_negative_diff` 的：

- macro：`51.73`
- hard：`35.83`

因此本轮的结论是：

> **在当前最小实验下，confusion-neighborhood negatives 没有带来额外收益。**

这不一定说明方向错了，更可能说明：

- 当前 confusion neighborhood 构造得还太粗
- 或者它和 pairwise compare 的增益重合了

---

## 6. Diagnostics

### 6.1 Candidate recall / oracle

由于 Stage 0 仍是同一个 `scenario SLR-C`，candidate recall 与 v1 一致：

- top-10 label recall：`85.36%`
- sample-any recall：`95.72%`
- sample-all recall：`77.14%`
- top-10 oracle macro：`88.75`

所以 v2 仍然是在解决同一个问题：

> 候选已经召回到了，剩下的是 candidate-local ranking。

### 6.2 verification gap

best v2 (`add`) 的 comparative verification gap：

- correct candidate mean：`5.52`
- wrong candidate mean：`-1.09`
- gap：`+6.61`
- sample mean gap：`+6.91`

这说明 pairwise comparative verifier 依然有明显判别力。

### 6.3 correlation

best v2 的 correlation：

- Pearson(all classes)：`0.152`

这比 v1 focused add-style residual 更低一些，说明 comparative 版本在一定程度上确实减少了与 base score 的共线性。

---

## 7. Per-class gains

对 `scenario SLR-C`，best v2 的最大增益类别包括：

- `FineDesignLearnArt-Art`：`+10.49`
- `Communicate`：`+4.10`
- `SuccInOccupHavGdJob`：`+3.89`
- `InLoveAnimal`：`+3.76`
- `Attractive`：`+2.48`

这些类别的共同特点是：

- 本来就容易在 top-k 内与近邻 intent 混淆
- 更适合用 pairwise evidence compare 去拉开

这和 design v2 的出发点是一致的。

---

## 8. 最终结论

本次 v2 尝试得出的最重要结论有三条：

1. **pairwise comparative verification 比 v1 的单类 residual 更接近问题本质**
   - 因为它直接回答的是 “A 比 B 为什么更合理”

2. **margin-aware gate 是有效的，但前提是不能被 `add_norm` 抵消**
   - `add_norm` 下 gate 基本无效
   - `add` 下 gate 才真正产生收益

3. **在当前最小稳定实验里，v2 已经优于 v1 focused verifier**
   - v1 focused：`51.35 / 59.06 / 58.12 / 53.58 / 35.24`
   - v2 best：`51.73 / 59.22 / 58.19 / 53.78 / 35.83`

因此，`design_0311_03_AgentEvidenceVerification_v2.md` 的这条方法线现在可以明确更新为：

> **值得继续。**

但继续的重点不是把 v2 再做大，而是把下面两点做扎实：

1. 保留 `hard_negative_diff + pairwise compare + add-gate`
2. 单独重做 `support_contradiction` 的稳定实验

因为当前最合理的判断是：

> v2 的核心增益已经出现，但还没有把全部设计空间稳定吃满。  
> 现阶段最稳妥的正式方法版本，应该从 `hard_negative_diff` 这一支先收敛出来。
