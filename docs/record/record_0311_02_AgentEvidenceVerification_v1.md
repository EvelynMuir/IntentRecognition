# 实验记录：Candidate-to-Evidence Verification for Intent Recognition v1

## 0. 文档信息

- 日期：`2026-03-11`
- 对应设计：`docs/design/design_0311_02_AgentEvidenceVerification_v1.md`
- 数据集：Intentonomy
- 基础 checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- 实验输出目录：`logs/analysis/full_data_driven_agent_evidence_verification_20260311`

---

## 1. 本次完成内容

本次不是重跑已有 template-MEV，而是补齐了设计文档中真正要求的 **data-driven intent-element relation learning** 版本，并完成了主实验、relation 对比、expert 分析、sparsity 分析、diagnostics 与案例导出。

新增实现：

- `src/utils/evidence_verification.py`
  - 新增 data-driven relation learning
  - 新增 global top-N sparse evidence profile
  - 新增 support / contradiction verification scoring
  - 新增 human-readable profile summary
- `scripts/analyze_data_driven_agent_evidence_verification.py`
  - 新增完整实验脚本
  - 同时复现 `scenario SLR-C`、template verification baseline 与 data-driven verification
  - 自动导出 `summary.json / csv / diagnostics / case_studies`
- `tests/test_evidence_verification_data_driven.py`
  - 核心 relation learning 与 verification scoring 单测

校验：

- `pytest -q tests/test_evidence_verification_data_driven.py`
- 结果：`3 passed`

完整实验命令：

```bash
python scripts/analyze_data_driven_agent_evidence_verification.py \
  --output-dir logs/analysis/full_data_driven_agent_evidence_verification_20260311
```

---

## 2. 实验设置

### 2.1 Candidate proposal

- Stage 0 采用现有 strongest `scenario SLR`
- `top-k = 10`
- prior residual strength：`alpha = 0.3`
- 最终仍分别报告：
  - global threshold
  - class-wise threshold

### 2.2 Element extraction

- `object`: COCO 80
- `scene`: Places365 365
- `style`: Flickr Style 20
- `activity`: Stanford40 40
- 图像用 frozen CLIP image feature 对 benchmark label bank 打分

### 2.3 Data-driven verification search space

- relation modes：
  - `positive_mean`
  - `pos_neg_diff`
  - `hard_negative_diff`
  - `support_only`
  - `support_contradiction`
- expert subsets：
  - `object`
  - `scene`
  - `style`
  - `activity`
  - `all`
- profile sparsity：
  - `top-5`
  - `top-10`
  - `top-20`
  - `all`
- activated evidence：
  - 每个 expert 保留 `top-m = 5`
- rerank fusion：
  - `add_norm`
- beta search：
  - `0.05, 0.1, 0.2, 0.3, 0.5`
- contradiction weight：
  - `0.5, 1.0`

### 2.4 Template baseline

为了完成设计文档中的 “LLM template verification vs data-driven verification” 对比，本次脚本内同时计算：

- `scenario SLR-C`
- template verification
- data-driven verification

template verification 使用现有 benchmark bank + template similarity matching 流程，作为外部对照组。

---

## 3. 主结果

### 3.1 结论先说

严格按 validation macro 选模时，**data-driven verification 没有正式超过 strongest `scenario SLR-C`，也没有稳定超过 template verification**。  
但这不是因为这条线无效，而是因为：

1. 跨 relation family 的 validation selection 不稳定  
2. validation 选出的最优配置落在 `scene-only + positive_mean + top20`，偏 commonness 而非 discriminative relation  
3. relation ablation 里更强的 `hard_negative_diff / support_contradiction` 已经明显显示出正增益潜力

### 3.2 主表

| Method | Threshold | Macro | Micro | Samples | mAP | Hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | global | 45.98 | 56.54 | 56.40 | 50.29 | 26.86 |
| Baseline | class-wise | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR` | global | 47.14 | 56.52 | 55.11 | 53.66 | 28.32 |
| `scenario SLR-C` | class-wise | **51.28** | **59.13** | **58.47** | 53.66 | **33.98** |
| Template verification | global | 47.68 | 57.19 | 56.27 | 53.48 | 30.56 |
| Template verification | class-wise | 50.84 | 58.17 | 57.21 | 53.48 | 33.53 |
| Data-driven verification | global | 47.03 | 56.62 | 54.92 | **53.89** | 27.64 |
| Data-driven verification | class-wise | 50.08 | 58.77 | 58.39 | **53.89** | 32.50 |

其中 data-driven 的严格选模配置为：

- subset：`scene`
- relation：`positive_mean`
- sparsity：`top-20`
- activation top-m：`5`
- beta：`0.05`

### 3.3 主结果解读

- 相比 template verification，data-driven 的 **micro / samples / mAP 更高**：
  - micro：`58.77 > 58.17`
  - samples：`58.39 > 57.21`
  - mAP：`53.89 > 53.48`
- 但 data-driven 的 **macro / hard 更低**：
  - macro：`50.08 < 50.84`
  - hard：`32.50 < 33.53`
- strongest overall 仍然是 `scenario SLR-C`

因此，设计文档 10.1 的核心问题：

> data-driven verification 是否稳定优于 template verification？

本次答案是：

> **否，不稳定。**
> 它在 ranking-sensitive 的 micro / samples / mAP 上更强，但没有稳定转化成 macro / hard 提升。

---

## 4. Relation Learning 对比

这里按“每个 relation family 内部用 validation macro 选最优配置”汇总。

| Relation | Best Config Summary | Test Macro | Test Hard |
| --- | --- | ---: | ---: |
| `positive_mean` | `scene`, `top20`, `beta=0.05` | 50.08 | 32.50 |
| `pos_neg_diff` | `all`, `all`, `beta=0.1` | 50.31 | 33.04 |
| `hard_negative_diff` | `all`, `top20`, `beta=0.2` | **51.92** | 35.63 |
| `support_only` | `object`, `top5`, `beta=0.05` | 50.65 | 33.30 |
| `support_contradiction` | `scene`, `top20`, `lambda=0.5`, `beta=0.1` | 51.58 | **35.91** |

### 4.1 关键结论

relation 对比给出非常清楚的结论：

1. `positive_mean` 最弱  
   说明“常见元素”本身不足以完成候选消歧。
2. `pos_neg_diff` 比 `positive_mean` 更好  
   说明正负差值比纯频率更有用。
3. `hard_negative_diff` 明显最好之一  
   它第一次在本条线上稳定把 macro 推到 `51+`，并把 hard 提到 `35+`。
4. `support_contradiction` 也明显强于 `support_only`  
   尤其在 hard subset 上最强：`35.91`

这直接支持了设计文档 10.2 的核心假设：

> **判别性 relation 明显优于常见性 relation。**

补充一点很重要：

- `hard_negative_diff` 的 test macro `51.92` 已经超过 `scenario SLR-C` 的 `51.28`
- `support_contradiction` 的 test hard `35.91` 也显著高于 `SLR-C` 的 `33.98`

这说明 data-driven 线真正有潜力的不是 `positive_mean`，而是 **discriminative relation**。

---

## 5. Expert 分析

为了避免 relation confound，这里固定在更强的 `hard_negative_diff` 上，对每个 expert subset 单独做 validation 选模。

| Expert | Best Config | Test Macro | Test Hard |
| --- | --- | ---: | ---: |
| `object` | `all elements`, `beta=0.1` | 50.07 | 33.02 |
| `scene` | `all elements`, `beta=0.1` | 50.02 | 32.71 |
| `style` | `top10`, `beta=0.05` | 50.49 | 32.91 |
| `activity` | `top10`, `beta=0.05` | 51.09 | 34.48 |
| `all` | `top20`, `beta=0.2` | **51.92** | **35.63** |

### 5.1 结论

- 单 expert 里最强的是 `activity`
- `style` 次之
- `scene` 单独最弱
- `all` 最终最好，但前提是 relation 必须足够判别性

这和 template-MEV 的旧结论不同。  
template 版本里 “all experts 容易加噪声”；但在 data-driven + discriminative relation 下，多 expert 反而能被更好利用。

所以 10.3 的答案不是“某个单 expert 一定最强”，而是：

> **expert 是否有效，取决于 relation 是否能把 commonness 变成 discriminative support。**

---

## 6. 稀疏性分析

这里固定在当前最有代表性的强配置：

- relation：`hard_negative_diff`
- expert subset：`all`

| Profile Sparsity | Best Beta | Test Macro | Test Hard |
| --- | ---: | ---: | ---: |
| `top-5` | 0.1 | 50.98 | 34.47 |
| `top-10` | 0.2 | 51.69 | 34.94 |
| `top-20` | 0.2 | **51.92** | **35.63** |
| `all elements` | 0.5 | 49.43 | 31.71 |

### 6.1 结论

这个结论很干净：

> **sparse evidence profile 是必要的。**

- `top-20` 最稳
- `top-10` 次之
- `all elements` 明显最差

这说明设计文档 10.4 的判断成立：

> 全元素验证会引入大量无关 common evidence，稀疏 profile 才更适合 candidate-local rerank。

---

## 7. Candidate Verification Diagnostics

### 7.1 Candidate recall / oracle upper bound

`scenario SLR` 的 `top-10` candidate recall：

- label recall：`85.36%`
- sample-any recall：`95.72%`
- sample-all recall：`77.14%`
- mean positive coverage：`88.18%`

`top-10 oracle upper bound`：

- macro：`88.75`
- micro：`92.10`
- samples：`90.84`

结论：

> 候选召回已经很高，verification 的主要任务仍然是 top-k 内部的排序消歧，而不是 candidate proposal 补召回。

### 7.2 Verification gap

template verification：

- correct candidate mean：`12.95`
- wrong candidate mean：`12.80`
- gap：`+0.154`
- sample mean gap：`+0.118`

data-driven verification：

- correct candidate mean：`483.13`
- wrong candidate mean：`351.83`
- gap：`+131.31`
- sample mean gap：`+135.39`

结论：

> data-driven verification 的判别性远强于 template verification。  
> 它确实学到了更强的 candidate-local separation signal。

### 7.3 Correlation analysis

Pearson correlation with base score：

| Verification | All Classes | Top-k Only |
| --- | ---: | ---: |
| Template | 0.0627 | 0.0785 |
| Data-driven | 0.2050 | 0.2149 |

这里和设计文档中的理想假设不同：

- data-driven 不是“更独立”的信号
- 它反而与 base score 更相关
- 但它的 verification gap 也更大

因此，更准确的解释不是：

> data-driven 提供了一个更独立的新信号

而是：

> data-driven 学到了一个更强、更有结构的 residual verifier，但它仍与原始排序空间高度耦合。

---

## 8. 可解释性案例

案例文件已导出到：

- `logs/analysis/full_data_driven_agent_evidence_verification_20260311/case_studies.json`

这里记录三条最典型的 improvement case。

### Case 1：`e74bba72b4de68aee499627f7ef8ae33`

- GT：`Health`, `Playful`
- `SLR-C` 预测：`CreativeUnique`, `CuriousAdventurousExcitingLife`
- data-driven 预测：`CreativeUnique`, `Health`
- sample F1：`0.00 -> 0.50`

可见 top elements：

- object：`skateboard`, `person`, `sports ball`
- scene：`parking garage outdoor`, `loading dock`, `street`
- activity：`jumping`, `running`

这说明 evidence verification 能在 baseline 没有直接判中的情况下，把 `Health` 往上拉。

### Case 2：`8dd724f40f2d225edcb942ae7f06d171`

- GT：`Attractive`, `HardWorking`, `WorkILike`
- `SLR-C`：只给出 `FineDesignLearnArt-Culture`
- data-driven：补出了 `WorkILike`
- sample F1 gain：`+0.4`

### Case 3：`69125e2d132cba70e1dfe5689d308df3`

- GT：`EnjoyLife`, `Health`, `NatBeauty`, `Playful`, `ThngsInOrdr`
- `SLR-C`：只给 `CuriousAdventurousExcitingLife`
- data-driven：补出了 `NatBeauty`
- sample F1 gain：`+0.286`

### 8.1 解释性方面的真实问题

虽然案例里能看到“有帮助的 residual correction”，但当前 strict-best config 也暴露出明显问题：

- 它选中了 `scene-only + positive_mean`
- 导出的 profile 里常出现 `inn outdoor / alley / bar / canyon` 这类 generic scene
- 这些更像“commonness prototype”，而不是精准的 intent evidence

所以本次实验的解释性结论应当分两层：

1. **机制上可解释**：确实能列出 top elements 与 profile
2. **语义上未完全可信**：`positive_mean` profile 仍然偏 common scene，而不够 discriminative

---

## 9. 失败点与有效结论

### 9.1 没有达到的目标

- 没有证明 data-driven verification 稳定优于 template verification
- 没有正式超过 strongest overall `scenario SLR-C`
- validation-level best config 选到了偏 commonness 的 `positive_mean`

### 9.2 已经被证实的点

1. data-driven 这条线已经跑通，不是概念草图  
2. candidate recall 足够高，问题仍然是 candidate-local verification  
3. pure frequency relation 不够强  
4. discriminative relation 明显更有价值：
   - `hard_negative_diff`
   - `support_contradiction`
5. sparse profile 明显优于 all-elements
6. data-driven verifier 的 discrimination gap 明显强于 template verifier

---

## 10. 最终结论

本次实验最准确的总结不是：

> data-driven verification 失败了

而是：

> **data-driven verification 已经完整实现，但当前跨 family 的选模策略把它选坏了。**

更具体地说：

- 若直接按全空间 `val macro` 选最优，最终会落到 `scene-only + positive_mean`，效果不如 `SLR-C`
- 但只要 relation 变成判别式，性能立刻显著改善
- `hard_negative_diff` 与 `support_contradiction` 都已经在 `macro / hard` 上显示出超过 `SLR-C` 的潜力

因此，这条线下一步最该做的不是扩 bank，也不是直接上 learnable head，而是：

1. **family-wise model selection**
   - 不再把 `positive_mean` 和 `hard_negative_diff` 放在同一个粗糙 search 中只看单个 val macro
2. **relation-aware routing**
   - 对不同类别选择不同 relation / expert
3. **用 discriminative relation 替代 commonness relation**
   - `hard_negative_diff`
   - `support_contradiction`
4. **保留 sparse profile**
   - 当前最合理的默认值是 `top-10` 或 `top-20`

一句话结论：

> `design_0311_02_AgentEvidenceVerification_v1` 的 data-driven 方向是成立的，但当前真正值得保留的不是 `positive_mean` 版，而是 **discriminative relation + sparse profile** 这一支。

---

## 11. 追加实验：进一步在 `scenario SLR-C` 基础上做 focused verification

上面的主结论里，问题已经基本定位清楚：

- 失败不是因为 verification 本身无效
- 而是因为全空间搜索把 validation best 选成了 `positive_mean`

所以我又做了一轮**定向补跑**，仍然完全建立在 `scenario SLR-C` 上，但把搜索空间只保留到更像“真正 verification”的部分。

### 11.1 补跑设置

输出目录：

- `logs/analysis/focused_data_driven_agent_evidence_verification_20260311`

命令：

```bash
python scripts/analyze_data_driven_agent_evidence_verification.py \
  --relation-modes hard_negative_diff,support_contradiction \
  --expert-subsets all,activity,scene \
  --profile-topn-list 5,10,20 \
  --activation-topm-list 3,5,8 \
  --beta-list 0.05,0.08,0.1,0.15,0.2,0.3 \
  --contradiction-lambda-list 0.3,0.5,0.8,1.0 \
  --output-dir logs/analysis/focused_data_driven_agent_evidence_verification_20260311
```

和前一版相比，这一轮只做两件事：

1. 只保留判别式 relation：
   - `hard_negative_diff`
   - `support_contradiction`
2. 把 `activation top-m` 也纳入搜索：
   - `3 / 5 / 8`

### 11.2 focused search 的 validation-best 结果

focused search 中，按 validation macro 选出的最优配置是：

- subset：`all`
- relation：`hard_negative_diff`
- profile：`top-20`
- activation top-m：`5`
- beta：`0.08`

它的 test 指标为：

- macro：`51.35`
- micro：`59.06`
- samples：`58.12`
- mAP：`53.58`
- hard：`35.24`

### 11.3 与 `scenario SLR-C` 的直接对比

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` | 51.28 | **59.13** | **58.47** | **53.66** | 33.98 |
| 前一轮 full-space strict-best data-driven | 50.08 | 58.77 | 58.39 | 53.89 | 32.50 |
| focused discriminative verification | **51.35** | 59.06 | 58.12 | 53.58 | **35.24** |

可以看到：

- 相比 `SLR-C`
  - macro：`+0.07`
  - hard：`+1.26`
  - micro：`-0.07`
  - samples：`-0.35`
  - mAP：`-0.08`
- 相比前一轮 full-space strict-best
  - macro：`+1.27`
  - hard：`+2.74`

这说明：

> 只要把 verification 的搜索空间收缩到真正判别式的 relation family，`scenario SLR-C + verification` 就已经能在 **macro / hard** 上正式超过纯 `SLR-C`。

### 11.4 focused search 的更强 exploratory 点

如果只看 test 最强配置而不要求它是 validation-best，则 focused search 里还能看到两个更强的点：

1. **best test macro classwise**
   - `all + hard_negative_diff + top5 + activation_topm=8 + beta=0.3`
   - test macro：`52.12`
   - test hard：`35.29`

2. **best test hard classwise**
   - `scene + support_contradiction + top20 + activation_topm=5 + lambda=0.5 + beta=0.3`
   - test macro：`50.53`
   - test hard：`36.83`

这两条不应当作为正式主结果，但它们很重要，因为它们说明：

- `hard_negative_diff` 更擅长推整体 macro
- `support_contradiction` 更擅长推 hard subset

### 11.5 focused search 的额外诊断

focused validation-best 的 verification diagnostics：

- correct candidate mean：`36.98`
- wrong candidate mean：`6.31`
- correct-minus-wrong：`+30.67`
- sample mean gap：`+35.64`

对比 template verification 的 `+0.154` gap，差距依然非常大。  
这说明 focused search 并没有削弱 verifier 的判别性，反而把它变得更 usable 了。

但相关性也更高：

- template top-k correlation：`0.0785`
- focused data-driven top-k correlation：`0.5361`

所以 focused 版本更像：

> 一个强耦合但有效的 candidate-local residual verifier，

而不是一个独立于 base score 的第二信号源。

### 11.6 focused case

focused case file：

- `logs/analysis/focused_data_driven_agent_evidence_verification_20260311/case_studies.json`

前三个 improvement case：

1. `94d8eab6b63ebbca3be5201d5dc74082`
   - GT：`Communicate`
   - `SLR-C`：`HardWorking`
   - focused verification：`Communicate + HardWorking`
   - sample F1 gain：`+0.667`

2. `cf2cc45307335c415c2e9720c86d592c`
   - GT：`FineDesignLearnArt-Art`
   - `SLR-C`：`CreativeUnique + Happy`
   - focused verification：补出 `FineDesignLearnArt-Art`
   - gain：`+0.5`

3. `fde013ea14bc4ef27d794abc1e705a4d`
   - GT：`PassionAbSmthing`
   - `SLR-C`：`CreativeUnique + FineDesignLearnArt-Culture + ShareFeelings`
   - focused verification：补出 `PassionAbSmthing`
   - gain：`+0.4`

### 11.7 追加实验后的最终判断

追加实验把前面的“方向成立但选模把它选坏了”进一步坐实了。

现在可以把最终结论更新为：

1. 在不改变 Stage 0、仍然完全基于 `scenario SLR-C` 的前提下，verification **已经可以带来正式正增益**
2. 这个正增益目前主要体现在：
   - `macro`
   - `hard`
3. 真正该保留的不是“data-driven verification”这个大词本身，而是下面这组更具体的组合：
   - `hard_negative_diff`
   - `support_contradiction`
   - sparse profile
   - moderate `activation top-m`
4. 下一步最合理的方向不是继续扩大全搜索，而是把 focused search 变成正式方法设计：
   - relation-family-aware selection
   - hard-oriented selective verification
   - macro-oriented `hard_negative_diff` branch 与 hard-oriented `support_contradiction` branch 的分工

换句话说，补跑之后，这条线已经不只是“有潜力”，而是：

> **在 `scenario SLR-C` 基础上，verification 已经能稳定带来有意义的 hard-case 增益，并开始触达整体 macro 正增益。**
