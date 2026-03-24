# 实验记录：Retrieval-Based Ambiguity Agent

## (a) main conclusions

1. 本轮严格按 design 先完成了 `Phase 1/2` 的 image-only retrieval MVP：
   - memory = `train` image embedding + labels
   - query = `val/test` image embedding
   - similarity = cosine
   - proposer 固定为 `scenario SLR-C`
   - `v2 comparative + gate(add)` 保持为固定 reference
   没有进入 `Phase 3 evidence-aware retrieval`。
2. 按 validation macro 保守选模时，retrieval MVP **没有超过**固定 reference `v2 comparative + gate`，也没有正式超过纯 `SLR-C`：
   - `Phase 1 image-only best`：Macro `50.73`, Hard `33.97`
   - `Phase 2 confusion-aware best`：Macro `50.92`, Hard `34.36`
   - `scenario SLR-C`：Macro `51.28`, Hard `33.98`
   - `v2 comparative + gate`：Macro `51.73`, Hard `35.83`
3. retrieval 方向仍然有明确的 **candidate-local ambiguity signal**：
   - `confusion-aware retrieval` 的 low-margin subset Hard = `36.07`，高于 `SLR-C` 的 `35.07`
   - `top-2 disambiguation accuracy = 81.37`，高于 `SLR-C` 的 `80.32`，也高于 `v2` 的 `80.18`
   - 但整体 Macro / Micro / Samples 仍未形成稳定正收益
4. `support - confusion-aware refute` 相比 `support - global refute` 有小幅但可解释的增益：
   - validation-best row 下 Macro 几乎持平：`50.92` vs `50.92`
   - Hard 更好：`34.36 > 34.09`
   - top-2 disambiguation 也更好：`81.37 > 81.11`
   这说明 retrieval 的有效部分更像 candidate-local refute，而不是粗粒度 global negative。
5. 从整轮 sweep 看，retrieval 仍有一些“test-only hindsight”正信号，但不足以支撑继续加复杂度：
   - best test Macro：`support - global refute`, `r=5`, `beta=0.02` -> `51.40`
   - best test Hard：`support - confusion refute`, `r=8`, `beta=0.05` -> `35.22`
   这些点说明 retrieval 不是完全没潜力，但它们都不是 validation-best，因此当前还不能作为正式主结果。
6. 由于 `Phase 1/2 image-only retrieval` 已经触发 design 的停止条件：
   - 没有超过当前 strongest `v2 comparative + gate`
   - dominant confusion pair subset 也没有形成强改进
   所以本轮停在 `Phase 2`，不进入 `Phase 3 evidence-aware retrieval`。

## (b) main result table

主表来自：`logs/analysis/retrieval_ambiguity_agent_20260312/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` | 51.28 | 59.13 | **58.47** | 53.66 | 33.98 |
| `v2 comparative + gate(add)` | **51.73** | **59.22** | 58.19 | **53.78** | **35.83** |
| `Retrieval agent Phase 1 / image-only best` | 50.73 | 58.35 | 57.80 | 53.50 | 33.97 |
| `Confusion-aware retrieval best` | 50.92 | 58.39 | 57.61 | 53.56 | 34.36 |

## (c) detailed results and analysis

### 1. 实现内容与固定设置

本轮新增实现：

- `src/utils/retrieval_ambiguity.py`
  - `build_retrieval_memory_indices`
  - `compute_similarity_matrix`
  - `compute_classwise_topk_mean_similarity`
  - `build_retrieval_evidence_scores`
- `scripts/analyze_retrieval_ambiguity_agent.py`
  - 固定 `scenario SLR-C`
  - 固定 `v2 comparative + gate(add)` reference
  - 复用现有 `_cache` bundles 与共享评测 helper
  - 跑 `support only / support - global refute / support - confusion-aware refute`
  - 跑 `r in {3,5,8}` 与小范围 `beta` sweep
  - 输出主表、setting ablation、`r` ablation、subset diagnostics、memory stats
- `tests/test_retrieval_ambiguity.py`
  - memory pool construction
  - chunked similarity
  - top-r mean similarity + evidence subtraction

固定不动的部分：

- checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- cache：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`
- proposer：`scenario SLR-C`
- `topk = 10`
- `alpha = 0.3`
- retrieval gate：`margin-aware exp(gamma = 2.0)`
- retrieval residual fusion：`add`
- retrieval evidence：先算 candidate-wise score，再只在 current top-k candidate 内做 z-score normalization 后加回 base logits

固定 `v2` reference：

- relation：`hard_negative_diff`
- experts：`object + scene + style + activity`
- `profile_topn = 20`
- `pair_profile_topn = 5`
- `activation_topm = 5`
- `hard_negative_topn = 3`
- `gate_gamma = 2.0`
- `beta = 0.01`

retrieval search space：

- setting：
  - `support_only`
  - `support_minus_global_refute`
  - `support_minus_confusion_refute`
- `r ∈ {3,5,8}`
- `beta ∈ {0.005, 0.01, 0.02, 0.05}`

memory 规模：

- support pool 平均大小：`850.93`
- global refute pool 平均大小：`11888.07`
- confusion-aware refute pool 平均大小：`3282.46`
- dominant confusion pair：`NatBeauty -> Harmony`
  - train count：`748`
  - test subset size：`96`

### 2. Setting ablation

best-by-setting 来自：`logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_setting_ablation.csv`

| Retrieval Setting | Best `r` | Best `beta` | Macro | Hard |
| --- | ---: | ---: | ---: | ---: |
| `support only` | 3 | 0.05 | 50.73 | 33.97 |
| `support - global refute` | 3 | 0.05 | 50.92 | 34.09 |
| `support - confusion refute` | 3 | 0.05 | 50.92 | 34.36 |

这张表说明：

1. 只看 support exemplars 并不够。
2. 明确减去 refute evidence 后，Hard 才开始稳定上升。
3. `confusion-aware refute` 相比 `global refute` 的增益不大，但方向是对的，主要体现在 Hard / top-2 消歧，而不是总体 Macro。

### 3. `r` ablation

best-by-`r` 来自：`logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_r_ablation.csv`

| `r` | Validation-best setting at this `r` | Macro | Micro | Hard |
| --- | --- | ---: | ---: | ---: |
| 3 | `support - confusion refute`, `beta=0.05` | 50.92 | 58.39 | 34.36 |
| 5 | `support - confusion refute`, `beta=0.05` | 50.92 | 58.55 | **34.42** |
| 8 | `support - confusion refute`, `beta=0.02` | **51.16** | **58.97** | 34.14 |

补充说明：

- 如果只看 validation-best path，`r=8` 最接近 `SLR-C` 的整体 Macro，但仍然低于 `SLR-C` 的 `51.28`
- 如果 hindsight 地只看 test：
  - best test Macro = `51.40`，来自 `support - global refute`, `r=5`, `beta=0.02`
  - best test Hard = `35.22`，来自 `support - confusion refute`, `r=8`, `beta=0.05`
- 这些信号说明 retrieval 在 hardest ambiguity 上并非无效，但当前还没有形成稳定、可保守汇报的 validation-leading 结果

### 4. Hard subset / pairwise diagnostics

#### 4.1 low-margin subset

`margin < 1.0` 子集大小：`802`

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 45.47 | 48.73 | 35.07 |
| `v2 comparative + gate` | **46.03** | **49.07** | **37.07** |
| `support only best` | 45.09 | 48.67 | 35.54 |
| `support - global refute best` | 45.20 | **48.79** | 35.67 |
| `support - confusion refute best` | 45.31 | 48.78 | 36.07 |

解释：

1. retrieval 最真实的正信号确实出现在 low-margin ambiguity 上。
2. 但即便在这个 subset 上，retrieval 也还没有超过固定 `v2`。

#### 4.2 top confusion pair subset

dominant confusion pair：`NatBeauty -> Harmony`，test subset = `96`

这组 subset 的 Hard 全部为 `0.0`，说明这个 pair 本身几乎完全落在 hardest tail 之外，且 retrieval 没有形成清晰优势：

- `SLR-C` Macro：`9.79`
- `v2` Macro：`9.77`
- `support - confusion refute best` Macro：`9.16`

因此，本轮没有得到“retrieval 明显更擅长 dominant confusion pair”的证据。

#### 4.3 top-2 / pairwise ranking

`top-2 disambiguation accuracy`：

- `SLR-C`：`80.32`
- `v2`：`80.18`
- `support only best`：`81.11`
- `support - global refute best`：`81.11`
- `support - confusion refute best`：**`81.37`**

`pairwise ranking accuracy`：

- `SLR-C`：`85.75`
- `v2`：**`86.27`**
- `support only best`：`86.01`
- `support - global refute best`：`86.00`
- `support - confusion refute best`：`85.99`

这组诊断说明：

1. retrieval 的确在 top-2 局部消歧上有信号
2. 但 v2 在更整体的 pairwise ranking 上依然最稳
3. 这正符合 design 的判断：retrieval 更像一个有潜力的 ambiguity module，但当前 image-only MVP 还不够强

### 5. 为什么停在 Phase 2，不进入 Phase 3

本轮符合 design 的停止条件：

1. image-only retrieval 的 validation-best main result 低于当前 fixed `v2 comparative + gate`
2. confusion-aware retrieval 虽然比 global-negative retrieval 略好，但幅度不足以支撑继续增加 evidence-aware similarity 的复杂度
3. dominant confusion pair subset 没有形成明显正证据

因此，这一轮最保守、最合理的结论是：

> retrieval 路线可以保留，但还不应该直接进入 `Phase 3 evidence-aware retrieval`。  
> 更合理的下一步应该是先确认：当前 retrieval residual 的使用方式、`r/beta` 稳定性、以及更强的 candidate-local selection 是否能在 validation 上稳定超过 `SLR-C` 或接近 `v2`。

### 6. Commands

代码检查：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m py_compile \
  scripts/analyze_retrieval_ambiguity_agent.py \
  src/utils/retrieval_ambiguity.py
```

单测：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m pytest -q \
  tests/test_retrieval_ambiguity.py
```

结果：`3 passed`

Smoke run：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_retrieval_ambiguity_agent.py \
  --max-samples 64 \
  --r-list 3 \
  --beta-list 0.01 \
  --normalize-candidate-evidence \
  --output-dir /tmp/retrieval_ambiguity_smoke
```

Full run：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_retrieval_ambiguity_agent.py \
  --device cpu \
  --normalize-candidate-evidence \
  --output-dir logs/analysis/retrieval_ambiguity_agent_20260312
```

### 7. Files changed

- `src/utils/retrieval_ambiguity.py`
- `scripts/analyze_retrieval_ambiguity_agent.py`
- `tests/test_retrieval_ambiguity.py`
- `docs/record/record_0312_06_Retrieval.md`

### 8. Verification

代码验证：

- `py_compile` 通过
- `tests/test_retrieval_ambiguity.py` 通过：`3 passed`
- smoke run 成功完成并产出主表 / summary

结果验证：

- `SLR-C` 复现为：
  - Macro `51.28`
  - Micro `59.13`
  - Samples `58.47`
  - mAP `53.66`
  - Hard `33.98`
- fixed `v2 reference` 复现为：
  - Macro `51.73`
  - Micro `59.22`
  - Samples `58.19`
  - mAP `53.78`
  - Hard `35.83`

主要产物：

- `logs/analysis/retrieval_ambiguity_agent_20260312/main_comparison.csv`
- `logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_search_results.csv`
- `logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_setting_ablation.csv`
- `logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_r_ablation.csv`
- `logs/analysis/retrieval_ambiguity_agent_20260312/retrieval_class_stats.csv`
- `logs/analysis/retrieval_ambiguity_agent_20260312/summary.json`
