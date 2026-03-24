# 实验记录：Confusion-Aware Multi-Agent Ambiguity Resolution v3 MVP

## 0. 文档信息

- 日期：`2026-03-12`
- 对应设计：`docs/design/design_0312_04_ConfusionAwareMultiAgentAmbiguityResolution_v3.md`
- 基础 checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- 主实验输出目录：`logs/analysis/phase1_agent_evidence_verification_v3_20260312`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. 本轮实现内容

本轮按 design 只落一个保守 MVP，不改 `SLR-C` base 本体。

新增代码：

- `src/utils/evidence_verification.py`
  - `build_confusion_aware_router`
  - `compute_specialist_pairwise_evidence`
  - `resolve_routed_specialist_evidence`
- `scripts/analyze_agent_evidence_verification_v3.py`
  - 固定 `scenario SLR-C`
  - 固定 v2 reference
  - 跑 v3 `all-specialists` / `routed-specialists`
  - 同时导出 trigger ablation、router stats、hard subset 与 pairwise/top-2 diagnostics
- `tests/test_evidence_verification_v3.py`
  - router 触发
  - routed specialist 选择
  - neighborhood-only resolver 更新

MVP 设计对应关系：

1. Proposal Agent：
   - 直接复用现有 `scenario SLR-C`
2. Rule-based Router：
   - 至少用 `top1-top2 margin + confusion hit`
3. Specialist Agents：
   - 继续复用当前 comparative evidence backend
   - 但改成按 specialist 输出 pairwise evidence
4. Resolver：
   - 只在 router 选中的 neighborhood 内聚合并回写 residual
5. Comparison：
   - `all-specialists`
   - `routed-specialists`

---

## 2. 环境校验与命令

强制使用的 Python：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python --version
```

语法与单测：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m py_compile \
  src/utils/evidence_verification.py \
  scripts/analyze_agent_evidence_verification_v3.py \
  tests/test_evidence_verification_v3.py

/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m pytest -q \
  tests/test_evidence_verification_v2.py \
  tests/test_evidence_verification_v3.py
```

结果：`6 passed`

Smoke run：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_agent_evidence_verification_v3.py \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --dispatch-modes all,routed \
  --trigger-modes margin_confusion \
  --margin-tau-list 1.0 \
  --v3-beta-list 0.01 \
  --output-dir logs/analysis/smoke_agent_evidence_verification_v3_20260312
```

Phase 1 第一轮：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_agent_evidence_verification_v3.py \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --output-dir logs/analysis/phase1_agent_evidence_verification_v3_20260312
```

---

## 3. 本轮设置

### 3.1 Base system

- base：`scenario SLR-C`
- `top-k = 10`
- `alpha = 0.3`
- 保持 class-wise threshold

### 3.2 Evidence backend

- relation：`hard_negative_diff`
- sparse profile：`top-20`
- pair profile：`top-5`
- activation top-m：`5`
- confusion neighborhood：`top-3` from `SLR` train confusion, inspect `top-10`

### 3.3 v2 reference

- comparative verification v2
- gate：`exp(gamma=2.0)`
- fusion：`add`
- `beta = 0.01`

### 3.4 v3 search

- dispatch：
  - `all`
  - `routed`
- trigger：
  - `margin_confusion`
  - `always`
  - `margin_only`
  - `confusion_only`
- `margin_tau = {0.5, 1.0}`
- `beta = {0.005, 0.01, 0.02}`
- routed specialist policy：
  - 对 router 选中的 pair/group
  - 按 pairwise profile mass 选 top-2 specialists
- resolver：
  - 对 selected specialists 做 weighted sum
  - 再按 pair count 做 mean aggregation
  - 只更新 selected neighborhood

---

## 4. 主结果

`phase1_comparison.csv` 主表如下：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR-C` | 51.28 | 59.13 | 58.47 | 53.66 | 33.98 |
| v2 best reference | 51.73 | 59.22 | 58.19 | 53.78 | **35.83** |
| v3 all-specialists | 50.85 | 59.01 | 58.04 | 53.70 | 33.93 |
| v3 routed-specialists | **51.97** | 59.17 | **58.55** | 53.36 | 34.90 |

### 4.1 直接结论

1. `routed-specialists` 明显优于 `all-specialists`
   - macro：`51.97 > 50.85`
   - samples：`58.55 > 58.04`
   - hard：`34.90 > 33.93`

2. `v3 routed` 已经超过 `scenario SLR-C`
   - macro：`+0.69`
   - micro：`+0.04`
   - samples：`+0.08`
   - hard：`+0.92`

3. 但本轮 `v3 routed` 还没有超过 `v2 best reference`
   - macro：`+0.24`
   - samples：`+0.36`
   - hard：`-0.94`
   - mAP：`-0.42`

因此本轮最稳妥判断是：

> v3 的 **specialist routing + local resolver** 是成立的，  
> 但当前 router 的 confusion 触发过宽，尚未把“条件触发”的效率优势真正做出来，  
> 所以整体上仍未超过 v2 best。

---

## 5. Router / Specialist 诊断

### 5.1 best v3 all-specialists

- config：`margin_confusion + beta=0.01 + tau=0.5`
- trigger rate：`99.92%`
- avg specialists called：`4.0`
- avg neighborhood size：`7.67`

### 5.2 best v3 routed-specialists

- config：`margin_confusion + beta=0.02 + tau=0.5`
- trigger rate：`99.92%`
- avg specialists called：`2.0`
- avg neighborhood size：`7.67`
- specialist call histogram：
  - `scene`: `1215`
  - `activity`: `1184`
  - `object`: `16`
  - `style`: `15`

### 5.3 最重要的问题

当前 `confusion hit` 几乎覆盖全部 test 样本：

- `confusion_trigger_rate = 99.92%`
- `margin_trigger_rate = 43.17%`

这导致：

- `margin_confusion`
- `confusion_only`
- `always`

三者在本轮几乎等价。

换句话说，本轮已经验证了：

> **selective specialist dispatch 本身有价值**

但还没有验证到：

> **真正稀疏、节省计算的 conditional triggering**

下一轮如果继续 v3，第一优先级不是再调 beta，而是要把 confusion-hit 定义收紧，否则 router 名义上存在，实际上几乎总在全触发。

---

## 6. Hard subset / pairwise diagnostics

### 6.1 low-margin subset

`margin < 1.0` 的 test 子集上：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SLR-C` | 45.47 | 56.18 | 52.43 | 48.73 | 35.07 |
| v2 reference | 46.03 | 55.64 | 50.88 | 49.07 | **37.07** |
| v3 routed | **46.46** | 55.59 | 51.69 | **49.58** | 36.64 |

结论：

- v3 routed 在 low-margin subset 上相对 `SLR-C` 的 macro / mAP / hard 仍然有增益
- 但 `hard` 依旧低于 v2 reference

### 6.2 routed ambiguity cases 上的局部判别

在 routed-triggered cases 上：

- top-2 disambiguation accuracy
  - `SLR-C`: `80.29`
  - `v2`: `80.16`
  - `v3 all`: `80.69`
  - `v3 routed`: **`80.95`**

- pairwise ranking accuracy
  - `SLR-C`: `85.76`
  - `v2`: `86.08`
  - `v3 all`: `86.15`
  - `v3 routed`: **`86.77`**

这说明：

> v3 routed 的 specialist-based local adjudication 确实更贴近“候选局部消歧”这个目标。

---

## 7. 本轮结论

### 7.1 可以保留的结论

1. `SLR-C` base 不需要改，本轮 add-on 路线可直接叠在上面。
2. specialist-agent style interface 是成立的。
3. routed specialist dispatch 比 all-specialists 明显更合理。
4. neighborhood-only resolver 没有把系统打崩，并在 `macro / samples / pairwise accuracy` 上有正收益。

### 7.2 当前的主要问题

1. confusion router 太宽，导致触发率接近 `100%`
2. 因而 v3 还没有真正体现“条件触发节省计算”的实验价值
3. 在当前定义下，v3 routed 仍未超过 v2 best 的 `Hard`

### 7.3 下一步建议

若继续按 design 推进，最优先做：

1. 收紧 confusion hit 定义
   - 只看 top-2 / top-3
   - 或只保留高频 directed confusion pairs
   - 或要求双向 hit / stronger overlap
2. 再重跑 `margin_only / confusion_only / margin+confusion`
   - 这次才能真正看清 router 的意义
3. 在 router 收紧后，再比较 runtime / avg specialists called

---

## 8. 产物

主产物：

- `logs/analysis/phase1_agent_evidence_verification_v3_20260312/summary.json`
- `logs/analysis/phase1_agent_evidence_verification_v3_20260312/phase1_comparison.csv`
- `logs/analysis/phase1_agent_evidence_verification_v3_20260312/v3_search_results.csv`

Smoke 产物：

- `logs/analysis/smoke_agent_evidence_verification_v3_20260312/summary.json`

一句话总结：

> v3 MVP 已经证明“routed specialists + neighborhood-only resolver”有方法价值，但当前 confusion router 太宽，导致 conditional triggering 还没有真正成立；在这一轮里，v3 routed 超过了 `SLR-C`，但仍未超过 v2 best。 
