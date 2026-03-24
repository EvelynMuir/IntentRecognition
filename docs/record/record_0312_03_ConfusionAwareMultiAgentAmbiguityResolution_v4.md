# 实验记录：Confusion-Aware Multi-Agent Ambiguity Resolution v4 Router Tightening

## 0. 文档信息

- 日期：`2026-03-12`
- 对应设计：`docs/design/design_0312_03_ConfusionAwareMultiAgentAmbiguityResolution_v4_TaskList.md`
- 基础 checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- 主实验输出目录：`logs/analysis/v4_router_tightening_round1_20260312`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. 本轮目标与改动边界

v4 这一轮只做 router study，不扩后端方法栈。唯一主目标是把 v3 里几乎全触发的宽 router，收紧成真正稀疏、可解释、面向 hard ambiguity 的 conditional router。

本轮固定不动的部分：

- base system：`scenario SLR-C`
- specialist backend：沿用 v3 routed specialists
- evidence backend：沿用 v3 comparative evidence pipeline
- relation：`hard_negative_diff`
- resolver：沿用 v3 当前 `neighborhood-only` resolver

本轮只动：

- `trigger_mode`
- `margin_tau`
- `top2 / top3` confusion scope

兼容性小改动：

- 在 `build_confusion_aware_router` 中新增 top-2 / top-3 trigger modes 和 router stats
- 但保留 broad `margin_confusion` 的原始 v3 行为：broad hit 仍对整个 selected neighborhood 做 pair expansion，而不是只保留 hit pairs

---

## 2. 环境校验与命令

固定 Python：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python --version
```

语法与单测：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m py_compile \
  src/utils/evidence_verification.py \
  scripts/analyze_agent_evidence_verification_v3.py \
  tests/test_evidence_verification_v2.py \
  tests/test_evidence_verification_v3.py

/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m pytest -q \
  tests/test_evidence_verification_v2.py \
  tests/test_evidence_verification_v3.py
```

结果：`9 passed`

本轮 v4 主命令：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_agent_evidence_verification_v3.py \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --dispatch-modes routed \
  --trigger-modes always,margin_only,margin_confusion,confusion_top2_only,margin_and_confusion_top2,margin_or_confusion_top2,confusion_top3_only,margin_and_confusion_top3,margin_or_confusion_top3 \
  --margin-tau-list 0.3,0.5,0.7,1.0 \
  --v3-beta-list 0.02 \
  --output-dir logs/analysis/v4_router_tightening_round1_20260312
```

---

## 3. 本轮设置

### 3.1 Base system

- base：`scenario SLR-C`
- `top-k = 10`
- `alpha = 0.3`
- 继续使用 class-wise threshold

### 3.2 Frozen backend

- relation：`hard_negative_diff`
- profile：`top-20`
- pair profile：`top-5`
- activation top-m：`5`
- contradiction lambda：`0.0`
- confusion neighborhoods：由 `SLR train logits` 构建，`confusion_topk=10`，每类保留 `top-3` neighbors

### 3.3 v2 reference

- gate：`exp(gamma=2.0)`
- fusion：`add`
- `beta = 0.01`

### 3.4 v4 router search

- dispatch：`routed`
- beta：固定 `0.02`
- `margin_tau = {0.3, 0.5, 0.7, 1.0}`
- trigger modes：
  - baseline：`always`
  - baseline：`margin_only`
  - wide reference：`margin_confusion`
  - priority 1：`confusion_top2_only`
  - priority 1：`margin_and_confusion_top2`
  - priority 1：`margin_or_confusion_top2`
  - priority 2：`confusion_top3_only`
  - priority 2：`margin_and_confusion_top3`
  - priority 2：`margin_or_confusion_top3`
- 本轮未启用 additional confusion pair filtering

---

## 4. 主结果

下表汇总每个 trigger config 的 best row。`tau` 表示在 `{0.3, 0.5, 0.7, 1.0}` 中按 val macro 选出的阈值；`beta` 固定为 `0.02`。

| Config | Macro | Micro | Samples | mAP | Hard | trigger_rate | margin_trigger_rate | confusion_trigger_rate | avg_specialists_called | avg_neighborhood_size | avg_pairs_resolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always (tau=0.3)` | **51.97** | 59.17 | 58.55 | 53.36 | **34.90** | 100.00% | 28.29% | 99.92% | 2.000 | 7.660 | 27.571 |
| `margin_only (tau=1.0)` | 51.76 | 59.11 | 58.42 | 53.47 | 34.86 | 65.95% | 65.95% | 99.92% | 1.319 | 5.041 | 18.087 |
| `margin_confusion (tau=0.3)` | **51.97** | 59.17 | 58.55 | 53.36 | **34.90** | 99.92% | 28.29% | 99.92% | 1.998 | 7.659 | 27.570 |
| `confusion_top2_only (tau=0.3)` | 50.51 | 59.00 | 58.91 | 53.41 | 31.30 | 65.13% | 28.29% | 65.13% | 1.303 | 1.303 | 0.651 |
| `margin_and_confusion_top2 (tau=0.3)` | 50.76 | 58.88 | 58.21 | 53.35 | 32.78 | 18.67% | 28.29% | 65.13% | 0.373 | 0.373 | 0.187 |
| `margin_or_confusion_top2 (tau=1.0)` | 51.51 | **59.26** | **59.28** | 52.92 | 34.07 | 88.57% | 65.95% | 65.13% | 1.760 | 1.771 | 0.886 |
| `confusion_top3_only (tau=0.3)` | 49.31 | 57.34 | 57.95 | 53.30 | 32.78 | 88.90% | 28.29% | 88.90% | 1.778 | 2.377 | 1.787 |
| `margin_and_confusion_top3 (tau=1.0)` | 50.48 | 58.56 | 58.40 | **53.61** | 32.90 | 58.39% | 65.95% | 88.90% | 1.168 | 1.553 | 1.162 |
| `margin_or_confusion_top3 (tau=0.7)` | 49.55 | 57.58 | 58.26 | 53.33 | 32.68 | 95.39% | 53.70% | 88.90% | 1.908 | 2.507 | 1.852 |

### 4.1 直接结论

1. wide reference 没有问题，`margin_confusion` 仍与原 v3 routed 对齐：
   - Macro：`51.97`
   - Hard：`34.90`
   - `confusion_trigger_rate = 99.92%`

2. best top-2 keep-going candidate 是 `margin_or_confusion_top2 (tau=1.0)`：
   - 相比 `SLR-C`，Macro：`+0.23`，Hard：`+0.10`
   - 但仍低于 wide reference：Macro `-0.46`，Hard `-0.82`

3. 真正最稀疏的 conditional router 是 `margin_and_confusion_top2 (tau=0.3)`：
   - `trigger_rate = 18.67%`
   - 但 Macro / Hard 明显掉到 `50.76 / 32.78`

4. top-3 variants 整体不如 top-2，且没有比 `SLR-C` 更稳的 Hard 收益。

---

## 5. Router sparsity / efficiency findings

### 5.1 宽 router reference

`margin_confusion` 仍几乎全触发：

- `trigger_rate = 99.92%`
- `avg_specialists_called = 1.998`
- `avg_neighborhood_size = 7.659`
- `avg_pairs_resolved = 27.570`

这确认了 v4 的原始判断：v3 的问题不是 routed specialists 无效，而是 confusion router 太宽。

### 5.2 Top-2 tightening

`top2 confusion` 本身已经明显更干净：

- `confusion_trigger_rate = 65.13%`
- 相比 broad `99.92%`，直接下降 `34.79` 个点

其中两个最重要的版本：

- `margin_and_confusion_top2`
  - `trigger_rate = 18.67%`
  - 相比 wide reference，`avg_specialists_called` 下降 `81.3%`
  - `avg_neighborhood_size` 下降 `95.1%`
  - `avg_pairs_resolved` 下降 `99.3%`
  - 但 Hard 只有 `32.78`

- `margin_or_confusion_top2`
  - `trigger_rate = 88.57%`
  - 相比 wide reference，`avg_neighborhood_size` 下降 `76.9%`
  - `avg_pairs_resolved` 下降 `96.8%`
  - 仍保持对 `SLR-C` 的小幅整体优势：Macro `51.51 > 51.28`，Hard `34.07 > 33.98`

换句话说，top-2 router 已经把“局部 pair 级工作”做得很稀疏，但还没有同时把 trigger rate 压得足够低而不掉 Hard。

### 5.3 Top-3 tightening

`top3 confusion` 明显更宽：

- `confusion_trigger_rate = 88.90%`
- 最稀疏的 `margin_and_confusion_top3` 也只有 `58.39%` trigger
- best top-3 Hard 只有 `32.90`

top-3 scope 带来了更多触发，但没有换来更好的 overall Hard，说明这一轮 top-3 不值得优先保留。

### 5.4 最重要发现

- `top2 confusion` 比 `top3 confusion` 更符合“局部歧义”定义，至少在 sparsity 上是明显更干净的。
- `margin_and_confusion_top2` 证明了 router 可以真正变成 conditional module，但当前边界太严，收益被性能损失抵消。
- `margin_or_confusion_top2` 是 round-1 最值得继续推进的折中点，因为它还保留了对 `SLR-C` 的整体优势，同时把 neighborhood / pair 工作量压得很低。
- 如果后续还要继续收紧，下一步优先级不是改后端，而是做 confusion pair filtering / directed thresholding，争取把 top-2 的 trigger rate 再往下压，而不是回到 top-3。

---

## 6. Hard subset findings

### 6.1 `low-margin` subset (`n = 802`)

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 45.47 | 48.73 | 35.07 |
| `v2 best` | 46.03 | 49.07 | **37.07** |
| `margin_confusion` | **46.46** | **49.58** | 36.64 |
| `margin_or_confusion_top2` | 45.72 | 48.54 | 35.50 |
| `margin_and_confusion_top3` | 45.43 | 49.21 | 34.35 |

结论：tight router 里只有 `margin_or_confusion_top2` 还能在 low-margin subset 上勉强守住对 `SLR-C` 的 Hard 优势，但 wide reference 仍更强。

### 6.2 `top2-confusion` subset (`n = 792`)

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 52.04 | 55.47 | 35.12 |
| `v2 best` | 52.72 | **55.71** | **36.81** |
| `margin_confusion` | **52.98** | 55.50 | 35.71 |
| `margin_or_confusion_top2` | 52.46 | 54.82 | 34.96 |
| `margin_and_confusion_top3` | 51.13 | 54.86 | 32.42 |

结论：在真正的 top-2 confusion subset 上，top-2 tightened router 还没有超过 wide reference，也没有超过 `v2 best`。

### 6.3 `top3-confusion` subset (`n = 1081`)

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 51.37 | **55.11** | 33.77 |
| `v2 best` | 51.71 | 55.30 | **35.13** |
| `margin_confusion` | **52.00** | 54.52 | 34.39 |
| `margin_or_confusion_top2` | 51.48 | 54.08 | 33.20 |
| `margin_and_confusion_top3` | 50.55 | 54.70 | 32.48 |

结论：top-3 scope 不但更宽，而且在自己的目标子集上也没有体现出更强的 Hard 收益。

### 6.4 Local disambiguation diagnostics

| Router config | Triggered top-2 acc | Pairwise ranking acc | Triggered samples / pairs |
| --- | ---: | ---: | ---: |
| `margin_confusion` | 80.95 | **86.77** | `756 / 10889` |
| `margin_or_confusion_top2` | 77.00 | 77.00 | `639 / 639` |
| `margin_and_confusion_top2` | 53.51 | 53.51 | `114 / 114` |
| `confusion_top3_only` | **81.10** | 81.21 | `672 / 1235` |
| `margin_and_confusion_top3` | 69.33 | 74.47 | `375 / 760` |

结论：

- local adjudication 仍然是有信息量的，但 tight router 选中的子集更小、更偏难，收益没有直接转化成 overall Hard 提升。
- `confusion_top3_only` 的局部判别不差，但它整体指标明显下滑，说明“局部对了”不等于“全局值得保留”。

---

## 7. 结论

对照 v4 success criteria，这一轮可以做出比较稳妥的判断：

- Success Criterion A：成立。`margin_and_confusion_top2` 把 trigger rate 从 `99.92%` 压到 `18.67%`。
- Success Criterion B：部分成立。`margin_or_confusion_top2` 还能守住对 `SLR-C` 的轻微 overall 优势，但最稀疏的 `AND` 版本会明显掉 Hard。
- Success Criterion C：部分成立。tight router 里只有 `margin_or_confusion_top2` 仍稳住了 `SLR-C` 之上。
- Success Criterion D：不充分成立。tight router 在 hard subsets 上还没有比 wide reference 更强。

本轮最稳妥总结是：

> v4 已经证明 `top2 confusion` 是正确的 tightening 方向，  
> 也证明 router 可以真正变成稀疏 conditional module；  
> 但当前最稀疏的边界还太严，round-1 最好的折中点仍是 `margin_or_confusion_top2`，而不是 `AND` 版本。

下一步建议：

- 先把 `margin_or_confusion_top2` 固定为 v4 主线候选
- 再只在 `top2 confusion` 分支上做 high-frequency pair filtering / directed thresholding
- 暂时不继续扩 top-3，也不改后端 resolver / specialists / evidence stack
