# 实验记录：Prototype Evidence Memory on v2 Comparative Verifier

## 0. 文档信息

- 日期：`2026-03-12`
- 对应设计：`docs/design/design_0312_04.md`
- 基础 checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- 主实验输出目录：`logs/analysis/prototype_evidence_memory_20260312`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. 本轮实现

本轮严格按 design 的“最小改动”原则做：

- `scenario SLR-C` 保持不变
- v2 comparative / gate / fusion 保持不变
- 只在 verifier profile 一层增加 prototype memory

新增代码：

- `src/utils/evidence_verification.py`
  - `build_evidence_vectors`
  - `learn_prototype_memory_relations`
  - `select_prototype_profile_ids`
  - `compute_pairwise_comparative_scores` 新增 `candidate_profile_ids`
- `scripts/analyze_agent_evidence_verification_prototype_memory.py`
  - 固定 `SLR-C`
  - 固定 v2 strongest reference
  - 跑 `K=2 / K=3` prototype memory
  - 导出主表、prototype class stats、prototype usage、summary
- `tests/test_prototype_evidence_memory.py`
  - prototype 选择
  - 小类 fallback
  - prototype-conditioned pairwise scoring

---

## 2. 运行命令

语法检查：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m py_compile \
  src/utils/evidence_verification.py \
  scripts/analyze_agent_evidence_verification_prototype_memory.py \
  tests/test_prototype_evidence_memory.py
```

单测：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m pytest -q \
  tests/test_evidence_verification_v2.py \
  tests/test_prototype_evidence_memory.py
```

结果：`6 passed`

主实验：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_agent_evidence_verification_prototype_memory.py \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --output-dir logs/analysis/prototype_evidence_memory_20260312
```

额外 sanity check：

- 手动检查 `prototype_k=1` 经 prototype path 计算得到的 residual 与 v2 reference 基本一致
- 最大绝对误差：
  - val：`5.87e-4`
  - test：`6.26e-4`

这说明本轮负结果不是“prototype 代码路径与 v2 主线不一致”导致的。

---

## 3. 固定设置

### 3.1 Base proposal

- base：`scenario SLR-C`
- `topk = 10`
- `alpha = 0.3`

### 3.2 v2 reference

- experts：`object + scene + style + activity`
- relation：`hard_negative_diff`
- sparse profile：`top-20`
- pair profile：`top-5`
- activation top-m：`5`
- gate：`exp(gamma=2.0)`
- fusion：`add`
- `beta = 0.01`

### 3.3 Prototype memory

- prototype selection：`best prototype`
- prototype source：`focused evidence vector`
- source top-m：`5`
- `K = 2 / 3`
- fallback：
  - `min_positive_samples = 64`
  - `min_cluster_size = 16`
- random seed：`0`

---

## 4. 主结果

`logs/analysis/prototype_evidence_memory_20260312/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scenario SLR-C` | 51.28 | 59.13 | **58.47** | 53.66 | 33.98 |
| v2 comparative + gate | **51.73** | **59.22** | 58.19 | **53.78** | **35.83** |
| v2 + prototype memory (`K=2`) | 48.12 | 55.78 | 56.36 | 51.59 | 27.99 |
| v2 + prototype memory (`K=3`) | 48.52 | 55.66 | 55.88 | 50.30 | 31.22 |

直接结论：

1. 在当前 strongest v2 setting 上，prototype memory 没有带来增益。
2. `K=3` 明显好于 `K=2`，但仍显著低于 v2 reference。
3. 本轮应把 `single profile > best prototype (K=2/3)` 视为负结果。

---

## 5. 诊断

### 5.1 Fallback / cluster 统计

- `K=2`
  - `num_profiles = 56`
  - `num_multi_prototype_classes = 28`
  - `num_fallback_classes = 0`
  - `min_cluster_size = 55`
- `K=3`
  - `num_profiles = 84`
  - `num_multi_prototype_classes = 28`
  - `num_fallback_classes = 0`
  - `min_cluster_size = 39`

说明：

- 这轮不是“小类样本太少导致大量 fallback”
- 数据量足够支撑 `K=2/3` 聚类
- 问题更像是 prototype split 本身削弱了当前 verifier profile 的稳定性

### 5.2 Low-margin subset

`margin < 1.0` 子集：

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 45.47 | 48.73 | 35.07 |
| v2 reference | **46.03** | **49.07** | **37.07** |
| prototype `K=2` | 41.40 | 46.57 | 28.83 |
| prototype `K=3` | 42.11 | 44.63 | 32.19 |

结论：

- prototype memory 在真正困难子集上也没有形成收益
- `K=3` 仍比 `K=2` 更稳，但离 v2 reference 仍有明显差距

### 5.3 Pairwise / top-2 诊断

全 test：

- top-2 disambiguation accuracy
  - `SLR-C`：`80.32`
  - v2 reference：`80.18`
  - prototype `K=2`：`80.71`
  - prototype `K=3`：`79.52`
- pairwise ranking accuracy
  - `SLR-C`：`85.75`
  - v2 reference：`86.27`
  - prototype `K=2`：`85.23`
  - prototype `K=3`：`85.24`

low-margin 子集：

- top-2 disambiguation accuracy
  - `SLR-C`：`68.01`
  - v2 reference：`67.77`
  - prototype `K=2`：`68.72`
  - prototype `K=3`：`66.59`
- pairwise ranking accuracy
  - `SLR-C`：`83.24`
  - v2 reference：`84.01`
  - prototype `K=2`：`82.45`
  - prototype `K=3`：`82.43`

解释：

- `K=2` 在 top-2 局部选择上有轻微提升
- 但 pairwise ranking 明显下降
- 最终全局 macro / hard 也随之下降

这更像是：

> prototype 选择提高了少量 top-2 mode matching，  
> 但 prototype-specific sparse relation profile 不够稳，导致整体 comparative ranking 变差。

---

## 6. 结论

本轮结论比较明确：

1. 在当前 strongest verifier path 上，prototype evidence memory 不成立为主线增益。
2. `best prototype (K=2/3)` 都没有超过 `single profile`。
3. `K=3` 比 `K=2` 稳一些，但仍低于 v2 reference 和 `SLR-C`。
4. 当前 prototype split 方向更可能破坏了 class-level discriminative profile 的统计稳定性，而不是提升类内多样性建模。

因此，对 `design_0312_04` 的当前判断是：

> 在现有 focused evidence + sparse pairwise profile + fixed gate/fusion 的框架下，  
> prototype memory 这条 MVP 方向不值得继续作为主方法推进。

如果后续还想继续，只建议作为低优先级讨论分支，且应先改变下面至少一项再试：

- prototype source（如 `full` vs `focused`）
- prototype relation 学习方式
- 更保守的 conditional use，而不是全量替换 single profile

---

## 7. 产物

- `logs/analysis/prototype_evidence_memory_20260312/summary.json`
- `logs/analysis/prototype_evidence_memory_20260312/main_comparison.csv`
- `logs/analysis/prototype_evidence_memory_20260312/prototype_class_stats_k2.csv`
- `logs/analysis/prototype_evidence_memory_20260312/prototype_class_stats_k3.csv`
- `logs/analysis/prototype_evidence_memory_20260312/prototype_usage_test_k2.csv`
- `logs/analysis/prototype_evidence_memory_20260312/prototype_usage_test_k3.csv`
