# 实验记录：Region-Grounded Agent

## (a) main conclusions

1. 本轮按 design 优先完成了 `Phase 1: region-conditioned belief update MVP`，并在 Phase 1 路径稳定后补了一个最轻量的 `Phase 2 soft routing` 对比；没有继续做 graph reasoning。
2. 在当前最保守、最可解释的实现下，`Phase 1 hybrid` 没有超过固定基线 `scenario SLR-C`，也明显没有接近当前 strongest `v2 comparative + gate`。
3. `Phase 1 hybrid` 的最好点来自：
   - `object/activity = region-conditioned`
   - `scene/style = global`
   - `attn_scale = 20`
   - `local_weight = 0.5`
   - `global_weight = 1.0`
   - `beta = 0.05`
   - `gate_gamma = 0`
4. 这个 best `Phase 1 hybrid` 的 test 指标为：
   - Macro `50.65`
   - Micro `58.15`
   - Samples `57.35`
   - mAP `53.43`
   - Hard `34.53`
5. `Phase 1 hybrid` 相比 `SLR-C` 的唯一明确正信号在 `Hard`：
   - `34.53 > 33.98`
   - 以及 `top-2 disambiguation accuracy = 81.11 > 80.32`
   但它在 `Macro / Micro / Samples / mAP` 全部下降，因此还不能说明当前 region-grounded updater 已经成立为主线方法。
6. 区域化本身也没有赢过更便宜的 `all-global` 版本：
   - `Phase 1 all-global`：Macro `50.77`, Hard `34.25`
   - `Phase 1 hybrid`：Macro `50.65`, Hard `34.53`
   也就是 region grounding 在 hardest ambiguity 上有一点增益，但整体平均指标没有超过 all-global。
7. `Phase 2 soft routing` 只带来了很小的形态变化，没有形成强 routing 证据：
   - best Phase 2：Macro `50.80`, Hard `34.13`
   - routing 平均权重接近均匀分配：`object 0.244 / activity 0.260 / scene 0.242 / style 0.254`
   - mean entropy `1.19`，距离 `ln(4)=1.386` 不远
   这符合 design 里“controller 学成平均分配”的风险提示。
8. 因为 `Phase 1` 没有接近 `v2 comparative + gate`，而 `Phase 2` 的 routing 也没有形成明显专家选择，所以本轮没有进入 graph reasoning，符合 design 的停止条件。

## (b) main result table

主表来自：`logs/analysis/phase1_region_grounded_agent_20260312/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR-C` | 51.28 | 59.13 | **58.47** | 53.66 | 33.98 |
| `v2 comparative + gate` | **51.73** | **59.22** | 58.19 | **53.78** | **35.83** |
| `Phase 1: scene/style global only` | 50.31 | 57.82 | 56.99 | 53.41 | 34.00 |
| `Phase 1: all-global evidence` | 50.77 | 58.86 | 58.20 | 53.70 | 34.25 |
| `Phase 1: region-conditioned updater` | 50.65 | 58.15 | 57.35 | 53.43 | 34.53 |
| `Phase 2: soft routing` | 50.80 | 58.83 | 58.04 | 53.34 | 34.13 |

## (c) detailed results and analysis

### 1. 实现内容

本轮严格按 design 的优先顺序推进：

1. 固定 proposer 为 `scenario SLR-C`
2. 复用当前 strongest `v2` cache stack
3. 先做 `Phase 1 region-conditioned belief update`
4. 在 Phase 1 路径稳定后，补一个最轻量的 `Phase 2 soft routing`
5. 不做 graph reasoning

本轮新增代码：

- `src/utils/region_grounded_reasoning.py`
  - `compute_candidate_region_summaries`
  - `compute_candidate_class_evidence_scores`
  - `normalize_topk_candidate_matrix`
  - `compute_soft_routing_weights`
  - `apply_soft_routing`
- `scripts/analyze_region_grounded_agent.py`
  - 固定 `scenario SLR-C`
  - 固定 `v2` strongest reference
  - 新增独立 `val/test` patch-token cache
  - 跑 `Phase 1` 的 `scene/style only` / `all-global` / `region+global hybrid`
  - 跑最轻量 `Phase 2 soft routing`
- `tests/test_region_grounded_reasoning.py`
  - region attention
  - top-k candidate scatter
  - top-k normalization
  - soft routing

### 2. 方法设置

#### 2.1 固定不动的部分

- checkpoint：`logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`
- proposer：`scenario SLR-C`
- `topk = 10`
- `alpha = 0.3`
- CLIP backbone：`ViT-L/14`
- base cache：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

#### 2.2 固定 evidence backend

- experts：`object + activity + scene + style`
- relation：`hard_negative_diff`
- `profile_topn = 20`
- `activation_topm = 5`
- `hard_negative_topn = 3`

也就是说，本轮没有改 proposer，也没有改 v2 的 train-side relation 学习逻辑，只是把 inference 侧的 object/activity 证据换成了 candidate-conditioned region summary。

#### 2.3 Region path

- patch source：OpenAI CLIP projected patch tokens
- region cache：只缓存 `val/test`
- object/activity：candidate-conditioned local evidence
- scene/style：global evidence
- `region_attn_scale ∈ {5, 10, 20}`

#### 2.4 Phase 1 搜索空间

- `local_weight ∈ {0.5, 1.0, 1.5}`
- `global_weight ∈ {0.25, 0.5, 1.0}`
- `beta ∈ {0.005, 0.01, 0.02, 0.05}`
- `gate_gamma ∈ {0, 2}`

#### 2.5 Phase 2 搜索空间

- 在 `Phase 1 best hybrid` 上固定 `attn_scale = 20`
- `temperature ∈ {0.5, 1.0, 2.0}`
- `beta ∈ {0.005, 0.01, 0.02, 0.05}`
- `gate_gamma ∈ {0, 2}`

### 3. 关键对比

#### 3.1 与固定 baselines 的直接比较

相对 `SLR-C`：

- `Phase 1 all-global`
  - Macro `-0.51`
  - Micro `-0.27`
  - Samples `-0.27`
  - mAP `+0.04`
  - Hard `+0.27`
- `Phase 1 hybrid`
  - Macro `-0.63`
  - Micro `-0.98`
  - Samples `-1.12`
  - mAP `-0.23`
  - Hard `+0.55`
- `Phase 2 soft routing`
  - Macro `-0.48`
  - Micro `-0.30`
  - Samples `-0.43`
  - mAP `-0.32`
  - Hard `+0.15`

相对 `v2 comparative + gate`：

- `Phase 1 all-global`
  - Macro `-0.96`
  - Hard `-1.59`
- `Phase 1 hybrid`
  - Macro `-1.09`
  - Hard `-1.31`
- `Phase 2 soft routing`
  - Macro `-0.93`
  - Hard `-1.71`

结论很直接：

> 当前 conservative MVP 还没有把 region grounding 推到能够替代 `v2` 的程度。

#### 3.2 Region grounding 是否有用

这一轮最关键的 ablation 其实是：

- `all-global`
- `region object/activity + global scene/style`

结果：

- `all-global`：Macro `50.77`, Hard `34.25`
- `hybrid`：Macro `50.65`, Hard `34.53`

解释：

1. region grounding 并不是完全没信号
2. 它主要体现在 `Hard` 和局部 top-2 消歧
3. 但这点 hardest-case 增益还不足以覆盖整体平均指标的回撤

所以当前最准确的判断不是“region grounding 无效”，而是：

> 当前这版 region-conditioned updater 只捕捉到了局部 hard-case 增益，  
> 但还没有形成足够稳定的整体 belief-update 收益。

#### 3.3 Phase 2 soft routing 是否有用

best `Phase 2`：

- `temperature = 1.0`
- `beta = 0.05`
- `gate_gamma = 0`

test：

- Macro `50.80`
- Hard `34.13`

routing 统计：

- avg weights
  - `object = 0.244`
  - `activity = 0.260`
  - `scene = 0.242`
  - `style = 0.254`
- mean entropy：`1.1917`

这说明：

1. soft routing 比 `Phase 1 hybrid` 的 Macro 略好
2. 但 Hard 反而低于 `Phase 1 hybrid`
3. router 仍然偏平均分配，没有形成明显的 expert preference

所以当前 Phase 2 只能算一个轻量 comparison，不能算 controller 成立。

### 4. 子集与诊断

#### 4.1 low-margin subset

`margin < 1.0` 子集：

| Method | Macro | mAP | Hard |
| --- | ---: | ---: | ---: |
| `SLR-C` | 45.47 | 48.73 | 35.07 |
| `v2` | **46.03** | **49.07** | **37.07** |
| `Phase 1 all-global` | 45.14 | 48.87 | 35.45 |
| `Phase 1 hybrid` | 44.57 | 48.70 | 35.54 |
| `Phase 2 soft routing` | 44.89 | 48.64 | 35.02 |

这里再次说明：

- `Phase 1 hybrid` 在 hardest subset 上确实比 `SLR-C` 稍好
- 但 Macro 和整体质量仍然不如 `SLR-C`
- 与 `v2` 的差距依然明显

#### 4.2 top-2 / pairwise diagnostics

`top-2 disambiguation accuracy`：

- `SLR-C`：`80.32`
- `v2`：`80.18`
- `Phase 1 all-global`：`80.85`
- `Phase 1 hybrid`：**`81.11`**
- `Phase 2 soft routing`：`80.85`

`pairwise ranking accuracy`：

- `SLR-C`：`85.75`
- `v2`：**`86.27`**
- `Phase 1 all-global`：`86.22`
- `Phase 1 hybrid`：`86.10`
- `Phase 2 soft routing`：`86.09`

这组诊断很重要：

1. `Phase 1 hybrid` 在 top-2 局部选择上最好
2. 但 `v2` 的整体 pairwise ranking 仍然最强
3. 说明 region grounding 目前更多是在帮某些 top-2 局部案例，而不是稳定提升全局排序

#### 4.3 Attention / cache sanity checks

- `phase1_identity_max_abs_diff = 0.0`
  - 说明 belief-update 路径在 `beta=0` 时与 `SLR-C` 完全一致
- region cache global feature 对齐误差
  - val max abs diff：`0.00291`
  - test max abs diff：`0.00282`
  - 说明新 patch-token 提取路径与现有 CLIP global cache 是对齐的
- best hybrid attention entropy
  - val：`5.3261`
  - test：`5.3261`
  - 说明当前 candidate-conditioned attention 仍偏平滑，不是非常尖锐的局部读取

### 5. 为什么这轮停在 Phase 2

design 的停止条件里有两个本轮已经命中的信号：

1. `Phase 1` 没有接近 `v2 comparative + gate`
2. `Phase 2` 的 controller 没有形成明显 expert selection，而是接近平均分配

因此本轮不继续 graph reasoning 是正确的，避免把复杂度叠在一个尚未成立的前提上。

### 6. Commands

语法检查：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m py_compile \
  src/utils/region_grounded_reasoning.py \
  scripts/analyze_region_grounded_agent.py \
  tests/test_region_grounded_reasoning.py
```

单测：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m pytest -q \
  tests/test_region_grounded_reasoning.py \
  tests/test_evidence_verification_v2.py \
  tests/test_evidence_verification_v3.py
```

结果：`13 passed`

Smoke run：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_region_grounded_agent.py \
  --device cpu \
  --num-workers 0 \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --region-attn-scale-list 10 \
  --phase1-local-weight-list 1.0 \
  --phase1-global-weight-list 0.5 \
  --phase1-beta-list 0.01 \
  --phase1-gate-gamma-list 0,2 \
  --phase2-temperature-list 1.0 \
  --phase2-beta-list 0.01 \
  --phase2-gate-gamma-list 0,2 \
  --output-dir logs/analysis/smoke_region_grounded_agent_20260312
```

Full run：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python \
  scripts/analyze_region_grounded_agent.py \
  --device cpu \
  --num-workers 0 \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --region-cache-dir logs/analysis/smoke_region_grounded_agent_20260312/_region_cache \
  --output-dir logs/analysis/phase1_region_grounded_agent_20260312
```

### 7. Files changed

- `src/utils/region_grounded_reasoning.py`
- `scripts/analyze_region_grounded_agent.py`
- `tests/test_region_grounded_reasoning.py`
- `docs/record/record_0312_05_RegionGroundedAgent.md`

### 8. Verification

代码验证：

- `py_compile` 通过
- `13` 个相关单测通过

结果验证：

- `SLR-C` 复现为：
  - Macro `51.28`
  - Micro `59.13`
  - Samples `58.47`
  - mAP `53.66`
  - Hard `33.98`
- `v2 best reference` 复现为：
  - Macro `51.73`
  - Micro `59.22`
  - Samples `58.19`
  - mAP `53.78`
  - Hard `35.83`
- `Phase 1 identity` sanity：
  - `max abs diff = 0.0`
- region-cache / global-cache 对齐 sanity：
  - val `0.00291`
  - test `0.00282`

主要产物：

- `logs/analysis/phase1_region_grounded_agent_20260312/main_comparison.csv`
- `logs/analysis/phase1_region_grounded_agent_20260312/phase1_search_results.csv`
- `logs/analysis/phase1_region_grounded_agent_20260312/phase2_search_results.csv`
- `logs/analysis/phase1_region_grounded_agent_20260312/summary.json`

辅助 region cache：

- `logs/analysis/smoke_region_grounded_agent_20260312/_region_cache/val_clip_region.npz`
- `logs/analysis/smoke_region_grounded_agent_20260312/_region_cache/test_clip_region.npz`

### 9. 最终判断

本轮最稳妥的结论是：

> `Region-Grounded Agent` 这条线在当前 conservative MVP 下，  
> 只证明了“candidate-conditioned local evidence 对 hardest/top-2 ambiguity 有一点帮助”，  
> 但还没有证明“region-grounded belief update”已经优于当前 strongest `v2 comparative + gate`，  
> 也没有证明 `soft routing controller` 已经成立。  
> 因此这条线可以保留为研究方向，但不应在当前版本上继续堆 graph reasoning。
