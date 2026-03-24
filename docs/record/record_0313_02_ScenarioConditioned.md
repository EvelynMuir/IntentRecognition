# 实验记录：Scenario-Conditioned Decision Learning Sanity Check

## 0. 文档信息

- 日期：`2026-03-13`
- 对应设计：`docs/design/design_0313_02_ScenarioConditioned.md`
- 主实验输出目录：`logs/analysis/scenario_conditioned_decision_20260313`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. main conclusions

1. 本轮严格按 design 的“先做轻量 sanity check”建议，没有直接上 condition relation graph，而是只在当前 strongest `SLR-C` 上加：
   - static bias
   - scenario-conditioned bias
   - scenario-conditioned threshold
   - scenario-conditioned bias + threshold
2. 最好结果来自：
   - `SLR-C + bias + threshold (soft)`
   - `temperature = 2.0`
   - `lr = 0.03`
   - `weight_decay = 0.0`
3. 这个 best variant 的 test `class-wise` 指标为：
   - Macro `50.91`
   - Micro `58.91`
   - Samples `58.44`
   - mAP `53.66`
   - Hard `33.43`
4. 它**没有超过**原始 `SLR-C`：
   - `SLR-C`：Macro `51.28`, Hard `33.98`
   - best scenario-conditioned：Macro `50.91`, Hard `33.43`
   也就是：
   - Macro `-0.37`
   - Hard `-0.55`
5. 但 conditioned 版本**确实略好于 static bias**：
   - `SLR-C + static bias`：Macro `50.69`, Hard `33.15`
   - `SLR-C + scenario-conditioned bias (soft)`：Macro `50.86`, Hard `33.76`
   - `SLR-C + scenario-conditioned bias (hard)`：Macro `50.84`, Hard `33.73`
   说明 scenario 不是完全没用，它在 decision layer 里有**弱正信号**，只是当前这版轻量 adapter 还不够把它转成正式主结果。
6. `scenario-conditioned threshold (soft)` 和 `scenario-conditioned bias (soft)` 落在几乎同一结果点：
   - 两者 test Macro 都是 `50.86`
   - 两者 test Hard 都是 `33.76`
   这说明在当前“class-wise threshold 再搜索”的评测口径下，threshold-only 和 bias-only 的作用几乎退化成同一种 logit shift。
7. 结论应当写得更谨慎一些：

> scenario 作为 decision-layer context **有弱信号，但证据不够强**。  
> 当前这轮 sanity check 还不足以支持“直接升级为 scenario-conditioned relation graph 主线”。

---

## 2. 本轮实现

本轮新增：

- `scripts/analyze_scenario_conditioned_decision.py`
  - 复用现有 `train/val/test *_base.npz` 与 `*_clip.npz`
  - 复建 `scenario SLR-C`
  - 从现有 `scenario` text pool 得到 scenario prior
  - 在 `SLR-C logits` 之上训练超轻量 decision adapter
  - 输出 `main_comparison.csv` 与 `summary.json`

实现边界：

1. 本轮没有改 backbone。
2. 本轮没有重训主分类器。
3. 本轮没有做 label relation graph / message passing。
4. 本轮只验证 design 中最小 sanity check：

[
z'_j = z_j + b_j(s)
]

和

[
\tau_j(x) = \tau_j^{base} + \Delta \tau_j(s)
]

对应的 logit-space 实现。

---

## 3. 运行命令

语法检查：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -m py_compile scripts/analyze_scenario_conditioned_decision.py
```

主实验：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -u scripts/analyze_scenario_conditioned_decision.py \
  --device cpu \
  --output-dir logs/analysis/scenario_conditioned_decision_20260313 \
  --max-epochs 80 \
  --patience 12 \
  --lrs 0.03,0.01 \
  --weight-decays 0.0 \
  --temperatures 1.0,2.0
```

说明：

- 这轮是 decision-layer 小模型，直接 CPU 就够了
- 没有额外补单测，只有 `py_compile`

---

## 4. 固定设置

### 4.1 Base system

- baseline logits：来自 cache `*_base.npz`
- scenario prior：由现有 `scenario` text pool 与 CLIP image features 计算
- `SLR-C` 重建方式：
  - `topk = 10`
  - `alpha = 0.3`
  - `fusion_mode = add_norm`

### 4.2 搜索空间

- `lr ∈ {0.03, 0.01}`
- `weight_decay = 0.0`
- `temperature ∈ {1.0, 2.0}`
- `max_epochs = 80`
- `patience = 12`

### 4.3 变体

- `SLR-C + static bias`
- `SLR-C + scenario-conditioned bias (soft)`
- `SLR-C + scenario-conditioned bias (hard)`
- `SLR-C + scenario-conditioned threshold (soft)`
- `SLR-C + bias + threshold (soft)`

---

## 5. 主结果

主表来自：`logs/analysis/scenario_conditioned_decision_20260313/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR-C` | **51.28** | **59.13** | **58.47** | **53.66** | **33.98** |
| `SLR-C + static bias` | 50.69 | 58.49 | 58.02 | 53.66 | 33.15 |
| `SLR-C + scenario-conditioned bias (soft)` | 50.86 | 58.53 | 57.89 | **53.66** | **33.76** |
| `SLR-C + scenario-conditioned bias (hard)` | 50.84 | 58.52 | 57.88 | **53.66** | 33.73 |
| `SLR-C + scenario-conditioned threshold (soft)` | 50.86 | 58.53 | 57.89 | **53.66** | **33.76** |
| `SLR-C + bias + threshold (soft)` | 50.91 | 58.91 | 58.44 | 53.66 | 33.43 |

### 5.1 直接比较

相对 `SLR-C`：

- `static bias`
  - Macro `-0.59`
  - Hard `-0.83`
- `scenario bias (soft)`
  - Macro `-0.42`
  - Hard `-0.22`
- `scenario bias (hard)`
  - Macro `-0.44`
  - Hard `-0.25`
- `scenario threshold (soft)`
  - Macro `-0.42`
  - Hard `-0.22`
- `bias + threshold (soft)`
  - Macro `-0.37`
  - Hard `-0.55`

解释：

1. scenario-conditioned 版本整体上**比 static bias 更好**
2. 但还没有真正超过 `SLR-C`
3. 所以当前结论应该是：

> condition 有信号，但现阶段只是“减小损失”，还不是“形成正式增益”。

### 5.2 soft vs hard assignment

- `soft`：Macro `50.86`, Hard `33.76`
- `hard`：Macro `50.84`, Hard `33.73`

两者几乎持平，说明：

1. 当前 scenario posterior 的 sharpness 不是主要瓶颈
2. hard assignment 并没有明显破坏性能
3. 但 soft assignment 也没有展现出足够强的优势

### 5.3 bias vs threshold

这一轮最值得注意的现象是：

- `scenario-conditioned bias (soft)` 和 `scenario-conditioned threshold (soft)` 完全打到同一结果点

这意味着在当前实现里：

1. threshold shift 基本退化成了另一种 logit bias
2. 当前 sanity check 还没有真正学到“更合理的 scenario-dependent decision boundary”
3. 只做 class-wise threshold 之后，再叠一个 threshold adapter，额外信息并不充分

---

## 6. 诊断

### 6.1 best variant 的 per-class gain

相对 `SLR-C`，best variant (`bias + threshold soft`) 最大的提升类别包括：

- `HardWorking`：`+0.0163`
- `InLove`：`+0.0159`
- `ManagableMakePlan`：`+0.0094`
- `Health`：`+0.0062`
- `CuriousAdventurousExcitingLife`：`+0.0043`

这些增益都偏小，没有出现真正“被大幅救回”的 hardest 类。

### 6.2 confusion pairs

`SLR-C` 的高频 confusion pair 仍然基本保留，例如：

- `Attractive -> EnjoyLife`
- `Playful -> CuriousAdventurousExcitingLife`
- `FineDesignLearnArt-Art -> CreativeUnique`

best variant 下只出现了轻微重排，没有看到 dominant confusion pair 被系统性修正。

这进一步说明：

> 当前 scenario-conditioned adapter 还没有真正重写 label decision structure，  
> 它更像是在已有 `SLR-C` 上做小幅的 score reshaping。

---

## 7. 结论与下一步

### 7.1 这轮应该怎么定性

这轮最准确的定性不是“完全失败”，也不是“主线成立”，而是：

> **weak positive internal signal, but no formal leaderboard gain**

也就是：

1. `conditioned > static` 说明 scenario 确实带来了一点 decision-layer signal
2. `best < SLR-C` 说明这个信号还不足以支撑当前主线升级

### 7.2 是否继续做 conditional relation graph

基于这轮结果，我不建议立刻跳到完整 relation graph，原因是：

1. 最轻量的 bias / threshold sanity check 还没涨
2. threshold-only 仍然退化成 bias-like behavior
3. top confusion pairs 也没有出现清晰修正

如果还想继续走 `ScenarioConditioned`，更合理的下一步应该是二选一：

1. 明确换一个更能区分 bias vs threshold 的评测与实现
   - 例如减少后验 class-wise threshold 搜索，让 scenario-threshold 的价值真正显式出来
2. 把 condition 引入 candidate-local relation / pairwise correction
   - 而不是继续做全局 logit shift

### 7.3 当前最保守结论

对 `design_0313_02_ScenarioConditioned.md` 的当前判断应写为：

> scenario 作为中间上下文变量**并非完全无效**，  
> 但在目前这版“decision-layer bias / threshold adapter”下，还没有形成正式超过 `SLR-C` 的证据。  
> 因此，这条线可以保留，但还不能替代当前 strongest `SLR-C` 主线。
