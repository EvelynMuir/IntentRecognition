# 实验记录：Privileged Distillation with Text-Only Teacher

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0316_02_Distillation.md`
- 本轮输出目录：`logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`
- 对照目录：
  - `logs/analysis/privileged_distillation_image_text_seedfix_20260316`：seed-fixed `image + text teacher`

---

## 1. 变更点

本轮唯一核心改动：

- teacher 从
  - `image_feature + rationale_feature -> MLP -> logits`
- 改为
  - `rationale_feature only -> MLP -> logits`

也就是：

> `Teacher = BGE rationale vector -> MLP -> multi-label logits`

student 与蒸馏损失保持不变：

- baseline：`ASL`
- standard KD：`ASL + KL`
- dynamic gated KD：`omega * ASL + (1 - omega) * KL`

---

## 2. 运行命令

### 2.1 smoke

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --max-test-samples 128 \
  --max-epochs 2 \
  --patience 2 \
  --batch-size 128 \
  --output-dir logs/analysis/privileged_distillation_text_teacher_smoke_20260316
```

### 2.2 full run

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_text_teacher_seedfix_20260316
```

---

## 3. 主结果

主表来自：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_teacher(text_only)` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |
| `baseline` | 48.27 | 52.72 | 53.95 | 51.44 | 29.24 |
| `standard_kd` | **50.39** | 57.48 | 58.11 | 53.60 | **31.28** |
| `dynamic_gated_kd` | 50.11 | **57.48** | **59.19** | **53.82** | 31.18 |

相对 baseline：

- `standard KD`
  - Macro `+2.12`
  - Micro `+4.76`
  - Samples `+4.17`
  - Hard `+2.04`
  - mAP `+2.17`
- `dynamic gated KD`
  - Macro `+1.84`
  - Micro `+4.77`
  - Samples `+5.24`
  - Hard `+1.94`
  - mAP `+2.38`

---

## 4. 与上一版 image+text teacher 的对比

上一版结果：

- `logs/analysis/privileged_distillation_image_text_seedfix_20260316`

上一版主结果：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_teacher(image+text)` | 45.87 | 53.44 | 54.04 | 48.25 | 28.08 |
| `baseline` | 47.40 | 56.42 | 56.41 | 50.67 | 28.45 |
| `standard_kd` | 47.36 | 55.87 | 57.00 | 52.06 | 26.59 |
| `dynamic_gated_kd` | 48.57 | 56.10 | 57.13 | 52.29 | 28.58 |

本轮 `text-only teacher` 相比上一版：

1. teacher 自身：
   - Macro 基本持平：`45.80` vs `45.87`
   - Hard 更高：`30.21` vs `28.08`
   - 但 Micro / Samples / mAP 更低
2. student 蒸馏效果：
   - `standard KD`
     - Macro `49.59` vs `47.36`
     - Hard `30.44` vs `26.59`
   - `dynamic gated KD`
     - Macro `50.31` vs `48.57`
     - Samples `59.07` vs `57.13`
     - Hard `30.24` vs `28.58`

3. 由于脚本现在对 `teacher / baseline / standard_kd / dynamic_kd` 分别固定 seed，
   这轮对比是可直接横向比较的，不再受前序模型 RNG 消耗影响。

结论：

> 纯文本 teacher 没有比 `image+text teacher` 更弱，  
> 反而在 student 蒸馏后的最终指标上更好。

---

## 5. 解释

这个结果说明了两点：

1. 在当前设定下，teacher 侧再拼接 frozen image feature 不一定有益
2. rationale 文本本身已经包含足够强的高层语义信号，足以充当蒸馏 teacher

一个合理推测是：

- `image+text teacher` 中的 image branch 给 teacher 引入了额外噪声或冗余
- 而 `text-only teacher` 更纯粹地表达了 rationale 中的语义决策边界

换句话说：

> 对 teacher 而言，当前更像是在学习“文本先验分布”，  
> 而不是做真正的多模态融合。

---

## 6. Agreement Slice

切片表：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316/agreement_slice_analysis.csv`

仍然只能做 train slice，因为公开 `val/test` 没有 agreement soft label。

`agreement = 1/3` 子集上：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 33.10 | 36.08 | 36.26 | **25.97** |
| `standard_kd` | 31.64 | 36.15 | 36.36 | 23.69 |
| `dynamic_gated_kd` | 31.30 | 36.04 | **36.48** | 22.18 |

这里的现象和上一版类似：

1. noisy slice 上，`Samples` 有提升
2. 但 `Hard` 仍未压过 baseline
3. 所以“dynamic gate 专门碾压最低 agreement slice”这条证据还不够强

---

## 7. 当前结论

如果只在这两版之间选一版继续推进，建议保留：

- `text-only teacher`

理由：

1. 结构更简单
2. teacher 本身不比 `image+text` 差
3. student 蒸馏结果更好
4. 更符合“privileged information = rationale text prior”这条叙事

当前更合理的主线表述应改为：

> `Text-only rationale teacher` is a stronger and cleaner distillation target than the previous multimodal teacher in this cached-feature setting.

---

## 8. Rationale Ablation

本轮补做了 teacher-side rationale source 消融，保持：

- `teacher_input_mode = text_only`
- `student / seed / optimizer / batch_size / patience` 全部固定

三个版本：

1. `full`
   - 使用 `features`
   - 对应完整 rationale 文本
2. `step1_only`
   - 使用 `step1_features`
3. `step1_step2`
   - 使用 `pos_features`
   - 即 `Step 1 + Step 2`

输出目录：

- `full`：
  - `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`
- `step1_only`：
  - `logs/analysis/privileged_distillation_step1_only_seedfix_20260316`
- `step1_step2`：
  - `logs/analysis/privileged_distillation_step12_seedfix_20260316`

### 8.1 Teacher 自身

| Teacher Text Source | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | **45.80** | 51.42 | 52.61 | **47.04** | **30.21** |
| `step1_only` | 44.37 | **51.64** | **53.44** | 45.53 | 22.91 |
| `step1_step2` | 44.52 | 50.17 | 52.31 | 45.87 | 25.81 |

结论：

1. teacher 自身看，`full rationale` 最稳
2. `step1_only` 会明显伤 `Hard`
3. `step1_step2` 比 `step1_only` 好，但仍不如 `full`

### 8.2 Standard KD

| Teacher Text Source | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 49.59 | **58.26** | **58.88** | **53.18** | **30.44** |
| `step1_only` | **49.76** | 57.59 | 58.73 | 52.89 | 29.91 |
| `step1_step2` | 49.37 | 57.03 | 57.93 | 52.52 | 29.14 |

结论：

1. `step1_only` 在 Macro 上略高于 `full`
2. 但 `full` 在 `Micro / Samples / mAP / Hard` 上都更好
3. `step1_step2` 整体落后于 `full`

### 8.3 Dynamic Gated KD

| Teacher Text Source | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | **50.31** | **57.97** | **59.07** | 53.02 | **30.24** |
| `step1_only` | 49.81 | 56.39 | 57.65 | 52.82 | 29.28 |
| `step1_step2` | 49.69 | 54.45 | 55.97 | **53.08** | 28.82 |

结论：

1. `dynamic gated KD` 下，`full rationale` 是明确最优
2. `step1_only` 和 `step1_step2` 都有退化
3. `step1_step2` 虽然 mAP 略高 `+0.06`，但分类指标整体明显更差，不能视为更优

### 8.4 最终结论

这组三路消融的结论很清楚：

1. `full rationale` 仍然是最好的 teacher 文本来源
2. 只保留 `Step 1`
   - 会损伤 teacher 自身的 hardest-class 判别
   - 也不能带来更好的最终 student
3. `Step 1 + Step 2`
   - 比 `Step 1 only` 略稳
   - 但仍没有超过 `full`

所以后续主实验建议继续使用：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`

---

## 9. Gate Formula Ablation

本轮额外测试了新的 dynamic gate：

- 旧版：
  - `teacher_weight = 1 - omega`
  - `supervised_weight = omega`
- 新版：
  - `teacher_weight = alpha + beta * (1 - omega)`
  - `supervised_weight = 1 - teacher_weight`
  - 取值：
    - `alpha = 0.3`
    - `beta = 0.7`

运行目录：

- 新 gate：
  - `logs/analysis/privileged_distillation_text_teacher_alpha03_beta07_20260316`
- 对照旧 gate：
  - `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

在当前最佳设定下对比：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`

### 9.1 结果对比

| Gate | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old: 1 - omega` | **50.31** | **57.97** | **59.07** | **53.02** | **30.24** |
| `new: 0.3 + 0.7 * (1 - omega)` | 49.24 | 56.27 | 57.07 | 52.38 | 28.27 |

相对旧 gate，新 gate 变化为：

- Macro `-1.06`
- Micro `-1.70`
- Samples `-2.00`
- mAP `-0.65`
- Hard `-1.97`

### 9.2 解释

这版 gate 的问题很直观：

1. 对高一致性样本，teacher 仍有至少 `30%` 权重
2. 在当前 teacher 还不是 oracle upper bound 的前提下，这会把 teacher 偏差硬性注入到 easy / clean 样本
3. 结果就是：
   - 全局指标下降
   - hardest classes 也下降

所以当前不建议采用：

- `teacher_weight = 0.3 + 0.7 * (1 - omega)`

更稳的选择仍然是：

- `teacher_weight = 1 - omega`
