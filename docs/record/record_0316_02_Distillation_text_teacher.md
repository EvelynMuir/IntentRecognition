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
