# 实验记录：Gated Distillation Variants

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0316_03_GatedDistillation.md`
- 统一设定：
  - `teacher_input_mode = text_only`
  - `teacher_text_feature_source = full`
  - `seed-fixed`
  - `batch_size = 256`

---

## 1. 对照基线

旧版 dynamic gate 对照来自：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

其 `dynamic_gated_kd` 最优结果：

- Macro `50.31`
- Micro `57.97`
- Samples `59.07`
- mAP `53.02`
- Hard `30.24`

旧版 gate 定义：

- `teacher_weight = 1 - omega_sample`

其中 `omega_sample` 是正类 agreement 的 sample-wise pooled scalar。

---

## 2. 本轮实现

在 [analyze_privileged_distillation.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_privileged_distillation.py) 中新增：

- `--dynamic-kd-variant`
  - `sample_inverse`
  - `sample_affine`
  - `classwise_inverse`
  - `classwise_entropy`
  - `classwise_dynamic_temperature`
- `--entropy-gate-lambda`

本轮实际尝试了设计文档中的三种改动：

1. `classwise_inverse`
   - `teacher_weight_{i,c} = 1 - soft_label_{i,c}`
2. `classwise_entropy`
   - `teacher_weight_{i,c} = max(1 - soft_label_{i,c}, lambda * H_{i,c})`
   - 其中 `H_{i,c}` 是当前 student Bernoulli entropy，归一化到 `[0,1]`
   - 本轮取 `lambda = 1.0`
3. `classwise_dynamic_temperature`
   - `teacher_weight_{i,c} = 1 - soft_label_{i,c}`
   - `tau_{i,c} = tau_base * (2 - soft_label_{i,c})`

说明：

1. 本轮完全按设计文档精神实现为 class-wise 版本
2. 对负类，当前直接使用原始 `soft_label = 0`
   - 所以：
     - `classwise_inverse` 下负类 `teacher_weight = 1`
     - `classwise_dynamic_temperature` 下负类 `tau = 2 * tau_base`

---

## 3. 运行命令

### 3.1 Class-wise Gating

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --teacher-text-feature-source full \
  --dynamic-kd-variant classwise_inverse \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_classwise_inverse_20260316
```

### 3.2 Visual Entropy-Aware Gating

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --teacher-text-feature-source full \
  --dynamic-kd-variant classwise_entropy \
  --entropy-gate-lambda 1.0 \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_classwise_entropy_l1_20260316
```

### 3.3 Dynamic Temperature Scaling

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --teacher-text-feature-source full \
  --dynamic-kd-variant classwise_dynamic_temperature \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_classwise_dyn_temp_20260316
```

---

## 4. 主结果

只比较 `dynamic_gated_kd`：

| Variant | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old sample inverse` | **50.31** | **57.97** | **59.07** | **53.02** | 30.24 |
| `classwise_inverse` | 49.54 | 56.83 | 57.79 | 52.67 | 30.13 |
| `classwise_entropy (lambda=1.0)` | 49.12 | 57.04 | 58.75 | 52.37 | 29.95 |
| `classwise_dynamic_temperature` | 48.99 | 57.86 | 58.03 | 51.15 | **30.68** |

相对旧版 `sample_inverse`：

### 4.1 classwise_inverse

- Macro `-0.76`
- Micro `-1.14`
- Samples `-1.28`
- mAP `-0.35`
- Hard `-0.11`

### 4.2 classwise_entropy

- Macro `-1.19`
- Micro `-0.93`
- Samples `-0.33`
- mAP `-0.65`
- Hard `-0.29`

### 4.3 classwise_dynamic_temperature

- Macro `-1.32`
- Micro `-0.11`
- Samples `-1.04`
- mAP `-1.87`
- Hard `+0.44`

---

## 5. 结论

### 5.1 Class-wise Gating

没有带来提升。

虽然设计上更精细，但在当前实现里：

1. 负类 `soft_label=0`
2. 于是大量负类获得 `teacher_weight=1`
3. KD 信号变得过强，稀释了监督信号

结果上：

- Macro / Micro / Samples 全部下降

### 5.2 Visual Entropy-Aware Gating

也没有超过旧版。

`lambda=1.0` 下：

1. entropy gate 确实让更多类进入 teacher 监督
2. 但当前 teacher 还不是 oracle upper bound
3. 所以额外放大的 teacher 权重没有转化成收益

结果上：

- 比 `classwise_inverse` 更稳一点
- 但仍整体低于旧版

### 5.3 Dynamic Temperature Scaling

这是三者里最值得保留的改动，但仍不足以替代旧版。

优点：

- Hard 从 `30.24` 升到 `30.68`

缺点：

- Macro / Samples / mAP 都下降
- 尤其 mAP 掉得明显

所以当前判断是：

> `Dynamic Temperature` 有一点 hardest-class 潜力，  
> 但单独使用还不够，不能直接替代当前最佳的 `sample_inverse` gate。

---

## 6. 最终建议

当前排序：

1. `sample_inverse` 旧版 gate
2. `classwise_dynamic_temperature`
3. `classwise_inverse`
4. `classwise_entropy (lambda=1.0)`

如果继续往下试，最值得做的是：

1. 保留 `sample_inverse` 作为主线
2. 单独在这个主线上叠加更保守的温度缩放
   - 比如只对低 agreement 正类做 dynamic temperature
3. 不建议继续直接使用当前这版 `classwise_inverse`
4. `classwise_entropy` 若要继续，至少需要重新扫 `lambda`
   - 当前 `lambda=1.0` 没有带来收益

