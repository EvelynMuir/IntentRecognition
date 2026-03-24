# 实验记录：ResNet101 Backbone Full Method

## 0. 文档信息

- 日期：`2026-03-18`
- 目标：在 `ResNet101` backbone 上跑完整方法
- 输出目录：`logs/analysis/resnet101_distillation_slrc_20260318`

---

## 1. 本轮方法定义

这轮不是 plain ResNet101 distillation，而是完整结构：

> `CLIP-based SLR-C prior + ResNet101 residual student + text-only teacher distillation`

具体组成：

1. student backbone：
   - `ResNet101`
   - cache 来自 `logs/analysis/resnet101_distill_cache_binarized_20260317/_cache`
2. fixed prior：
   - `SLR-C`
   - 由 CLIP cache `logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`
     重建
3. teacher：
   - `text-only`
   - `full rationale`
4. student form：

\[
z^{stu}(x) = z^{slr-c}(x) + r_\theta(v_x^{resnet})
\]

其中 `z^{slr-c}` 来自 CLIP-based scenario prior，`r_\theta` 学习的是 ResNet feature 上的 residual。

说明：

1. 这里的 backbone 替换只发生在 student residual 分支
2. `SLR-C` 仍然用 CLIP cache 重建，因为它本身依赖 CLIP text-image 对齐空间

---

## 2. 本轮实现

沿用脚本：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

本轮新增能力：

- 支持 student cache 和 SLR cache 解耦
- 通过：
  - `--reuse-cache-dir` 指 student backbone cache
  - `--slr-cache-dir` 指 CLIP cache for SLR-C prior

---

## 3. 运行命令

```bash
PROJECT_ROOT=/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra \
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_distillation_slrc.py \
  --reuse-cache-dir logs/analysis/resnet101_distill_cache_binarized_20260317/_cache \
  --slr-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --teacher-run-dir logs/analysis/privileged_distillation_text_teacher_seedfix_20260316 \
  --device cuda \
  --output-dir logs/analysis/resnet101_distillation_slrc_20260318
```

---

## 4. 主结果

主表来自：

- `logs/analysis/resnet101_distillation_slrc_20260318/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher_text_only` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |
| `slr_c_fixed` | 42.25 | 51.84 | 51.17 | 43.08 | 23.83 |
| `slr_c_residual_sup` | **42.87** | **52.96** | **52.08** | 42.65 | 23.21 |
| `slr_c_residual_standard_kd` | 42.14 | 52.10 | 51.72 | **43.34** | 23.22 |
| `slr_c_residual_dynamic_kd` | 41.53 | 51.61 | 51.80 | 43.18 | 22.10 |

---

## 5. 对比分析

### 5.1 相对 ResNet101 plain distillation

参考：

- `logs/analysis/resnet101_privileged_distillation_binarized_20260317`

其中 `dynamic_gated_kd` 为：

- Macro `39.35`
- Micro `44.73`
- Samples `46.91`
- mAP `39.08`
- Hard `21.53`

本轮 `slr_c_residual_dynamic_kd`：

- Macro `41.53`
- Micro `51.61`
- Samples `51.80`
- mAP `43.18`
- Hard `22.10`

相对 plain ResNet101 dynamic distillation：

- Macro `+2.18`
- Micro `+6.88`
- Samples `+4.89`
- mAP `+4.10`
- Hard `+0.57`

这说明：

1. `SLR-C prior` 对弱 backbone 仍然是有帮助的
2. 即使 residual + KD 没有超过 fixed SLR-C，本身也明显优于纯 ResNet101 distillation

### 5.2 相对 fixed SLR-C

固定 `SLR-C`：

- Macro `42.25`
- Micro `51.84`
- Samples `51.17`
- mAP `43.08`
- Hard `23.83`

各残差版相对 `SLR-C`：

- `residual_sup`
  - Macro `+0.62`
  - Micro `+1.12`
  - Samples `+0.91`
  - mAP `-0.43`
  - Hard `-0.62`
- `residual_standard_kd`
  - Macro `-0.11`
  - Micro `+0.26`
  - Samples `+0.55`
  - mAP `+0.26`
  - Hard `-0.61`
- `residual_dynamic_kd`
  - Macro `-0.72`
  - Micro `-0.23`
  - Samples `+0.63`
  - mAP `+0.10`
  - Hard `-1.73`

最重要的现象：

1. `SLR-C fixed` 本身已经很强
2. 在 ResNet101 上，残差 student 很难进一步稳定提升 `Hard`
3. 最稳的增益来自 `residual_sup`，但它只改善了 `Macro / Micro / Samples`
4. `dynamic KD` 在 CLIP backbone 上是主力，但在 ResNet101 full method 上没有继续带来提升

---

## 6. 结论

这轮结果不能支撑“ResNet101 backbone 的完整方法比 fixed SLR-C 更强”，但能说明两件事：

1. `SLR-C prior` 对 ResNet101 仍然有显著帮助
2. `ResNet101 full method` 明显优于 plain ResNet101 distillation

更准确的定性是：

> 在弱 backbone 场景下，`SLR-C` 是主要性能来源；  
> residual distillation 只能带来有限修正，且当前没有正式超过 fixed `SLR-C`。

所以如果论文只需要一个 backbone ablation，这轮结果应写成：

- `CLIP backbone`：完整方法成立且收益明显
- `ResNet101 backbone`：完整方法仍优于 plain distillation，但无法稳定超过 fixed SLR-C

---

## 7. 推荐写法

论文里可以这样表述：

> Replacing the strong CLIP backbone with ResNet101 degrades all variants substantially.  
> Although adding the fixed SLR-C prior still improves over plain ResNet distillation,  
> the full residual distillation stack no longer consistently surpasses fixed SLR-C,  
> indicating that the proposed method benefits from a reasonably strong visual backbone.

