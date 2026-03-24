# 实验记录：Distillation with SLR-C Prior

## 0. 文档信息

- 日期：`2026-03-17`
- 目标：把 `SLR-C` 接入当前 cache-based distillation 框架
- 主实验输出目录：`logs/analysis/distillation_slrc_20260317`

---

## 1. 方法定义

这轮不是简单把 `SLR-C` 当成对照，而是把它真正接进 student：

1. 先固定重建 `scenario SLR-C`
2. 把 `SLR-C logits` 作为 frozen prior
3. student 不再从零输出 logits，而是学习一个 residual：

\[
z^{final}(x) = z^{slr-c}(x) + r_\theta(v_x)
\]

其中：

- `z^{slr-c}(x)`：由现有 cache 复建的固定 `SLR-C` logits
- `v_x`：当前 CLIP image feature cache
- `r_\theta`：一个小型 MLP residual head

然后在这个 prior 上训练三种 student：

- `slr_c_residual_sup`
- `slr_c_residual_standard_kd`
- `slr_c_residual_dynamic_kd`

teacher 仍然复用当前最优的：

- `text-only teacher`
- `full rationale`

---

## 2. 本轮实现

新增脚本：

- [scripts/analyze_distillation_slrc.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/analyze_distillation_slrc.py)

核心步骤：

1. 从现有 `train/val/test *_base.npz` 和 `*_clip.npz` 复建 `scenario SLR-C`
2. 从 `privileged_distillation_text_teacher_seedfix_20260316/teacher_best.pt` 载入 teacher
3. 用固定 `SLR-C logits` 作为 prior，训练 residual student
4. 输出：
   - `main_comparison.csv`
   - `summary.json`

---

## 3. 运行命令

```bash
PROJECT_ROOT=/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra \
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_distillation_slrc.py \
  --device cuda \
  --output-dir logs/analysis/distillation_slrc_20260317
```

---

## 4. 主结果

主表来自：

- `logs/analysis/distillation_slrc_20260317/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher_text_only` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |
| `slr_c_fixed` | 51.08 | 58.81 | 58.02 | 53.66 | 33.98 |
| `slr_c_residual_sup` | 50.41 | 57.70 | 57.56 | 52.18 | 34.77 |
| `slr_c_residual_standard_kd` | **51.77** | 58.97 | 58.99 | 53.42 | **36.43** |
| `slr_c_residual_dynamic_kd` | 51.52 | **59.75** | **60.30** | **54.04** | 34.60 |

### 4.1 相对固定 SLR-C

`slr_c_residual_standard_kd`：

- Macro `+0.69`
- Micro `+0.16`
- Samples `+0.97`
- mAP `-0.24`
- Hard `+2.46`

`slr_c_residual_dynamic_kd`：

- Macro `+0.44`
- Micro `+0.94`
- Samples `+2.29`
- mAP `+0.38`
- Hard `+0.63`

### 4.2 相对当前不带 SLR-C 的最佳主线

当前最佳主线参考：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

其中 `dynamic_gated_kd` 为：

- Macro `50.31`
- Micro `57.97`
- Samples `59.07`
- mAP `53.02`
- Hard `30.24`

对比本轮：

`slr_c_residual_standard_kd`：

- Macro `+1.46`
- Micro `+1.00`
- Samples `-0.08`
- mAP `+0.40`
- Hard `+6.19`

`slr_c_residual_dynamic_kd`：

- Macro `+1.21`
- Micro `+1.78`
- Samples `+1.23`
- mAP `+1.02`
- Hard `+4.36`

---

## 5. 结论

这轮结论非常明确：

1. `SLR-C` 接进当前框架是成立的
2. 不是小修小补，而是正式正增益
3. 最强的两个版本分别是：
   - 如果你更看重 `Macro / Hard`：`slr_c_residual_standard_kd`
   - 如果你更看重 `Micro / Samples / mAP`：`slr_c_residual_dynamic_kd`

最关键的是：

> `SLR-C` 作为 frozen prior，和当前的 text-only distillation 是互补的。  
> 它不是替代 teacher，而是把 student 的起点从“纯视觉特征自由分类”提升到了“强 scenario-aware proposal + residual correction”。

---

## 6. 方法表述建议

如果要把这条线写进最终方法，建议口径是：

> `We use fixed SLR-C logits as a strong scenario-aware prior,`  
> `and train a residual student on top with teacher-guided distillation.`

也就是：

1. `SLR-C` 负责 proposal / calibrated prior
2. residual student 负责视觉特征上的细修正
3. text-only teacher 负责在 noisy / hard cases 上给出 softer supervisory signal

---

## 7. 当前推荐

如果只选一个最终版本继续推进，建议优先保留：

- `slr_c_residual_dynamic_kd`

原因：

1. 它在 `Micro / Samples / mAP` 上全面最好
2. `Hard` 也明显高于无 `SLR-C` 主线
3. Macro 虽然略低于 `slr_c_residual_standard_kd`，但差距不大

如果论文更强调 hardest categories，也可以保留：

- `slr_c_residual_standard_kd`

