# 实验记录：ResNet101 Binarized Distillation

## 0. 文档信息

- 日期：`2026-03-17`
- 目标：在 plain `ResNet101` backbone 上复现当前最佳的 text-only teacher 蒸馏设定
- 训练 run：
  - `logs/train/runs/2026-03-17_11-26-46`
- cache：
  - `logs/analysis/resnet101_distill_cache_binarized_20260317`
- 蒸馏输出：
  - `logs/analysis/resnet101_privileged_distillation_binarized_20260317`

---

## 1. 训练配置

plain ResNet101 重新训练时使用：

- `experiment=intentonomy_resnet101`
- `+data.binarize_softprob=true`
- `data.batch_size=64`
- `model.compile=false`
- `+trainer.precision=16-mixed`
- `seed=20260317`

说明：

1. 原始 plain `ResNet101` run 不带 `binarize_softprob`
2. 本轮为了和蒸馏设定对齐，重新训练了一个 binarized-label 版本
3. 最终 best checkpoint 是：
   - `logs/train/runs/2026-03-17_11-26-46/checkpoints/epoch_012.ckpt`

---

## 2. Distillation 设定

蒸馏沿用当前主线：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`
- `dynamic_kd_variant = sample_inverse`

ResNet101 cache 提取后，特征维度为：

- `2048`

---

## 3. 主结果

主表来自：

- `logs/analysis/resnet101_privileged_distillation_binarized_20260317/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_teacher` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |
| `baseline` | 36.56 | 41.87 | 43.10 | 36.31 | 18.99 |
| `standard_kd` | 37.50 | 45.34 | 47.65 | 38.68 | 19.69 |
| `dynamic_gated_kd` | **39.35** | **44.73** | **46.91** | **39.08** | **21.53** |

相对 baseline：

- `standard KD`
  - Macro `+0.94`
  - Micro `+3.47`
  - Samples `+4.55`
  - mAP `+2.37`
  - Hard `+0.70`
- `dynamic gated KD`
  - Macro `+2.79`
  - Micro `+2.86`
  - Samples `+3.81`
  - mAP `+2.77`
  - Hard `+2.54`

---

## 4. 结论

这轮的信号很干净：

1. ResNet101 baseline 比 CLIP backbone 弱很多
2. teacher 完全没有变弱，因此 teacher 对 student 的优势更大
3. 在 ResNet101 上，`dynamic_gated_kd` 的收益比 `standard_kd` 更明显

也就是说：

> 当视觉 backbone 较弱时，  
> `text-only teacher + dynamic gated distillation` 的帮助会更明显。

这是目前这条方法线里一个比较重要的正结果。

