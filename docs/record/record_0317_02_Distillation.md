# Result Notes

## Final Method

当前主线方法：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`
- `dynamic_kd_variant = sample_inverse`

结果目录：

- `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

主结果：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 47.40 | 56.42 | 56.41 | 50.67 | 28.45 |
| `standard_kd` | 49.59 | **58.26** | 58.88 | **53.18** | **30.44** |
| `dynamic_gated_kd` | **50.31** | 57.97 | **59.07** | 53.02 | 30.24 |
| `teacher` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |

结论：

- `dynamic_gated_kd` 是当前最佳主方法
- `standard_kd` 在 `Micro / Hard` 上更强

## Ablations

### Teacher Input

对照：

- `image_text`: `logs/analysis/privileged_distillation_image_text_seedfix_20260316`
- `text_only`: `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

| Teacher Input | Dynamic Macro | Dynamic Micro | Dynamic Samples | Dynamic mAP | Dynamic Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image + text` | 48.57 | 56.10 | 57.13 | 52.29 | 28.58 |
| `text_only` | **50.31** | **57.97** | **59.07** | **53.02** | **30.24** |

结论：

- `text_only teacher` 明显优于 `image + text teacher`

### Rationale Source

对照目录：

- `full`: `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`
- `step1_only`: `logs/analysis/privileged_distillation_step1_only_seedfix_20260316`
- `step1_step2`: `logs/analysis/privileged_distillation_step12_seedfix_20260316`

`dynamic_gated_kd`：

| Text Source | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | **50.31** | **57.97** | **59.07** | 53.02 | **30.24** |
| `step1_only` | 49.81 | 56.39 | 57.65 | 52.82 | 29.28 |
| `step1_step2` | 49.69 | 54.45 | 55.97 | **53.08** | 28.82 |

结论：

- `full rationale` 仍然是最优 teacher 文本来源

### Gate Formula

`alpha + beta * (1 - omega)` 对照：

- 新版：`logs/analysis/privileged_distillation_text_teacher_alpha03_beta07_20260316`
- 旧版：`logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

| Gate | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher_weight = 1 - omega` | **50.31** | **57.97** | **59.07** | **53.02** | **30.24** |
| `teacher_weight = 0.3 + 0.7 * (1 - omega)` | 49.24 | 56.27 | 57.07 | 52.38 | 28.27 |

结论：

- 给 clean/easy 样本固定注入 teacher 权重会伤整体表现
- 不建议使用 `0.3 + 0.7 * (1 - omega)`

### Gated Distillation Variants

对照目录：

- `classwise_inverse`: `logs/analysis/privileged_distillation_classwise_inverse_20260316`
- `classwise_entropy`: `logs/analysis/privileged_distillation_classwise_entropy_l1_20260316`
- `classwise_dynamic_temperature`: `logs/analysis/privileged_distillation_classwise_dyn_temp_20260316`

| Variant | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sample_inverse` | **50.31** | **57.97** | **59.07** | **53.02** | 30.24 |
| `classwise_inverse` | 49.54 | 56.83 | 57.79 | 52.67 | 30.13 |
| `classwise_entropy` | 49.12 | 57.04 | 58.75 | 52.37 | 29.95 |
| `classwise_dynamic_temperature` | 48.99 | 57.86 | 58.03 | 51.15 | **30.68** |

结论：

- 三种改动都没有超过当前 `sample_inverse`
- `dynamic_temperature` 只在 `Hard` 上略有提升

### Feature-Level Soft-SupCon

对照目录：

- `without_feature`: `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`
- `with_feature`: `logs/analysis/privileged_distillation_with_feature_w01_fix_20260316`

配置：

- `feature_distill_mode = soft_supcon`
- `feature_distill_weight = 0.1`
- `feature_distill_temperature = 0.1`

| Setting | Method | Macro | Micro | Samples | mAP | Hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `without_feature` | `standard_kd` | 49.59 | **58.26** | **58.88** | 53.18 | 30.44 |
| `with_feature` | `standard_kd` | **50.74** | 58.02 | 58.19 | **53.30** | **31.29** |
| `without_feature` | `dynamic_gated_kd` | **50.31** | **57.97** | **59.07** | **53.02** | **30.24** |
| `with_feature` | `dynamic_gated_kd` | 49.68 | 57.51 | 58.76 | 52.85 | 29.71 |

结论：

- `Soft-SupCon` 对 `standard_kd` 有一点帮助
- 但对当前主线 `dynamic_gated_kd` 是负增益

## ResNet101

plain `ResNet101 + binarize_softprob=true` 最终训练 run：

- run: `logs/train/runs/2026-03-17_11-26-46`
- best ckpt: `logs/train/runs/2026-03-17_11-26-46/checkpoints/epoch_012.ckpt`
- cache: `logs/analysis/resnet101_distill_cache_binarized_20260317`
- distillation: `logs/analysis/resnet101_privileged_distillation_binarized_20260317`

蒸馏设定保持主线不变：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`
- `dynamic_kd_variant = sample_inverse`

主结果：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher` | 45.80 | 51.42 | 52.61 | 47.04 | 30.21 |
| `baseline` | 36.56 | 41.87 | 43.10 | 36.31 | 18.99 |
| `standard_kd` | 37.50 | **45.34** | **47.65** | 38.68 | 19.69 |
| `dynamic_gated_kd` | **39.35** | 44.73 | 46.91 | **39.08** | **21.53** |

结论：

- ResNet101 baseline 明显弱于 CLIP backbone
- 但 teacher 仍然很强，所以蒸馏收益更明显
- 在 ResNet101 上，`dynamic_gated_kd` 比 `standard_kd` 更有优势

## Streamlit 服务

用于可视化：

- baseline 与 GT 一致，但 student 与 GT 不一致
- baseline 与 GT 不一致，但 student 与 GT 一致

默认脚本：

- [scripts/streamlit_baseline_student_mismatch.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/scripts/streamlit_baseline_student_mismatch.py)

启动命令：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -m streamlit run \
  scripts/streamlit_baseline_student_mismatch.py \
  --server.headless true \
  --server.port 8501
```

默认读取：

- run: `logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`
- cache: `logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

访问地址：

- local: `http://localhost:8501`
- external: `http://38.84.161.210:8501`

筛选模式：

- `Baseline 对，Student 错`
- `Baseline 错，Student 对`
