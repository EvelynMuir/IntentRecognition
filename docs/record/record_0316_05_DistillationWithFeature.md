# 实验记录：Distillation with Feature-Level Soft-SupCon

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0316_04_DistillationWithFeature.md`
- 输出目录：`logs/analysis/privileged_distillation_with_feature_w01_fix_20260316`
- 对照目录：`logs/analysis/privileged_distillation_text_teacher_seedfix_20260316`

---

## 1. 方案实现

本轮在现有蒸馏脚本上新增了一个 feature-level regularizer：

- `feature_distill_mode = soft_supcon`

实现位置：

- `scripts/analyze_privileged_distillation.py`

具体做法：

1. student MLP 拆成：
   - `encoder`
   - `classifier`
   - `feature_proj`
2. 取 student hidden feature 经过 `feature_proj` 后做 `L2 normalize`
3. 用 teacher soft probability `P_tea` 计算 batch 内两两 cosine similarity
4. 将这个相似度归一化为 soft target 分布
5. 用它监督 student feature 的 batch 内 soft contrastive objective

本轮配置：

- `teacher_input_mode = text_only`
- `teacher_text_feature_source = full`
- `dynamic_kd_variant = sample_inverse`
- `feature_distill_weight = 0.1`
- `feature_distill_temperature = 0.1`
- `feature_proj_dim = 256`

运行命令：

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --teacher-input-mode text_only \
  --teacher-text-feature-source full \
  --dynamic-kd-variant sample_inverse \
  --feature-distill-mode soft_supcon \
  --feature-distill-weight 0.1 \
  --feature-distill-temperature 0.1 \
  --feature-proj-dim 256 \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_with_feature_w01_fix_20260316
```

---

## 2. 主结果

对照不带 feature loss 的最佳文本 teacher 蒸馏：

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `without_feature / baseline` | 47.40 | 56.42 | 56.41 | 50.67 | 28.45 |
| `without_feature / standard_kd` | 49.59 | **58.26** | **58.88** | 53.18 | 30.44 |
| `without_feature / dynamic_gated_kd` | **50.31** | 57.97 | 59.07 | **53.02** | 30.24 |
| `with_feature / baseline` | 47.88 | 56.95 | 57.03 | 51.06 | 28.67 |
| `with_feature / standard_kd` | **50.74** | 58.02 | 58.19 | **53.30** | **31.29** |
| `with_feature / dynamic_gated_kd` | 49.68 | 57.51 | 58.76 | 52.85 | 29.71 |

相对不带 feature 的同名方法：

### 2.1 standard KD

- Macro `+1.15`
- Micro `-0.24`
- Samples `-0.69`
- mAP `+0.12`
- Hard `+0.85`

### 2.2 dynamic gated KD

- Macro `-0.63`
- Micro `-0.46`
- Samples `-0.31`
- mAP `-0.17`
- Hard `-0.52`

---

## 3. 训练诊断

有一个比较明显的现象：

- `dynamic_gated_kd` 的 `feature_loss` 大约稳定在 `5.53`
- 随 epoch 几乎不下降

这说明：

1. 当前 Soft-SupCon 分支确实在反向传播
2. 但它没有形成强的优化驱动力
3. 更像是一个几乎常量的弱正则，而不是显著重塑 feature geometry 的主信号

---

## 4. 结论

这轮结论不算正面：

1. `Soft-SupCon` 没有提升当前主线 `dynamic_gated_kd`
2. 它反而让 `dynamic_gated_kd` 小幅退化
3. 但它对 `standard_kd` 有一点帮助，尤其是：
   - Macro
   - Hard
   - mAP

所以当前最准确的判断是：

> `Feature-level Soft-SupCon` 不是当前 dynamic gated distillation 的有效增益项，  
> 但可能对更纯粹的 `standard KD` 路线有一定 regularization 价值。

---

## 5. 下一步建议

如果继续试 feature 路线，更合理的方向是：

1. 不要直接把 teacher soft similarity 全量施加到所有 student 对
2. 先做更稀疏的 pair selection
   - 例如只保留 top-k 相似 teacher neighbors
3. 或者降低 feature loss 权重
   - 例如 `0.02 / 0.05`
4. 或者只把 feature regularizer 加在 `standard KD`
   - 不强行叠在 `dynamic_gated_kd`

