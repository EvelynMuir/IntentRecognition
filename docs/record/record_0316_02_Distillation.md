# 实验记录：Privileged Distillation

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0316_02_Distillation.md`
- 主实验输出目录：`logs/analysis/privileged_distillation_full_20260316`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`
- 文本特征目录：`logs/analysis/vlm_full_20260316`

---

## 1. main conclusions

1. 已按设计实现一版完整的 `privileged distillation` cache-based prototype：
   - teacher：`CLIP image feature + BGE rationale feature -> MLP -> multi-label logits`
   - student：`CLIP image feature -> MLP`
   - loss:
     - baseline：`ASL(binary label)`
     - standard KD：`ASL + lambda * Bernoulli-KL`
     - dynamic gated KD：`omega * ASL + (1 - omega) * lambda * Bernoulli-KL`
2. 本轮 teacher 没有成为真正的 `oracle upper bound`：
   - teacher：Macro `45.87`, Hard `28.08`
   - baseline：Macro `47.49`, Hard `27.50`
   也就是说：
   - text-enhanced teacher 没有超过 image-only baseline
   - 这和设计里的预期不一致，当前 teacher 不能作为“绝对纠错标尺”
3. 尽管 teacher 不是 upper bound，蒸馏本身仍然带来了稳定收益：
   - baseline：Macro `47.49`, Micro `56.20`, Samples `56.61`, Hard `27.50`, mAP `50.91`
   - standard KD：Macro `48.96`, Micro `57.44`, Samples `57.76`, Hard `30.46`, mAP `52.22`
   - dynamic gated KD：Macro `49.41`, Micro `57.38`, Samples `58.29`, Hard `29.83`, mAP `52.58`
4. 这一版结果的最稳妥结论是：

> `teacher-side privileged signal` 是有用的，  
> 但当前 teacher 本身还不够强，不足以支撑“teacher 纠错上限”这条主 Story。  
> 更合理的表述是：`teacher-guided regularization / distillation` 能提升 frozen-CLIP baseline，  
> 其中 `dynamic gated KD` 在 Macro / Samples / mAP 上最好，`standard KD` 在 Hard 上最好。

5. agreement 切片分析只能落在 `train`：
   - 公开 `val/test` 标注文件不带 `softprob`
   - 现有 `val/test cache` 里的 `soft_labels` 也是二值标签
   - 因此设计文档里的“按 test agreement 切片”当前无法原样复现

---

## 2. 本轮实现

新增：

- `scripts/analyze_privileged_distillation.py`
  - 复用 frozen `CLIP ViT-L/14` image cache
  - 读取 `BGE rationale feature`
  - 训练 `teacher / baseline / standard KD / dynamic gated KD`
  - 输出：
    - `main_comparison.csv`
    - `summary.json`
    - `agreement_slice_analysis.csv`
    - `teacher_rejects_uncertain_positive.csv`
    - `teacher_adds_new_label.csv`

实现细节：

1. teacher 输入：
   - `train_clip["features"]`，维度 `768`
   - `rationale_full_bge_features["features"]`，维度 `1024`
2. student 输入：
   - 仅 `train_clip["features"]`
3. agreement gate：
   - 训练时用正类 softprob 的 `min` 作为样本 gate
   - 即如果一个样本任一正类只有 `1/3` 一致性，则该样本整体更倾向蒸馏分支
4. KD loss：
   - 使用 multi-label `Bernoulli KL`
   - teacher prob 使用 temperature=`2.0`

---

## 3. 运行命令

### 3.1 语法检查

```bash
python -m py_compile scripts/analyze_privileged_distillation.py
```

### 3.2 smoke test

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --max-test-samples 128 \
  --max-epochs 2 \
  --patience 2 \
  --batch-size 128 \
  --output-dir logs/analysis/privileged_distillation_smoke_20260316
```

### 3.3 full run

```bash
/home/evelynmuir/lambda/projects/IntentRecognition/.conda/bin/python -u \
  scripts/analyze_privileged_distillation.py \
  --max-epochs 15 \
  --patience 4 \
  --batch-size 256 \
  --student-agreement-pool min \
  --output-dir logs/analysis/privileged_distillation_full_20260316
```

---

## 4. 主结果

主表来自：

- `logs/analysis/privileged_distillation_full_20260316/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_teacher` | 45.87 | 53.44 | 54.04 | 48.25 | 28.08 |
| `baseline` | 47.49 | 56.20 | 56.61 | 50.91 | 27.50 |
| `standard_kd` | 48.96 | **57.44** | 57.76 | 52.22 | **30.46** |
| `dynamic_gated_kd` | **49.41** | 57.38 | **58.29** | **52.58** | 29.83 |

相对 baseline：

- `standard KD`
  - Macro `+1.47`
  - Micro `+1.24`
  - Samples `+1.16`
  - Hard `+2.96`
  - mAP `+1.31`
- `dynamic gated KD`
  - Macro `+1.92`
  - Micro `+1.18`
  - Samples `+1.68`
  - Hard `+2.33`
  - mAP `+1.67`

解释：

1. 蒸馏整体是成立的
2. `dynamic gated KD` 没有在 `Hard` 上超过 `standard KD`
3. 但 `dynamic gated KD` 在：
   - Macro
   - Samples
   - mAP
   三项上最好，因此更像“全局更稳的版本”

---

## 5. Teacher 诊断

teacher 最佳点在：

- `epoch 1`

结果：

- Teacher：Macro `45.87`, Hard `28.08`
- Baseline：Macro `47.49`, Hard `27.50`

这意味着：

1. teacher 的 `Hard` 略高于 baseline
2. 但整体 Macro 反而更低
3. 所以 teacher 不能被当作真正的 `oracle upper bound`

一个更可信的原因是：

1. train teacher 使用的是 `full rationale`
2. val/test teacher 使用的是 `baseline-pred rationale`
3. teacher train-test 间存在明显文本分布漂移
4. 在 frozen global CLIP feature 这条设定下，image-only MLP baseline 本身已经很强

换句话说：

> 当前 teacher 更像“带文本先验的辅助教师”，  
> 不是“显著强于视觉模型的上帝视角老师”。

---

## 6. Agreement Slice Analysis

切片表来自：

- `logs/analysis/privileged_distillation_full_20260316/agreement_slice_analysis.csv`

限制：

1. 当前只能做 `train` 切片
2. 原因是公开 `val/test` 无 `softprob`

### 6.1 train agreement = 1/3

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 32.20 | 36.84 | 36.62 | 35.42 | **24.59** |
| `standard_kd` | 32.15 | 37.35 | 36.85 | **37.21** | 23.37 |
| `dynamic_gated_kd` | **32.77** | **37.67** | **37.56** | 35.94 | 23.63 |

解读：

1. 在最低一致性子集上，`dynamic gated KD` 的 Macro / Micro / Samples 都超过 baseline
2. 这说明 gate 至少没有把 noisy slice 搞坏，且确实拉起了整体判别质量
3. 但这个切片里的 `Hard` 仍然没有超过 baseline，所以“噪声鲁棒”故事还不够强

---

## 7. Teacher Correction Candidates

候选导出：

- `logs/analysis/privileged_distillation_full_20260316/teacher_rejects_uncertain_positive.csv`
- `logs/analysis/privileged_distillation_full_20260316/teacher_adds_new_label.csv`

用途：

1. 自动筛出 `1/3 agreement` 样本
2. 优先导出：
   - teacher 强烈否定低一致性正类
   - teacher 强烈补充新标签
3. 后续可以人工挑几张明显错标图进论文图表

注意：

- 这些文件是“候选池”，不是已经人工审核过的最终图表

---

## 8. 当前结论与下一步

当前最合理的结论是：

1. `Privileged Distillation` 值得保留
2. 但当前最强主线不是“teacher upper bound + gated distillation”
3. 更稳的主线应改写成：

> `rationale-enhanced teacher signal can regularize a frozen visual student,`  
> `yielding consistent gains over baseline;`  
> `dynamic gating further improves macro-level robustness, though the current teacher is not a true oracle upper bound.`

如果要继续把它推成论文主线，下一步优先级应是：

1. 先把 teacher 做强
   - val/test 也生成 image-only full rationale，而不是 `baseline-pred rationale`
   - 或直接训练/评估一个更干净的 multimodal teacher
2. 拿到带 agreement 的 val/test 标注
   - 否则最关键的 noisy-slice 证据链永远不完整
3. 再比较：
   - stronger teacher
   - standard KD
   - dynamic gated KD

