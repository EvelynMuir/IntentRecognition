# 实验记录：VLM Late Fusion

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0316_01_VLMLateFusion.md`
- 主实验输出目录：`logs/analysis/vlm_late_fusion_full_20260316`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. main conclusions

1. 本轮严格按 `VLMLateFusion` 设计，保持视觉 Backbone 与 baseline 完全一致，不再做 teacher-side alignment，只在 logit 层引入文本 residual。
2. 文本来源设置为：
   - train：full rationale，`y_true` 用 GT 多标签，`y_confuse` 用 baseline false positive
   - val/test：`baseline_pred rationale`，不使用 GT，避免泄漏
3. 文本编码器使用：
   - `BAAI/bge-large-en-v1.5`
4. full run 的 test `class-wise` 指标为：
   - Macro `48.11`
   - Micro `48.32`
   - Samples `48.12`
   - mAP `50.26`
   - Hard `27.83`
5. 这个结果没有超过 baseline，且在 `Hard` 上明显更差：
   - baseline：Macro `48.23`, Hard `29.71`
   - late fusion：Macro `48.11`, Hard `27.83`
   也就是：
   - Macro `-0.12`
   - Hard `-1.88`
6. learned alpha 的走势也很清楚：
   - `alpha_pos` 从 `0.10` 增长到约 `0.17`
   - `alpha_ctx` 从 `0.10` 逐渐降到接近 `0`，甚至转负
   - `alpha_neg` 从 `0.10` 增长到约 `0.31`
   这说明：
   - Step 1 的正向视觉证据有一定弱信号
   - Step 2 的上下文逻辑没有被模型稳定接纳
   - Step 3 的反向惩罚被放大了，但并没有转化为 Hard 改善
7. 因此，这轮可以比较稳地定性为：

> `VLMLateFusion` 在当前 cached-feature / decision-layer residual 设定下不成立为主线方法。  
> 文本 residual 没有比 baseline 更好，且 hardest classes 被明显伤害。

---

## 2. 本轮实现

本轮新增或完善：

- `scripts/generate_vlm_rationales.py`
  - 支持 `label_source=baseline_pred`
  - 支持分批 `request_batch_size`
  - 支持即时写盘，便于全量续跑
- `scripts/extract_vlm_rationale_features.py`
  - 支持 `BGE` 编码
  - 支持导出 `step1/step2/step3` 文本与特征
- `scripts/analyze_vlm_late_fusion.py`
  - 使用 `W_proj` 将视觉特征投影到文本空间
  - 在 base logits 上直接加 `+ alpha_pos * S_pos + alpha_ctx * S_ctx - alpha_neg * S_neg`

本轮没有改：

1. 视觉 Backbone
2. baseline logits 来源
3. 基础分类损失

所以这是一个很纯粹的“decision-layer text residual”验证。

---

## 3. 数据与文本准备

### 3.1 train

train rationale 文件：

- `logs/analysis/vlm_full_20260316/rationale_full.jsonl`

规模：

- `12739` 条

### 3.2 val

val baseline-pred rationale：

- `logs/analysis/vlm_full_20260316/val_rationale_baseline_pred.jsonl`

规模：

- `498` 条

### 3.3 test

test baseline-pred rationale：

- `logs/analysis/vlm_full_20260316/test_rationale_baseline_pred.jsonl`

规模：

- `1216` 条

### 3.4 文本特征

- `logs/analysis/vlm_full_20260316/rationale_full_bge_features.npz`
- `logs/analysis/vlm_full_20260316/val_rationale_baseline_pred_bge_features.npz`
- `logs/analysis/vlm_full_20260316/test_rationale_baseline_pred_bge_features.npz`

---

## 4. 运行命令

### 4.1 语法检查

```bash
python -m py_compile \
  scripts/generate_vlm_rationales.py \
  scripts/extract_vlm_rationale_features.py \
  scripts/analyze_vlm_late_fusion.py
```

### 4.2 full-train rationale

```bash
python -u scripts/generate_vlm_rationales.py \
  --output-jsonl logs/analysis/vlm_full_20260316/rationale_full.jsonl \
  --mode rationale \
  --temperature 0.0 \
  --max-tokens 768 \
  --max-image-size 336 \
  --request-batch-size 256
```

### 4.3 val/test baseline-pred rationale

```bash
python -u scripts/generate_vlm_rationales.py \
  --base-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --annotation-file /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/annotation/intentonomy_val2020.json \
  --image-dir /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/images/low \
  --label-source baseline_pred \
  --threshold-source-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --output-jsonl logs/analysis/vlm_full_20260316/val_rationale_baseline_pred.jsonl \
  --mode rationale
```

```bash
python -u scripts/generate_vlm_rationales.py \
  --base-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/test_base.npz \
  --annotation-file /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/annotation/intentonomy_test2020.json \
  --image-dir /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/images/low \
  --label-source baseline_pred \
  --threshold-source-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --output-jsonl logs/analysis/vlm_full_20260316/test_rationale_baseline_pred.jsonl \
  --mode rationale
```

### 4.4 BGE 文本特征提取

```bash
python -u scripts/extract_vlm_rationale_features.py \
  --input-jsonl logs/analysis/vlm_full_20260316/rationale_full.jsonl \
  --output-npz logs/analysis/vlm_full_20260316/rationale_full_bge_features.npz \
  --text-encoder bge \
  --hf-model-name BAAI/bge-large-en-v1.5 \
  --hf-cache-dir /home/evelynmuir/lambda/hf-models
```

### 4.5 late fusion 主实验

```bash
python -u scripts/analyze_vlm_late_fusion.py \
  --reuse-cache-dir logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache \
  --train-text-npz logs/analysis/vlm_full_20260316/rationale_full_bge_features.npz \
  --val-text-npz logs/analysis/vlm_full_20260316/val_rationale_baseline_pred_bge_features.npz \
  --test-text-npz logs/analysis/vlm_full_20260316/test_rationale_baseline_pred_bge_features.npz \
  --annotation-file /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/annotation/intentonomy_train2020.json \
  --output-dir logs/analysis/vlm_late_fusion_full_20260316 \
  --device cuda \
  --max-epochs 10 \
  --patience 3 \
  --batch-size 256
```

---

## 5. 主结果

主表来自：

- `logs/analysis/vlm_late_fusion_full_20260316/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 48.23 | 52.04 | 52.94 | 50.29 | **29.71** |
| `VLM late fusion` | 48.11 | 48.32 | 48.12 | 50.26 | 27.83 |

### 5.1 直接比较

相对 baseline：

- Macro `-0.12`
- Micro `-3.72`
- Samples `-4.81`
- mAP `-0.03`
- Hard `-1.88`

解释：

1. late fusion 没有带来正增益
2. Hard 明显下降
3. 不是“平均指标略差但 hardest cases 更强”的形态，而是整体和 hardest 都更差

---

## 6. alpha 诊断

训练历史来自：

- `logs/analysis/vlm_late_fusion_full_20260316/summary.json`

learned alpha 变化：

- epoch 1
  - `alpha_pos = 0.120`
  - `alpha_ctx = 0.106`
  - `alpha_neg = 0.127`
- epoch 4（validation-best）
  - `alpha_pos = 0.167`
  - `alpha_ctx = 0.083`
  - `alpha_neg = 0.216`
- epoch 7
  - `alpha_pos = 0.114`
  - `alpha_ctx = -0.042`
  - `alpha_neg = 0.311`

这组数值非常有解释力：

1. `alpha_pos` 被保留下来，说明视觉证据文本有一点弱正信号
2. `alpha_ctx` 很快衰减并转负，说明 Step 2 的上下文推理不稳定，模型并不真正信任它
3. `alpha_neg` 持续增大，但 Hard 仍下降，说明 Step 3 的反向惩罚在当前形式下更像噪声放大器，而不是有效的 confusion suppressor

---

## 7. 结论

这轮 `VLMLateFusion` 的结论比较明确：

1. 保持 Backbone 不变、只在 logit 层做文本 residual，这条路线在当前设定下不成立。
2. `Step 1` 也许有一点弱正信号，但不够支撑整体 late fusion。
3. `Step 2 / Step 3` 进入 residual 后，最终更像是在扰动稳定的 baseline 决策，而不是做有效修正。
4. full-train 结果已经说明：问题不再是 sample size 不够，而是 late fusion 这个机制本身没有抓住有效耦合方式。

因此，对 `docs/design/design_0316_01_VLMLateFusion.md` 当前最保守、最合理的判断是：

> `VLMLateFusion` 不建议继续作为主线方法深入。  
> 如果后续还要保留 VLM 信息，更合理的方向是更局部的 candidate-level guidance，而不是全局的 logit residual。
