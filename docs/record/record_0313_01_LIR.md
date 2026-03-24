# 实验记录：LIR Phase 1 - Latent Intent Basis MVP

## 0. 文档信息

- 日期：`2026-03-13`
- 对应设计：`docs/design/design_0313_01_LIR.md`
- 主实验输出目录：`logs/analysis/lir_phase1_full_20260313`
- 基础缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`
- region cache：`logs/analysis/smoke_region_grounded_agent_20260312/_region_cache`

---

## 1. main conclusions

1. 本轮把 `design_0313_01_LIR.md` 的 `Phase 1` 做成了一个**正式 full-data MVP**：
   - 使用完整 train split `12739` 张图，而不是之前不完整的 `1024-sample pilot`
   - 抽取完整 train patch tokens
   - 训练 `K=32, top-4 routing` 的 latent basis residual
2. 正式 full-run 的 test `class-wise` 结果为：
   - Macro `48.52`
   - Micro `51.40`
   - Samples `51.99`
   - mAP `50.46`
   - Hard `29.35`
3. 这个结果**没有超过**最强固定参考，也没有正式超过 baseline：
   - relative to `baseline`：Macro `+0.29`，Hard `-0.36`
   - relative to `scenario SLR-C`：Macro `-2.76`，Hard `-4.63`
   - relative to `fixed benchmark-bank best`：Macro `-3.39`，Hard `-6.28`
   - relative to `comparative verifier best`：Macro `-3.21`，Hard `-6.48`
4. 与同日的 `1024-sample pilot` 相比，full-data 并没有带来质变：
   - pilot：Macro `48.43`, Hard `30.05`
   - full run：Macro `48.52`, Hard `29.35`
   也就是：样本数补齐后只有 `+0.09` Macro，`Hard` 反而下降 `-0.70`。
5. 最好 epoch 出现在 **epoch 1**，之后验证和测试指标都没有继续改善，说明当前 latent basis residual 的优化信号并不稳定。
6. basis usage 出现了明显集中：
   - top-5 basis (`1/31/6/4/13`) 合计占据约 `88.28%` 的总路由质量
   - basis `7` 和 `29` 基本完全未被使用
   - 多个类别共享几乎同一组 top-4 basis
   这已经接近 design 中提到的 `basis collapse / no stable structure` 停止条件。
7. 因此，这一轮 `Phase 1` 的正式结论是：

> 在当前“frozen baseline logits + latent basis residual”这条最小实现路径上，  
> learnable latent basis **没有证明自己优于 fixed bank，也没有形成 stable hard-case gain**。  
> 按 design 的停止条件，这条具体 MVP 不应直接进入 `Phase 2 hierarchy residual` 或 `Phase 3 soft label refinement`。

---

## 2. 本轮实现

本轮新增：

- `scripts/analyze_lir_phase1.py`
  - 复用现有 `train/val/test *_base.npz` 与 `*_clip.npz`
  - 复用现有 `val/test` region patch cache
  - 自动抽取完整 `train` split 的 CLIP projected patch tokens
  - 训练最小 latent basis residual
  - 导出 `summary.json`、`training_history.json`、`basis_usage.json`、`phase1_comparison.csv`、`latent_basis_best.pt`

本轮实现边界：

1. 这是一个 **analysis-side MVP**，不是新的 end-to-end Lightning 训练主干。
2. global branch 直接复用 frozen baseline logits。
3. latent basis branch 只训练 residual adapter：
   - learnable basis queries
   - patch-token cross-attention
   - top-k sparse routing
   - residual head
4. 本轮**没有**进入：
   - hierarchy residual
   - teacher soft target
   - graph / neighbor refinement

这样做的理由是：design 明确要求先回答最基础的问题，

> `latent basis > fixed bank` 是否成立。

而本轮答案是否定的。

---

## 3. 运行命令

语法检查：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -m py_compile scripts/analyze_lir_phase1.py
```

先抽取完整 train patch cache：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -u scripts/analyze_lir_phase1.py \
  --output-dir logs/analysis/lir_phase1_full_20260313 \
  --device cuda \
  --extract-batch-size 64 \
  --extract-num-workers 4 \
  --train-num-workers 4 \
  --train-patch-only
```

正式 full-run：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -u scripts/analyze_lir_phase1.py \
  --output-dir logs/analysis/lir_phase1_full_20260313 \
  --device cuda \
  --extract-batch-size 64 \
  --extract-num-workers 4 \
  --train-batch-size 128 \
  --eval-batch-size 256 \
  --train-num-workers 4
```

---

## 4. 固定设置

### 4.1 数据与缓存

- full train samples：`12739`
- val samples：`498`
- test samples：`1216`
- train patch cache：完整重抽，之后训练阶段 `reused`

### 4.2 模型

- input dim：`768`
- hidden dim：`256`
- `num_basis = 32`
- `routing_topk = 4`
- `dropout = 0.1`

### 4.3 训练

- optimizer：`AdamW`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- `sparse_weight = 5e-4`
- `diversity_weight = 1e-2`
- `train_batch_size = 128`
- `max_epochs = 8`
- `patience = 4`
- `clip_grad_norm = 1.0`

### 4.4 评价方式

- 主报告指标来自 `class-wise threshold`
- 同时保留 `global threshold` 作为附属诊断

---

## 5. 主结果

主表来自：`logs/analysis/lir_phase1_full_20260313/phase1_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR-C` | 51.28 | 59.13 | 58.47 | 53.66 | 33.98 |
| `fixed benchmark-bank best` | **51.92** | - | - | - | 35.63 |
| `comparative verifier best` | 51.73 | **59.22** | **58.19** | **53.78** | **35.83** |
| `latent basis MVP (full train)` | 48.52 | 51.40 | 51.99 | 50.46 | 29.35 |

### 5.1 直接解读

1. `latent basis MVP` 没有形成设计里期望的 `latent basis > fixed bank`。
2. 它甚至没有在 `Hard` 上超过 baseline。
3. 因此，本轮不能把“fixed bank 失败是坐标系问题”作为正式被验证的结论。

### 5.2 与 baseline 的关系

相对 `baseline`：

- Macro `+0.29`
- Micro `-0.64`
- Samples `-0.94`
- mAP `+0.17`
- Hard `-0.36`

这说明当前 latent basis residual 并没有形成一致收益，只是在个别阈值定义下拉起了少量 macro。

### 5.3 与 1024-sample pilot 的关系

pilot 主结果：

- Macro `48.43`
- Hard `30.05`

full run：

- Macro `48.52`
- Hard `29.35`

结论：

> 之前 pilot 的负结果并不是“只因为 train 数据不完整”。  
> 把 train 补齐到 `12739` 后，结果仍然维持在同一失败区间。

---

## 6. 训练动态

`logs/analysis/lir_phase1_full_20260313/training_history.json`

| Epoch | Val Macro (class-wise) | Test Macro (class-wise) | Test Hard (class-wise) |
| --- | ---: | ---: | ---: |
| 1 | **54.94** | **48.52** | 29.35 |
| 2 | 54.37 | 47.91 | 29.75 |
| 3 | 54.42 | 47.91 | 29.37 |
| 4 | 53.66 | 46.61 | 28.00 |
| 5 | 53.47 | 47.76 | **29.81** |

观察：

1. best validation macro 出现在第 1 个 epoch。
2. 后续训练 loss 持续下降，但主指标没有同步上升。
3. 这更像是：
   - residual head 在继续拟合
   - latent basis 结构本身没有提供稳定有效的判别增益

---

## 7. Basis 诊断

`logs/analysis/lir_phase1_full_20260313/basis_usage.json`

### 7.1 全局使用情况

- `avg_active_basis = 4.0`
  - 这与 `top-4 routing` 设计一致，本身信息量有限
- top basis mass：
  - basis `1`：`0.2355`
  - basis `31`：`0.1778`
  - basis `6`：`0.1659`
  - basis `4`：`0.1554`
  - basis `13`：`0.1482`

前五个 basis 总质量约为：

- `0.2355 + 0.1778 + 0.1659 + 0.1554 + 0.1482 = 0.8828`

即约 `88.28%`。

同时：

- basis `7`：`0.0`
- basis `29`：`0.0`

### 7.2 按类别看

多数类别的 top-4 basis 都高度重合，例如：

- class `0`：`1 / 4 / 6 / 31`
- class `2`：`1 / 31 / 6 / 4`
- class `3`：`1 / 4 / 6 / 31`
- class `5`：`1 / 31 / 6 / 4`

这说明：

1. basis 没有学出明显的 class-specific reuse pattern
2. 也没有形成更丰富的 sparse decomposition
3. 更像是收缩到少量共享 basis 的低秩近似

这与 design 的负向停止条件高度一致：

> 稀疏 basis 学不出稳定结构，且出现明显塌缩。

---

## 8. 结论与下一步

### 8.1 正式结论

这轮 `Phase 1` 应记为 **negative result**：

1. full-data latent basis MVP 没有超过 fixed bank
2. 没有超过 strongest verifier path
3. 没有形成稳定 hard-case 增益
4. basis usage 还出现了明显集中与部分 basis 完全闲置

因此，按 `design_0313_01_LIR.md` 的停止条件：

> 当前这一版 LIR MVP 不建议直接继续堆 `hierarchy residual` 或 `soft label refinement`。

### 8.2 更合理的后续方向

如果还想保留 “latent basis” 这条线，更合理的改法应该至少满足下面之一：

1. 不再只是 `frozen baseline logits + residual`，而是把 global head 与 basis head 一起重训
2. 重新设计 routing，使它不只是 top-k after attention，而是带更强的 load-balancing / usage regularization
3. 明确比较 `basis-only / global-only / global+basis`，先确认 residual 结构本身是不是瓶颈

否则，更保守的工程判断是：

> 当前 repo 里已经验证过的 `fixed bank / comparative verifier / SLR-C` 主线，  
> 仍然比这条 Phase 1 latent basis MVP 更可靠。
