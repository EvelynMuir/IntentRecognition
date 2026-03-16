# 实验记录：Confusion-Aware Multi-Prototype Learning

## 0. 文档信息

- 日期：`2026-03-13`
- 对应设计：`docs/design/design_0313_03_ConfusionAwareMultiPrototypeLearning.md`
- 主实验输出目录：`logs/analysis/confusion_aware_multi_prototype_20260313`
- 复用缓存目录：`logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache`

---

## 1. main conclusions

1. 本轮按 design 做了一个**最小可行版 CAML-MP**：
   - backbone 不动
   - 直接复用全量 CLIP feature cache 与 baseline logits
   - 在训练侧加入 prototype alignment 与 hard negative margin
   - 测试时仍以分类分数为主，不直接用 prototype score 做 inference
2. 正式跑的对比包括：
   - `baseline + prototype`
   - `baseline + prototype + random hard`
   - `baseline + prototype + confusion hard`
   - `baseline + 2 / 4 / 8 prototypes + confusion hard`
3. 按 validation macro 选模时，best variant 是：
   - `baseline + 2 prototypes + confusion hard`
   - val Macro `55.57`
   - 但 test 只有：
     - Macro `47.73`
     - Hard `29.12`
4. 不论按 validation 选模，还是直接看 test leaderboard，这条线都**没有形成正式正增益**：
   - `baseline` reference：Macro `48.23`, Hard `29.71`
   - `baseline + prototype`：Macro `48.37`, Hard `29.40`
   - `baseline + prototype + confusion hard`：Macro `48.19`, Hard `29.71`
   - `baseline + 2 prototypes + confusion hard`：Macro `47.73`, Hard `29.12`
   - `scenario SLR-C`：Macro `51.28`, Hard `33.98`
5. `confusion-aware` negative 并没有比 `random hard` 更好：
   - `single prototype + random hard`：Macro `48.19`, Hard `29.71`
   - `single prototype + confusion hard`：Macro `48.19`, Hard `29.71`
   当前完全打平，没有拿到“confusion-aware negatives 更有效”的证据。
6. `multi-prototype` 也没有拿到正信号：
   - `2 prototypes` 反而最差
   - `4 prototypes` 与 `single prototype + hard` 几乎完全重合
   - `8 prototypes` 也没有提升
7. prototype usage 并不是全面塌缩，但也没有带来收益：
   - `2 prototypes` 下有的类几乎全压到单一 prototype，例如 `Attractive = 312 / 0`
   - 也有少数类出现分流，例如 `EasyLife = 23 / 47`
   - `4 / 8 prototypes` 分布更分散，但整体指标仍然不涨
8. 因此，对这条线的最保守结论是：

> 当前这版 training-time prototype regularizer **没有证明“embedding 被拉开”是真正瓶颈**，  
> 也没有证明 confusion-aware hard negatives 或 multi-prototype 能稳定改善 visual intent recognition。

---

## 2. 本轮实现

本轮新增：

- `scripts/analyze_confusion_aware_multi_prototype.py`
  - 复用 `train/val/test *_base.npz` 与 `*_clip.npz`
  - 构造单 prototype / 多 prototype 初始化
  - 加入 prototype alignment loss
  - 加入 hard negative margin loss
  - 跑 random vs confusion negative
  - 导出主表与 prototype usage CSV

实现边界：

1. 这是一个 **analysis-side MVP**，不是新的 end-to-end Lightning 主干。
2. backbone 完全冻结。
3. 以 baseline logits 为固定 base，再训练一个小 residual adapter。
4. prototype 分支只在训练时塑形，不直接替代测试分数。

也就是说，这轮是在一个对 prototype 方法**相对友好**的前提下做的最小验证；即便如此仍然没有得到正式增益。

---

## 3. 运行命令

语法检查：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -m py_compile scripts/analyze_confusion_aware_multi_prototype.py
```

主实验：

```bash
source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
python -u scripts/analyze_confusion_aware_multi_prototype.py \
  --device cpu \
  --output-dir logs/analysis/confusion_aware_multi_prototype_20260313
```

固定超参：

- `lr ∈ {1e-3, 5e-4}`
- `weight_decay = 1e-4`
- `lambda_proto = 0.1`
- `lambda_hard = 0.1`
- `margin = 0.2`
- `temperature = 0.1`
- `max_epochs = 40`
- `patience = 8`
- confusion negatives：`topk = 10`, `topn = 3`

---

## 4. 主结果

主表来自：`logs/analysis/confusion_aware_multi_prototype_20260313/main_comparison.csv`

| Method | Macro | Micro | Samples | mAP | Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 48.23 | 52.04 | 52.94 | 50.29 | 29.71 |
| `scenario SLR-C` | **51.28** | **59.13** | **58.47** | **53.66** | **33.98** |
| `fixed benchmark-bank best` | **51.92** | - | - | - | **35.63** |
| `latent basis MVP` | 48.52 | 51.40 | 51.99 | 50.46 | 29.35 |
| `baseline + prototype` | 48.37 | 47.39 | 47.13 | 50.28 | 29.40 |
| `baseline + prototype + random hard` | 48.19 | 51.91 | 52.74 | 50.29 | 29.71 |
| `baseline + prototype + confusion hard` | 48.19 | 51.91 | 52.74 | 50.29 | 29.71 |
| `baseline + 2 prototypes + confusion hard` | 47.73 | 51.78 | 52.56 | 50.26 | 29.12 |
| `baseline + 4 prototypes + confusion hard` | 48.19 | 51.91 | 52.74 | 50.29 | 29.71 |
| `baseline + 8 prototypes + confusion hard` | 48.14 | 51.88 | 52.69 | 50.29 | 29.71 |

### 4.1 最核心的比较

相对 `baseline`：

- `prototype only`
  - Macro `+0.14`
  - Hard `-0.31`
- `prototype + confusion hard`
  - Macro `-0.04`
  - Hard `+0.00`
- `2 prototypes + confusion hard`
  - Macro `-0.50`
  - Hard `-0.59`

相对 `scenario SLR-C`：

- 最好 test 变体也仍然差：
  - Macro 低约 `3.0`
  - Hard 低约 `4.27`

结论很明确：

> prototype regularization 并没有把当前 baseline 推到接近 `SLR-C` 的区间。

### 4.2 validation 与 test 的不一致

按 validation macro 选模时，`2 prototypes + confusion hard` 是 best：

- val Macro `55.57`
- val Hard `42.92`

但 test 却掉到：

- Macro `47.73`
- Hard `29.12`

这说明：

1. 这条线当前稳定性不够
2. prototype 相关选择更容易在 val 上产生乐观假象
3. 不能把单次 val 好看当作方法成立

---

## 5. hard negative 消融

固定 single prototype：

| Hard Negative | Macro | Hard |
| --- | ---: | ---: |
| no hard negative | 48.37 | 29.40 |
| random negative | 48.19 | 29.71 |
| confusion negative | 48.19 | 29.71 |

解释：

1. 加 hard negative 后，Hard 的确能从 `29.40` 回到 `29.71`
2. 但 `random` 和 `confusion` 完全打平
3. 所以当前还不能说“confusion-aware negative 比普通 hard negative 更有效”

这意味着 design 里的核心 novelty 点：

> confusion-aware contrastive

在这一轮并没有被真正验证出来。

---

## 6. prototype 数量消融

固定 confusion hard negative：

| Prototypes / Class | Macro | Hard |
| --- | ---: | ---: |
| 1 | 48.19 | **29.71** |
| 2 | 47.73 | 29.12 |
| 4 | 48.19 | **29.71** |
| 8 | 48.14 | **29.71** |

解释：

1. `2 prototypes` 明显更差
2. `4 / 8 prototypes` 没有比 single prototype 更好
3. 当前没有证据支持“类内多模态是真正主瓶颈”

换句话说，design 的第三个关键问题：

> multi-prototype 是否必要？

这一轮答案仍然偏向 **no strong evidence**。

---

## 7. prototype usage

使用统计文件：

- `logs/analysis/confusion_aware_multi_prototype_20260313/prototype_usage_2.csv`
- `logs/analysis/confusion_aware_multi_prototype_20260313/prototype_usage_4.csv`
- `logs/analysis/confusion_aware_multi_prototype_20260313/prototype_usage_8.csv`

### 7.1 `K=2`

能看到明显不均匀：

- `Attractive = 312 / 0`
- `CuriousAdventurousExcitingLife = 131 / 0`
- `FineDesignLearnArt-Arch = 91 / 0`

也有部分分流：

- `EasyLife = 23 / 47`
- `EnjoyLife = 94 / 185`

### 7.2 `K=4 / 8`

分配更均匀，但没有变成性能提升。

这说明问题不是“所有 prototype 都塌缩到 1 个”这么简单，而更像是：

1. prototype usage 本身并非完全失败
2. 但这些被分出来的 visual modes，并没有转成更好的判别边界
3. 也没有帮助 hardest confusion

---

## 8. 结论与下一步

### 8.1 这轮如何定性

这轮应当被定为 **negative result**：

1. single prototype 没有正式超过 baseline
2. confusion-aware hard negative 没有优于 random hard negative
3. multi-prototype 没有带来收益
4. validation-best 与 test 不一致，稳定性不足

### 8.2 对 design 的判断

对 `design_0313_03_ConfusionAwareMultiPrototypeLearning.md` 当前应写成：

> 在当前 frozen-feature MVP 下，prototype alignment 与 confusion-aware hard negative  
> **没有证明“embedding 拉开”是当前主要瓶颈**，  
> 也没有证明 multi-prototype 能稳定改善 visual intent recognition。

### 8.3 是否继续深挖

按 design 的止损标准，这条线目前不值得继续深挖，理由是：

1. single prototype 没有明确正信号
2. confusion-aware hard negative 没有区别于 random negative
3. multi-prototype 没有带来 Hard / Macro 提升

如果后续还想保留 prototype 思路，至少需要非常明确地改变问题设定，例如：

1. prototype 直接进入 candidate-local inference，而不是只做 regularizer
2. 用更强的 sample-specific false positive negatives，而不是固定 class neighborhoods
3. 在 raw image / end-to-end head 上重做，而不是 cached-feature residual 近似

但以当前结果看，更保守的工程判断是：

> 这条线的优先级应低于当前已经有明确正信号的 `SLR-C / comparative verifier / fixed bank` 主线。
