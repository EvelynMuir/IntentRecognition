# 2026-03-09 Learnable Fusion 结果记录

## 1. 目标

本次实验对应：

- `docs/design/design_0309_LearnableFusion.md`

目标是验证：

> 在当前最强的 local rerank 基础上，引入一个 very-light learnable fusion，是否能进一步优于 plain rerank？

这里明确不做的事：

- 不动 CLIP backbone
- 不动 baseline head
- 不做 class-wise / uncertainty gate
- 不引入大模型或复杂训练分支

要做的事只有一件：

> 在 baseline `top-k` 候选内部，学习一个非常小的局部融合模块，把 baseline logit 和 text prior score 融成新的 logit。

---

## 2. 实验设置

### 2.1 固定 baseline

checkpoint 固定为：

- `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

统一协议下 baseline test：

- `macro = 0.4598`
- `micro = 0.5654`
- `samples = 0.5640`
- `hard = 0.2686`

### 2.2 比较基准

对每个 text source，都先构造一个 source-matched plain rerank 基准：

```text
z'_c = z_c + 0.3 * s_hat_c,   if c in TopK(z), k = 10
```

其中：

- `s_hat_c` 为 normalized text prior
- `topk = 10`
- `alpha = 0.3`

### 2.3 Learnable Fusion 方案

本次实际跑了两类：

#### A. Class-wise affine fusion

```text
z'_c = z_c + a_c * s_c + b_c
```

约束：

- 只在 `top-k` 内启用
- `a_c` 初始化为 `0.3`
- `b_c` 初始化为 `0`
- 正则：
  - `lambda_a = 1e-2`
  - `lambda_b = 1e-3`

#### B. Shared tiny MLP fusion

```text
delta_c = f([z_c, s_c])
z'_c = z_c + delta_c
```

设置：

- 输入特征只用 `[z_c, s_c]`
- hidden dim = `8`
- 最后一层零初始化
- 只在 `top-k` 内启用
- `delta` L2 regularization = `1e-4`

### 2.4 Source 与超参

text source:

- `short`
- `detailed`
- `mixed`

learning rate:

- `1e-3`
- `3e-4`

训练设置：

- `epochs = 20`
- `patience = 5`
- `batch_size = 256`

输出目录：

- `logs/analysis/full_learnable_local_fusion_v2_20260309`

---

## 3. Source-matched plain rerank 基准结果

这是 learnable fusion 必须要超过的直接基准。

| Source | test macro | test micro | test samples | test hard |
| --- | ---: | ---: | ---: | ---: |
| `short` | 0.4947 | 0.5756 | 0.5723 | 0.3276 |
| `detailed` | 0.4891 | 0.5660 | 0.5704 | 0.3198 |
| `mixed` | 0.5061 | 0.5887 | 0.5897 | 0.3529 |

直接结论：

- `mixed` 的 plain rerank 仍然是当前最强的 source-matched 基准
- learnable fusion 如果不能超过这些数字，就说明“学习一个 tiny fusion”没有带来实质价值

---

## 4. Learnable Fusion 主结果

### 4.1 按 validation macro 选最优

best overall by `val_macro`:

- `source = detailed`
- `fusion = classwise_affine`
- `lr = 3e-4`
- `best_epoch = 0`

对应 test：

- `macro = 0.4870`
- `micro = 0.5690`
- `samples = 0.5706`
- `hard = 0.3147`

相对 baseline：

- `macro +0.0272`
- `hard +0.0462`

但相对 source-matched plain rerank：

- `macro -0.0020`
- `hard -0.0051`

也就是说：

> 它比 baseline 强，但没有超过同 source 的 plain rerank。

### 4.2 按 source 看最优 learnable fusion

| Source | Best learned config | test macro | test hard | 相对 plain source rerank |
| --- | --- | ---: | ---: | --- |
| `short` | `classwise_affine, lr=3e-4` | 0.4932 | 0.3296 | `macro -0.0016`, `hard +0.0020` |
| `detailed` | `classwise_affine, lr=3e-4` | 0.4870 | 0.3147 | `macro -0.0020`, `hard -0.0051` |
| `mixed` | `classwise_affine, lr=3e-4` | 0.5050 | 0.3513 | `macro -0.0011`, `hard -0.0016` |

### 4.3 排行榜现象

top 排行里有一个非常清晰的模式：

- 前 6 名全部是 `classwise_affine`
- `shared_mlp` 明显落后

说明：

> 如果一定要做 learnable fusion，affine 比 tiny shared MLP 更靠谱。

但更关键的是：

- 即使是最好的 affine，也没有稳定超过 plain rerank
- 最好的结果往往出现在 `best_epoch = 0`

这意味着：

> 学习本身没有带来额外收益，反而很容易把已经有效的 heuristics 学坏。

---

## 5. 失败模式分析

### 5.1 最关键现象：best epoch 基本落在 0

这不是偶然，而是全局模式。

它说明：

- 初始化到 `alpha_0 = 0.3` 的 affine 版本，本质上已经非常接近当前最好解
- 训练更新一旦开始，通常不会继续提升
- 最终选出来的最好状态，往往就是“刚开始几乎没学坏”的状态

这和当前整体结论是一致的：

> 目前 bottleneck 更像是 `text source / prompt composition`，而不是 `fusion function` 本身。

### 5.2 Hard-case 分析

best-overall learnable fusion 相对 source-matched plain rerank：

- `plain_wrong_fusion_right_count = 0`
- `plain_right_fusion_wrong_count = 22`

也就是说：

- 没有新增真正修正的 hard case
- 反而新增了一批 degraded case

### 5.3 预测行为变化

相对 plain source rerank，best-overall learnable fusion：

- empty predictions 从 `54` 增加到 `72`
- `empty_prediction_delta = +18`

同时：

- 没有新增 recovered label
- 只有 dropped true labels

最常被进一步压掉的真值类包括：

- `CuriousAdventurousExcitingLife`
- `NatBeauty`
- `EnjoyLife`
- `Attractive`
- `Happy`
- `SocialLifeFriendship`

这说明 learnable fusion 当前最主要的问题不是“过度乐观”，而是：

> 它在进一步压缩预测分布，把原本还算合理的候选直接压没了。

换句话说，它更像一个 **over-shrinking module**，而不是更聪明的 reranker。

---

## 6. 对设计文档的反馈

结合 `docs/design/design_0309_LearnableFusion.md`，这轮实验能明确回答几个问题。

### 6.1 Learnable affine fusion 能不能超过 plain rerank？

当前答案是：

> 不能稳定超过。

它能超过 baseline，但不能稳定超过 source-matched plain rerank。

### 6.2 Shared tiny MLP fusion 值不值得继续？

当前答案是：

> 不值得优先继续。

原因：

- 始终落后于 affine
- 明显落后于 plain rerank
- 没有显示出任何独立优势

### 6.3 Learnable fusion 的主要问题是什么？

当前看不是 capacity 不够，而是训练目标与当前问题结构不匹配。

具体表现为：

- 它没有学会“更好地利用 text prior”
- 它主要学会了“更保守地压缩预测”
- 因此更容易丢掉正类

---

## 7. 结论

这轮实验的最终结论很明确：

1. `Learnable fusion` 不是当前最值得推进的方向。
2. `Class-wise affine` 明显优于 `shared MLP`，但也没有超过 plain rerank。
3. 当前最强的东西仍然是：
   - 更好的 `text source`
   - 更好的 `prompt composition`
   - 更强的 plain local rerank
4. 如果还要继续做 learnable fusion，前提应该是先解决“训练后整体收缩、空预测变多”的问题。

因此，这一轮之后的建议是：

- 暂缓 learnable fusion 作为主线
- 回到 `mixed prompt / discriminative prompt / semantic prior construction`
- 只有当 prompt/source 已经稳定后，再考虑是否值得重新尝试 learnable fusion

---

## 8. 当前状态

本轮 learnable fusion 实验已经完成，并形成明确负结论：

> 在当前设置下，very-light learnable fusion 不能稳定超过 plain rerank。

因此，这条线已经有了足够清楚的 stopping signal，不建议继续作为主线推进。
