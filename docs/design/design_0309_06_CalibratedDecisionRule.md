# 2026-03-09 Calibrated Decision Rule

## 1. 目标

这一步的目标不是再改 backbone 或 prompt，而是回答一个更直接的问题：

> 当前 `SLR-v0` 的主要剩余空间，是否其实来自 decision rule 本身？

也就是说：

- 同样的 score，如果只改 threshold 规则，能不能继续明显涨点？
- 如果能，下一步方法就不一定要继续堆 prior，而可能应优先做 `calibrated decision rule`

---

## 2. 固定口径

### 2.1 Baseline

固定 baseline checkpoint：

- `logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt`

统一协议下 baseline global threshold test：

- `macro = 0.4598`
- `micro = 0.5654`
- `samples = 0.5640`
- `hard = 0.2686`

### 2.2 SLR-v0 strongest config

固定 SLR-v0 strongest config：

- `source = llm`
- `topk = 10`
- `mode = add_norm`
- `alpha = 0.3`

在原分析协议下：

- global re-tuned test:
  - `macro = 0.4843`
  - `micro = 0.5773`
  - `samples = 0.5715`
  - `hard = 0.2879`

### 2.3 本轮实验脚本与输出

新增脚本：

- `scripts/analyze_calibrated_decision_rule.py`

输出目录：

- `logs/analysis/full_calibrated_decision_rule_20260309`

核心文件：

- `summary.json`

---

## 3. Step 1：在 SLR-v0 上重新做 threshold search

本步分别评估：

1. baseline + global threshold
2. baseline + class-wise threshold
3. baseline + group-wise threshold
4. SLR + original baseline threshold
5. SLR + re-tuned global threshold
6. SLR + class-wise threshold
7. SLR + group-wise threshold

### 3.1 Baseline 结果

| Setting | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| baseline + global | 0.4598 | 0.5654 | 0.5640 | 0.2686 |
| baseline + class-wise | 0.4823 | 0.5204 | 0.5294 | 0.2971 |
| baseline + group-wise frequency | 0.4590 | 0.5623 | 0.5596 | 0.2708 |

结论：

- baseline 上单做 class-wise threshold，`macro` 和 `hard` 确实能涨
- 但它会明显伤 `micro / samples`
- 这说明 baseline 的 decision rule 里确实有阈值空间，但它的 gain 带有明显 trade-off

### 3.2 SLR-v0 结果

| Setting | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| SLR + original baseline threshold | 0.4806 | 0.5599 | 0.5638 | 0.3541 |
| SLR + re-tuned global threshold | 0.4843 | 0.5773 | 0.5715 | 0.2879 |
| SLR + class-wise threshold | 0.5171 | 0.5960 | 0.5962 | 0.3344 |
| SLR + group-wise semantic prior gain | 0.4843 | 0.5773 | 0.5715 | 0.2879 |
| SLR + group-wise frequency | 0.4886 | 0.5707 | 0.5634 | 0.3025 |

### 3.3 结果解读

这里最关键的发现非常明确：

> 在当前 strongest SLR-v0 上，`class-wise threshold` 是目前最强的一步后处理增益。

相对 `SLR + re-tuned global threshold`：

- `macro +0.0329`
- `micro +0.0187`
- `samples +0.0247`
- `hard +0.0465`

这不是小修小补，而是非常实打实的一次跃迁。

更重要的是，它和 baseline 不同：

- baseline 的 class-wise threshold 会伤 `micro / samples`
- 但 SLR-v0 的 class-wise threshold 反而是全指标一起涨

这说明：

> 当前最强 rerank 已经把 score structure 调整到一个更适合做 class-specific decision rule 的状态。

这一步直接验证了原文里那句判断：

> “这一步最有可能继续涨。”

结果确实如此。

---

## 4. Step 2：做 group-wise calibration

本步优先试两种分组。

### 4.1 方案 A：按 semantic prior 增益分组

分组方式：

- `prior_benefit`
- `prior_neutral_or_risk`

分组依据：

- 使用 `SLR-v0 val per-class F1 - baseline val per-class F1`
- 正增益类归入 `prior_benefit`
- 非正增益类归入 `prior_neutral_or_risk`

结果：

- 两组最终都选到了同一个 threshold：`0.55`
- test 结果与 `SLR + re-tuned global threshold` 完全相同

结论：

> 当前这套 semantic-gain 二分组，没有提供额外的 decision rule 区分能力。

这说明：

- 现有“benefit / risk”分组太粗
- 或者当前 val 上这两组的最优 threshold 本来就没有显著差异

因此，这个版本暂时不值得作为主要方法形态。

### 4.2 方案 B：按类频次分组

分组方式：

- `head`
- `medium`
- `tail`

实现上按 train positive count 排序后，近似三等分。

得到的 group thresholds 为：

- `head = 0.57`
- `medium = 0.55`
- `tail = 0.53`

对应 test：

- `macro = 0.4886`
- `micro = 0.5707`
- `samples = 0.5634`
- `hard = 0.3025`

相对 `SLR + re-tuned global threshold`：

- `macro +0.0043`
- `hard +0.0146`

### 4.3 group-wise calibration 的判断

group-wise calibration 的结论可以拆成两层：

1. `semantic prior gain grouping`
   - 当前没用
2. `frequency grouping`
   - 有用，但明显不如 `class-wise threshold`

所以如果从“更像方法、比 per-class 更紧凑”的角度看：

> `frequency-based group-wise threshold` 是一个可以保留的中间版本。

但如果从“纯结果最好”看：

> 现在还是 `class-wise threshold` 更强。

---

## 5. Step 3：做 source ensemble

本步只做：

- `short`
- `detailed`
- `short + detailed`

这里的 `short + detailed` 定义为：

- 对两种 source 的 raw text logits 做等权平均
- 再进入同样的 `SLR-v0 add_norm` rerank 流程

并分别评估：

- global re-tuned threshold
- class-wise threshold

### 5.1 Global threshold 结果

| Source | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| `short` | 0.4947 | 0.5756 | 0.5723 | 0.3276 |
| `detailed` | 0.4891 | 0.5660 | 0.5704 | 0.3198 |
| `short + detailed` | 0.5023 | 0.5781 | 0.5779 | 0.3370 |

结论：

- `short + detailed` 明显优于 `short` 和 `detailed` 单独使用
- 单做 source ensemble 就已经比当前 `llm`-based SLR-v0 更强

### 5.2 Class-wise threshold 结果

| Source | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| `short` | 0.4958 | 0.5441 | 0.5385 | 0.3249 |
| `detailed` | 0.5025 | 0.5727 | 0.5722 | 0.3323 |
| `short + detailed` | 0.5105 | 0.5738 | 0.5736 | 0.3412 |

结论：

- `short + detailed + class-wise threshold` 是这一步里最强的 source ensemble 结果
- 它同时优于 `short` 和 `detailed`
- 说明简单 source ensemble 本身是有效的

### 5.3 和 SLR-v0 strongest config 的关系

把这里最强的 source ensemble 和当前 `llm`-based strongest SLR-v0 对比：

| Method | macro | micro | samples | hard |
| --- | ---: | ---: | ---: | ---: |
| `SLR-v0 (llm) + global re-tuned` | 0.4843 | 0.5773 | 0.5715 | 0.2879 |
| `short + detailed + global` | 0.5023 | 0.5781 | 0.5779 | 0.3370 |
| `short + detailed + class-wise` | 0.5105 | 0.5738 | 0.5736 | 0.3412 |

这里有两个很重要的结论：

1. `source ensemble` 本身就是强增益来源
2. `short + detailed` 当前比 `llm` 更稳、更强

这和之前 source 消融的判断一致：

> 当前最强的 semantic prior，不一定来自 raw LLM descriptions，本阶段反而是 `short / detailed / short+detailed` 更可靠。

---

## 6. 总结

这份设计文档里的三步，现在都已经完成，并且可以得出非常明确的结论。

### 6.1 结论一：decision rule 还有很大空间

最强信号来自：

- `SLR-v0 + class-wise threshold`

对应 test：

- `macro = 0.5171`
- `micro = 0.5960`
- `samples = 0.5962`
- `hard = 0.3344`

这是当前所有已做步骤里最显著的一次后处理收益。

### 6.2 结论二：group-wise calibration 可以做，但不是最强

- semantic prior gain grouping：当前无效
- frequency grouping：有用，但不如 class-wise threshold

因此：

- 如果想要一个更“方法化”的 compact rule，可以考虑 `frequency group-wise`
- 如果当前只追求最强结果，直接用 `class-wise threshold`

### 6.3 结论三：source ensemble 值得继续

`short + detailed` 明显有效。

目前最强 source ensemble 结果：

- `short + detailed + class-wise threshold`
- `macro = 0.5105`
- `hard = 0.3412`

这已经非常接近，且在 `hard` 上甚至略高于很多已有尝试。

---

## 7. 下一步建议

基于本轮结果，优先级建议如下：

1. 先把 `SLR-v0 + class-wise threshold` 视为当前最强 calibrated decision rule 基线
2. 如果需要更有方法感的版本，优先做 `frequency-based group-wise threshold`
3. source side 优先继续做：
   - `short`
   - `detailed`
   - `short + detailed`
   不建议现在把重点再放回 `llm`

一句话总结：

> 当前最值得继续推进的不是更复杂的 fusion，而是 `better semantic source + calibrated decision rule`。
