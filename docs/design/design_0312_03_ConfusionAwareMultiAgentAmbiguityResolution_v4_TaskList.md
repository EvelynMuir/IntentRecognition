# v4 任务清单：Confusion-Aware Conditional Multi-Agent Ambiguity Resolution

## 0. v4 的核心目标

v3 已经证明：

- `routed-specialists` 比 `all-specialists` 更合理
- `v3 routed` 超过了 `SLR-C`
- specialist-based local adjudication 对局部消歧有效

但 v3 还没有真正证明：

- **conditional triggering 成立**
- router 本身有方法价值
- multi-agent framework 能以更稀疏、更针对的方式处理 hard ambiguity

因此，v4 的唯一主目标是：

> **把 router 从“几乎总触发”的宽触发器，改造成真正稀疏、可解释、对 hard ambiguity 更有效的 conditional router。**

v4 不追求大规模扩方法，不优先增加新模块；优先做一个**更干净、更可解释、更适合论文叙事**的 router-tightening round。

---

## 1. v4 总体策略

### 做什么

v4 主要做三件事：

1. **收紧 confusion hit 定义**
2. **重新设计 trigger logic（特别是 AND / OR 逻辑）**
3. **把 trigger sparsity / compute efficiency 正式纳入主结果**

### 不做什么

v4 暂时不优先做：

- learned router
- learned resolver
- 更多 specialist agents
- 更多 evidence spaces
- 大范围 relation family 搜索
- 大规模调 beta / gamma

原因：当前主瓶颈已经非常明确，是 **router 太宽**，不是后端 evidence 模块不够复杂。

---

## 2. v4 方法改动任务

## Task 1. 收紧 confusion hit 定义

### 目标
把当前几乎全覆盖的 confusion trigger 改成真正只针对高风险局部歧义。

### 需要实现的版本

#### 1.1 Top-2 confusion hit
仅当 `top-1` 和 `top-2` 组成高频 confusion pair 时触发。

这是 v4 的**第一优先版本**。

#### 1.2 Top-3 local confusion hit
仅检查以下 pair：

- `(top1, top2)`
- `(top1, top3)`
- `(top2, top3)`

若其中任一 pair 命中高频 confusion pair，则触发。

#### 1.3 High-frequency confusion pair filtering
对 confusion pair 增加过滤条件，例如：

- 只保留 confusion matrix 前 `N` 个 pair
- 或只保留频次高于阈值 `f_min` 的 pair
- 或只保留 directed confusion 强度高于阈值的 pair

### 预期效果

- confusion trigger rate 显著下降
- ambiguity region 更集中
- router 更接近真正的 conditional module

---

## Task 2. 重构 trigger logic

### 目标
比较不同触发逻辑，验证 router 是否真的对 hard ambiguity 有选择能力。

### 需要实现的 trigger modes

#### 2.1 Baselines
- `always`
- `margin_only`

#### 2.2 Tight confusion modes
- `confusion_top2_only`
- `confusion_top3_only`

#### 2.3 Combined modes
- `margin AND confusion_top2`
- `margin OR confusion_top2`
- `margin AND confusion_top3`
- `margin OR confusion_top3`

### 说明

v4 特别要重视 `AND` 逻辑，因为它更符合：

> 只有在模型本身不确定、且候选确实落在高频 confusion region 时，才触发二阶段 multi-agent disambiguation。

### 预期目标

- 找到一个比 `always` 更稀疏的 routing 方案
- 同时维持 routed system 对 `SLR-C` 的优势
- 尽量缩小与 v2 在 `Hard` 上的差距

---

## Task 3. 显式记录 router 统计量

### 目标
把 router 从“辅助诊断”提升为论文主结果的一部分。

### 每个配置都必须输出

- `trigger_rate`
- `margin_trigger_rate`
- `confusion_trigger_rate`
- `avg_specialists_called`
- `avg_neighborhood_size`
- 如可能，再加：`avg_pairs_resolved`

### 可选但很有价值的统计

- 不同类别/类别组的触发率
- 不同 confusion pair 的命中频率
- routed specialist 使用直方图

### 作用
这些统计不是附录性质，而是 v4 主张的一部分：

> v4 是一个稀疏、条件触发、局部工作的 multi-agent disambiguation framework。

---

## Task 4. 固定后端，只测 router

### 目标
避免同时修改太多变量，让 v4 的结论清晰可解释。

### v4 中默认固定的部分

- base system：`scenario SLR-C`
- specialist set：沿用 v3 routed specialists
- evidence backend：沿用 v3 当前实现
- resolver：沿用 v3 当前 neighborhood-only resolver
- relation：默认先固定 `hard_negative_diff`

### 仅允许小范围调的变量

- `margin_tau`
- confusion pair filter threshold
- top-2 / top-3 触发范围
- 必要时少量 `beta`

### 原则

v4 是 **router study**，不是新的 full-method search。

---

## 3. v4 实验任务清单

## EXP-1. Router tightening 主表

### 目标
得到一张最关键的 v4 主结果表，回答：

- router 收紧后是否仍有收益？
- 是否真的更稀疏？
- 是否更贴近 hard ambiguity？

### 建议比较配置

1. `always`
2. `margin_only`
3. `confusion_top2_only`
4. `margin AND confusion_top2`
5. `margin OR confusion_top2`
6. `confusion_top3_only`
7. `margin AND confusion_top3`
8. `margin OR confusion_top3`

### 每个配置报告

主指标：
- Macro F1
- Micro F1
- Samples F1
- mAP
- Hard

Router 指标：
- trigger rate
- avg specialists called
- avg neighborhood size

### 判据
优先看：

1. trigger rate 是否明显低于 `always`
2. `Hard` 是否不掉，最好上升
3. `Macro` 是否维持或提升
4. routed 是否仍然优于 `SLR-C`

---

## EXP-2. Top-2 / Top-3 confusion 定义比较

### 目标
验证“confusion scope”本身是否决定了 router 的质量。

### 比较内容
- `top2 confusion`
- `top3 confusion`
- 原 v3 宽 confusion 定义（作为 reference）

### 关注点
- trigger sparsity
- Hard
- pairwise ranking accuracy
- top-2 disambiguation accuracy

### 预期
我更看好 `top2 confusion` 先跑出更干净的结果，因为它最符合“局部歧义”的定义。

---

## EXP-3. Hard subset 专项评估

### 目标
证明 v4 的改动确实服务于目标子问题，而不是全局调参偶然涨分。

### 子集定义建议
- `low-margin subset`
- `top2-confusion subset`
- `top3-confusion subset`
- semantic-neighbor subset（若已有）

### 额外报告指标
- pairwise ranking accuracy
- top-2 disambiguation accuracy
- Hard

### 关注点
如果 v4 在这些子集上明显更优，即使全局平均提升有限，也足以支持论文叙事。

---

## EXP-4. Efficiency / sparsity 分析

### 目标
证明 v4 不只是“改一个规则”，而是让 multi-agent system 真正变稀疏。

### 需要报告
- 平均每样本调用 specialists 数量
- 触发样本比例
- 触发样本的平均 neighborhood size
- 若可得，额外 runtime / latency

### 理想结果
例如出现这种趋势：

- trigger rate 大幅下降
- avg specialists called 下降
- Hard 不掉甚至上升

这会非常有论文价值。

---

## EXP-5. 失败模式分析（建议做）

### 目标
看 router 收紧之后，哪些样本被漏掉，哪些样本仍被误触发。

### 建议分析
- 原本 v3 routed 能修正、但 v4 不再触发的样本
- v4 仍触发但无收益甚至误伤的样本
- 不同 intent pair 的触发精度

### 作用
帮助决定后面是否需要：
- 轻量 learned router
- 更好的 confusion pair selection
- 更精确的 pair/group definition

---

## 4. v4 阶段性成功标准

v4 不要求立刻全面超过 v2，但至少要满足下面几条中的大部分：

### Success Criterion A
相比 v3 当前版本，trigger rate 显著下降。

### Success Criterion B
相比 `always` 或宽触发版本，`Hard` 保持不降，最好提升。

### Success Criterion C
相比 `SLR-C`，仍保持 routed ambiguity resolver 的整体优势。

### Success Criterion D
在 hard subset 上，top-2 disambiguation accuracy / pairwise ranking accuracy 保持强势。

如果这些条件成立，就说明：

> v4 已经把 v3 从“几乎全触发的多模块系统”推进成“真正的条件触发 multi-agent ambiguity resolver”。

---

## 5. v4 不同结果下的后续分支

## 情况 A：router 收紧后，性能更好且更稀疏

### 结论
v4 成功。

### 后续
- 可以继续写成论文主线
- 下一步可考虑轻量 learned router 或更强 resolver

---

## 情况 B：router 收紧后，稀疏了，但性能略掉

### 结论
router 方向是对的，但边界太严。

### 后续
- 轻调 `margin_tau`
- 放宽 confusion pair 过滤
- 改进 top-2/top-3 neighborhood 规则

---

## 情况 C：router 收紧后，性能明显下降

### 结论
当前系统收益主要来自 routed specialists / local resolver，本质上还不够依赖真正稀疏触发。

### 后续
- 需要重新判断论文叙事是否更适合写成“specialist local resolver”而非强 conditional-agent story

---

## 6. 建议的实现优先顺序

### Priority 1
先实现：
- `confusion_top2_only`
- `margin AND confusion_top2`
- `margin OR confusion_top2`

### Priority 2
再实现：
- `confusion_top3_only`
- `margin AND confusion_top3`
- `margin OR confusion_top3`

### Priority 3
最后补：
- high-frequency confusion pair filtering
- directed confusion thresholding
- 更多 router diagnostics

---

## 7. 一句话总结

v4 的任务不是扩方法，而是：

> **把 v3 中已经证明有效的 routed-specialist local resolver，配上一个真正稀疏、可解释、对 hard ambiguity 更精准的 conditional router。**

只有这一步做扎实，multi-agent 的论文叙事才会真正站稳。 
