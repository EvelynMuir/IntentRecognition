# 实验记录：VLM Rationale Alignment

## 0. 文档信息

- 日期：`2026-03-16`
- 对应设计：`docs/design/design_0315_01_VLM.md`
- 主实验起点：`Qwen2.5-VL-7B-Instruct + offline rationale generation + text-feature alignment`
- 关键输出目录：
  - smoke：`logs/analysis/vlm_smoke_20260315`
  - batch-32：`logs/analysis/vlm_batch32_20260315`
  - batch-128：`logs/analysis/vlm_batch128_20260315`
  - batch-256：`logs/analysis/vlm_batch256_20260316`
  - full rationale：`logs/analysis/vlm_full_20260316`
  - full-train student：`logs/analysis/vlm_full_step1_bge_train_20260316`

---

## 1. main conclusions

1. 本轮已经把 `design_0315_01_VLM.md` 的核心 pipeline 完整打通：
   - 本地 `Qwen2.5-VL-7B-Instruct` + `vllm`
   - 离线 rationale / caption 生成
   - 文本特征提取
   - 基于 cached CLIP image feature 的 student 对齐训练
2. 生成链路里的技术问题都已经解决：
   - 补齐了本地 Qwen2.5-VL snapshot 缺失的 processor/tokenizer 元数据
   - 解决了 `stdin` 下 vllm multiprocessing 的启动问题
   - 通过缩图 (`max_image_size=336`) 与 `max_model_len=2048` 解决了 multimodal token 爆炸和 KV cache 不足
   - 通过提高 `max_tokens=768` 解决了 rationale 文本截断
   - 全量生成脚本已改成分批 `generate` + 即时写盘，可稳定续跑
3. 在 student 侧，`teacher text` 确实比 `label_only` 更容易在小样本上给出一点弱正信号，但这个信号不稳定：
   - `32` 样本、共享 adapted feature 时，`caption / rationale` 都能把 Hard 拉到 `30.28`
   - 但扩大到 `128` 后又回落到 `29.35`
   - 扩到 full train `12739` 后，`Step1-only + BGE` 更是掉到 Hard `28.10`
4. `caption` 和 `rationale` 一直没有明显分开：
   - `32` 样本时几乎完全相同
   - `128` 样本时仍几乎完全相同
   - 即便把文本编码器从 `CLIP-text` 换成 `BAAI/bge-large-en-v1.5`，`Step1 only` 与 `Step1+Step2` 也仍然不分
5. 这说明当前主线的核心问题不再是：
   - 不是 `CLIP-text` 的 `77 token` 限制
   - 不是生成文本被截断
   - 也不再像是“样本量不够”
6. 当前更稳的判断是：

> `offline VLM rationale -> text embedding -> global feature alignment regularizer`
> 这条主线在当前实现下**不成立为正式方法**。  
> teacher 文本的差异没有稳定地传到最终分类边界里。

---

## 2. 本轮新增实现

新增脚本：

- `scripts/generate_vlm_rationales.py`
  - 离线生成 rationale / caption
  - 支持 `gt` 或 `baseline_pred` label source
  - 支持分批 `generate` 与续跑
  - 输出 `finish_reason / generated_token_count`
- `scripts/extract_vlm_rationale_features.py`
  - 支持 `CLIP-text` 与 `BAAI/bge-large-en-v1.5`
  - 可分别提取 `step1 / step2 / step3 / pos / neg` 特征
- `scripts/analyze_vlm_rationale_alignment.py`
  - 基础版 student 对齐训练
  - 后续改成共享 adapted feature
- `scripts/analyze_vlm_pos_neg_repulsion.py`
  - 实现 `T_pos only` 与 `T_neg` 抑制实验
- `scripts/analyze_vlm_cascaded_alignment.py`
  - 实现 `v_vis -> v_ctx` 级联 student

实现边界：

1. 整轮实验都没有改 backbone 主体。
2. student 训练完全建立在已有 `train/val/test *_base.npz` 与 `*_clip.npz` cache 上。
3. teacher 文本只用于训练，不进入验证/测试推理。
4. `val/test` 仍然只看 student 输出，不消费 GT rationale。

---

## 3. 运行与数据生成

### 3.1 关键生成设置

- VLM：`Qwen2.5-VL-7B-Instruct`
- 推理框架：`vllm`
- 图像预缩放：`max_image_size = 336`
- `max_model_len = 2048`
- `max_tokens = 768`
- request batching：`256`

### 3.2 smoke 验证

最小 smoke：

- `logs/analysis/vlm_smoke_20260315/rationale_smoke.jsonl`
- `logs/analysis/vlm_smoke_20260315/rationale_smoke_features.npz`

结果：

1. 单条 rationale 已成功生成并编码成文本向量
2. `finish_reason = stop`
3. 不再出现长度截断

### 3.3 小批量生成质量

`batch8`：

- rationale：
  - `8/8 stop`
  - 平均 `455.25` tokens
  - `8/8` 都完整包含 `Step 1 / Step 2 / Step 3`
- caption：
  - `8/8 stop`
  - 平均 `69.5` tokens

`batch32`：

- rationale：
  - `32/32 stop`
  - 平均 `450.41` tokens
- caption：
  - `32/32 stop`
  - 平均 `68.09` tokens

`batch128`：

- rationale：
  - `128/128 stop`
  - 平均 `454.12` tokens
- caption：
  - `128/128 stop`
  - 平均 `67.91` tokens

full train：

- `logs/analysis/vlm_full_20260316/rationale_full.jsonl`
- 总样本：`12739`
- 全部写出完成

---

## 4. Student 实验结果

### 4.1 baseline 参考

固定 baseline：

- Macro `48.23`
- Micro `52.04`
- Samples `52.94`
- mAP `50.29`
- Hard `29.71`

### 4.2 `8` 样本首轮对比

输出目录：

- `logs/analysis/vlm_compare_batch8_labelonly_20260315`
- `logs/analysis/vlm_compare_batch8_caption_20260315`
- `logs/analysis/vlm_compare_batch8_rationale_20260315`

结论：

- `label_only < caption ≈ rationale`
- 但 `train_n=8` 太小，只能说明 pipeline 可用，不能说明方法成立

### 4.3 `32` 样本，共享 adapted feature

输出目录：

- `logs/analysis/vlm_compare_batch32_labelonly_shared_20260315`
- `logs/analysis/vlm_compare_batch32_caption_shared_20260315`
- `logs/analysis/vlm_compare_batch32_rationale_shared_20260315`

结果：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| baseline | 48.23 | 52.04 | 52.94 | 29.71 |
| label_only | 48.31 | 47.36 | 47.23 | 29.29 |
| caption | 48.34 | 47.55 | 47.54 | 30.28 |
| rationale | 48.34 | 47.55 | 47.54 | 30.28 |

这一步最关键的信号是：

> 把 student 改成共享 adapted feature 后，teacher supervision 才真正开始影响分类结果。  
> 但 `caption` 和 `rationale` 仍然没有分开。

### 4.4 `128` 样本，共享 adapted feature

输出目录：

- `logs/analysis/vlm_compare_batch128_labelonly_shared_20260315`
- `logs/analysis/vlm_compare_batch128_caption_shared_20260315`
- `logs/analysis/vlm_compare_batch128_rationale_shared_20260315`

结果：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| baseline | 48.23 | 52.04 | 52.94 | 29.71 |
| label_only | 48.11 | 47.38 | 47.06 | 29.46 |
| caption | 48.16 | 47.45 | 47.10 | 29.35 |
| rationale | 48.16 | 47.46 | 47.11 | 29.35 |

结论：

- 从 `32 -> 128`，`caption` 和 `rationale` 仍几乎不分
- 而且两者都没超过 baseline

### 4.5 `256` 样本，`T_pos only`

#### CLIP-text / Step1-only

输出目录：

- `logs/analysis/vlm_posonly_batch256_20260316`

结果：

- Macro `48.10`
- Micro `47.80`
- Samples `47.51`
- Hard `29.06`

#### BGE / Step1-only

输出目录：

- `logs/analysis/vlm_step1only_batch256_bge_20260316`

结果：

- Macro `48.11`
- Micro `52.09`
- Samples `52.56`
- Hard `29.62`

这个结果比 `CLIP-text` 版更稳一些，但仍然：

- Macro 低于 baseline
- Hard 低于 baseline

### 4.6 全训练集 `12739`，BGE / Step1-only

输出目录：

- `logs/analysis/vlm_full_step1_bge_train_20260316`

结果：

| Method | Macro | Micro | Samples | Hard |
| --- | ---: | ---: | ---: | ---: |
| baseline | 48.23 | 52.04 | 52.94 | 29.71 |
| full-train `T_pos only` | 47.53 | 46.80 | 46.62 | 28.10 |

这是当前最重要的负结果：

> 扩到 full train 之后，VLM global alignment 路线不但没有变好，反而比 baseline 明显更差。

---

## 5. 进一步诊断

### 5.1 `caption` vs `rationale`

不论 `32` 还是 `128` 样本，`caption` 和 `rationale` 最终 student 指标几乎完全一致。

这不是因为数据读错了：

1. 两类 teacher 文本确实不同
2. 生成长度差异很大
3. 文本特征也不是完全相同

但这些差异没有变成分类边界差异。

### 5.2 `Step1 only` vs `Step1+Step2`

#### CLIP-text

在 `256` 样本下，`Step1 only` 明显优于 `Step1+Step2`。

#### BGE

把文本编码器换成 `BAAI/bge-large-en-v1.5` 后，`Step1 only` 与 `Step1+Step2` 仍然几乎完全一致：

- 两者 test 指标完全重合
- 两者在 BGE 空间里的 teacher feature 仍高度相似
  - mean cosine `0.975`

这说明：

> 问题不只是 `CLIP-text` 的 `77 token` 限制；  
> `Step2` 在当前 teacher-to-student 设定下，本身就没有形成有区分度的监督信号。

### 5.3 `Step3` 反事实抑制

两版 `T_neg` 实验都没有带来额外收益：

- 最早的 learned repulsion 版本没有涨
- 后来的直接 logit suppression 版本也没有涨

结论：

> `Step3` 目前没有证明自己能作为有效的 confuse-class 抑制信号。

### 5.4 级联 student

输出目录：

- `logs/analysis/vlm_cascade_batch256_bge_20260316`

结果：

- Macro `47.85`
- Hard `29.10`

结论：

- 新结构没有救回这条线
- 当前问题不只是 student 结构太简单

---

## 6. 最终判断

这轮 `design_0315_01_VLM.md` 的正式判断应当是：

> **negative result**

原因很明确：

1. 全量 `VLM rationale -> text embedding -> global alignment regularizer` 没有超过 baseline
2. full train `12739` 的 `T_pos only` 结果明显低于 baseline
3. `caption` 和 `rationale` 不分
4. `Step2` 没有提供稳定增益
5. `Step3` 反事实抑制没有形成额外收益

也就是说：

> 当前这条 VLM 主线的问题不再像是样本量、token 长度或编码器选择，  
> 而更像是“global text alignment 这件事本身，对当前 intent classification 决策边界帮助有限”。

---

## 7. 下一步建议

如果还要继续用 VLM，更合理的方向不是再继续堆这条 global alignment 主线，而是：

1. 把 VLM 只用于 **candidate-local / confusion-pair** 级别的 late fusion
2. 让 teacher 信号直接作用于：
   - confuse pair residual
   - candidate rerank
   - threshold / decision rule

而不是继续做：

- 全局 embedding regularizer
- 全局 `caption/rationale` 对齐

这也是为什么下一步更适合转到：

- `docs/design/design_0316_01_VLMLateFusion.md`

而不是继续深挖 `design_0315_01_VLM.md` 当前这条实现。*** End Patch
