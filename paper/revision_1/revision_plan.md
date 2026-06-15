# FDIL 修订计划（Pattern Recognition, Major Revision）

> 投稿: *Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity*
> 一审结果: **Major Revision**（5 位审稿人 + AE 元评审 + EiC 清单）
> 本文件用于追踪修订进度，并作为 response-to-reviewers letter 的骨架。
> 状态图例: ☐ 未开始 / ◐ 进行中 / ☑ 完成

---

## 1. 总体判断

可救。R1、R5 偏正面；R3 是 AE 给的 3 条具体要求；R4 列了 11 条多为可执行项；**R2（新颖性）是唯一可能致命的反对意见**。

### 两条致命线（P0，决定生死）

1. **新颖性（R2）** — PR 最看重。R2: 二元歧义区分"有吸引力但不构成新理论原则或算法思想，更像现有方法的正确组合"。R4#10 / R1（limited theoretical depth）/ R3 同向。
2. **对比公平性 + 阈值效应（R1）** — 最危险的技术隐患。Table 7 显示仅"global→class-wise 阈值"就带来 +4.3 Avg（52.77→57.09），比 FDIL 相对 CLIP baseline 的总增益 +3.68 还大。Table 1 里 FDIL 同时占了 CLIP ViT-L/14 backbone + class-wise 阈值两个便宜，而对比方法是原论文弱 backbone + 普通阈值的数字。

> **关键耦合**: 只要证明"公平协议下（同 backbone + 同阈值）FDIL 仍有干净可隔离的增益"，就同时回应 R1（公平性）与 R2（解耦确实有用）。**E1 的结果会决定整篇 claim 的话术，必须最先做。**

---

## 2. 逐条审稿意见 → 行动映射（response letter 骨架）

### Reviewer #1（偏正面，最有建设性）
| 意见 | 行动 | 实验/写作 | 状态 |
|------|------|-----------|------|
| 同 backbone + 同阈值的受控基线 | 公平对比网格 + 重跑 SOTA on CLIP 特征 | E1 | ◐ E1a 已完成，E1b 待外部方法复现 |
| class-wise 阈值效应过强 | 四宫格隔离阈值贡献 vs 方法贡献 | E1 | ☑ |
| 报告多次运行方差 | ≥3 seed，mean±std | E2 | ☐ |
| validation-only 模型选择需澄清 | 写明协议 | W12 | ☐ |
| 释放/记录生成 artifact（rationale、prior） | 复现性声明 + 释放方案 | W12 | ☐ |
| 复现细节不足 | 补实现细节 | W12 | ☐ |
| 弱化 "functional decoupling" / "SOTA" 表述，除非消融能更干净地隔离 | 依 E1/E3 结果调整措辞 | W2, E3 | ☐ |
| comparison interpretability 不足 | per-class / confusion-pair / hard-category attribution，解释增益来源与受损类别 | E12 | ☐ |
| threshold 可能主导 F1 | 增加 threshold-free 指标与统计检验 | E7, E13 | ☐ |

### Reviewer #2（负面，新颖性 — 致命）
| 意见 | 行动 | 实验/写作 | 状态 |
|------|------|-----------|------|
| 新颖性不足，"只是现有方法组合" | (a) 信息论/特征选择视角形式化解耦原则；(b) unified vs decoupled 正面对照；(c) 第二数据集证明泛化 | W2, E3, E4 | ☐ |
| VLM/text prior 可能只是外部知识堆叠 | shuffled/random/label-only/caption-only 等 negative controls，证明不是任意文本先验都有效 | E14 | ☐ |

### Reviewer #3（AE 元评审）
| 意见 | 行动 | 实验/写作 | 状态 |
|------|------|-----------|------|
| 1. Abstract 写明量化提升 | 加具体数字 | W3 | ☐ |
| 2. §3.3.1 场景选择的代表性/覆盖性论证 | 补论证 + 未覆盖情况预期 | W5 | ☐ |
| 3. Table 2 Easy 子集 FDIL(81.01)<HLEG(81.49) 未讨论 | 补对照实验与解释 | E11 | ☐ |

### Reviewer #4（11 条，多为可执行）
| # | 意见 | 行动 | 实验/写作 | 状态 |
|---|------|------|-----------|------|
| 1 | proofreading（如 [7] 期刊缩写） | 全文校对 + 修参考文献格式 | W9, W10 | ☐ |
| 2 | 深入分析现有方法在两类歧义上的不足 | 重写 intro 批判段 | W1 | ☐ |
| 3 | intro 缺明确 purpose | 重写 intro | W1 | ☐ |
| 4 | intro 补 LLM-based 视觉意图最新进展 | 扩展综述 | W1 | ☐ |
| 5 | notation 表 | 新增 | W6 | ☐ |
| 6 | 更大规模数据集 | EMOTIC 全 pipeline | E4 | ☐ |
| 7 | 与 2025/2026 大模型对比 + t-SNE | 大模型对比 + t-SNE | E5, E6 | ☐ |
| 8 | ROC 曲线 | ROC/PR 曲线 | E7 | ☐ |
| 9 | 新增 Discussion 章节 | 扩写（现有偏薄，大段被注释） | W4 | ☐ |
| 10 | 更多理论分析（信息论/特征选择） | 形式化论证 | W2 | ☐ |
| 11 | Params/FLOPs/inference 对比 | Table 8 加入其他方法 | E8 | ☐ |

### Reviewer #5（正面，4 个小修）
| # | 意见 | 行动 | 实验/写作 | 状态 |
|---|------|------|-----------|------|
| 1 | Figure 2 caption(Top/Bottom) 与图内 A/B 不一致 | 对齐 caption/图 | W7 | ☐ |
| 2 | Table 5 补 Lexical+Scenario、Canonical+Scenario | 补两行消融 | E9 | ☐ |
| 3 | Table 6 K=5/K=28 优于 K=10 的非单调异常 | 补解释/中间 K 平滑曲线 | E10 | ☐ |
| 4 | Appendix Table B.1/B.2 "FDIR"→"FDIL" | 全文修正 | W8 | ☐ |

### EiC 清单
| 项 | 行动 | 写作 | 状态 |
|----|------|------|------|
| a | 补 2025/2026 SOTA；arXiv→正式版；拆群引逐条点评；最终 35–55 条 | W10 | ☐ |
| b | 引用 PR *领域*（非仅 PR 期刊）近作 | W10 | ☐ |
| c | ≤35 页、double-spaced single column | W11 | ☐ |

---

## 3. 实验清单

### P0 — 决定成败
- ◐ **E1 公平对比网格**: `{CLIP baseline, FDIL} × {global, class-wise threshold}` 四宫格；明确 baseline 53.41 用的哪种阈值。
  - **输出**: `logs/analysis/e1_fair_comparison_20260615/REPORT.md`, `e1_core_four_grid.csv`, `e1_inhouse_controlled_baselines.csv`, `e1_threshold_decomposition.csv`
  - **结论**: `baseline 53.41` 是 CLIP baseline 在 class-wise threshold 下的 `AvgF1=(Macro+Micro+Samples)/3`，不是 mAP；同 class-wise 协议下 final FDIL 相对 CLIP baseline 为 Macro `+3.80`、AvgF1 `+3.68`、Hard `+5.03`；同 global 阈值下 final FDIL 的 Macro/Hard 几乎不变，主要提升 AvgF1 `+0.84`，因此修稿需明确区分方法贡献与 calibration 贡献。
  - **E1a 必做**: ☑ 所有 in-house controllable baselines 在同一 CLIP ViT-L/14 特征、同一 train/val/test、同一 validation-only threshold search 下比较。
  - **E1b 尽力做**: ☐ 重跑/复现代表性 SOTA（HLEG/LabCR/PIP-Net/IntCLIP）于 CLIP ViT-L/14 特征 + class-wise 阈值；若外部方法无法无痛换 backbone，需要在 response 中说明协议限制，并用 E1a 作为公平性主证据。
  - 基础: `scripts/analyze_calibrated_decision_rule.py`, `scripts/run_clip_feature_baseline.py`
- ☐ **E2 多种子方差**: FDIL + 关键基线各 ≥3 seed，主表/消融加 mean±std。
  - 基础: `scripts/run_multiseed_slr_calibration.sh` + `scripts/aggregate_multirun_stability.py`（现成）
- ☐ **E3 解耦原则正面证据**: unified（单机制处理两种歧义）vs decoupled（FDIL）对照，证明解耦非空话。（需新增脚本）

### P1 — 分量重
- ☐ **E4 EMOTIC 第二数据集**: 跑完整 FDIL pipeline 证明泛化。**`scripts/run_emotic_full_pipeline.sh` 全套现成——最高性价比。**
- ☐ **E5 2025/2026 大模型对比**: 恢复 Table 1 注释行（Qwen2-VL-7B/Ovis1.6/GPT-4o）+ 补 1–2 个新 MLLM。基础: `scripts/generate_vlm_rationales.py`
- ☐ **E6 t-SNE**: FDIL vs baseline 的 intent 嵌入可分性（hard/语义相邻类）。
- ☐ **E7 ROC/PR 曲线**: 与 SOTA 对比（评估日志已有 logits）。
- ☐ **E8 效率对比扩展**: Table 8 加入 IntentMLM/IntCLIP/HLEG 的 Params/FLOPs/Latency。
- ☐ **E12 comparison interpretability / attribution**: per-class gain/loss、confusion-pair shift、hard-category attribution；解释 FDIL 到底改善哪些 semantic ambiguity，哪些 easy/visual-clear 类可能被 calibration 轻微伤害。
- ☐ **E13 threshold-free 指标 + 统计显著性**: mAP / macro-AUC / micro-AUC / PR-AUC 数表；FDIL vs CLIP baseline、UTD-only、SLR-C-only 做 paired bootstrap 或 paired permutation test。
- ☐ **E14 VLM/text negative controls**: shuffled rationale、random scenario prior、label-only text、caption-only、ungated teacher、错误 agreement gate；证明增益不是"任意 VLM 文本 + class-wise threshold"带来的。
- ☐ **E15 leakage / protocol audit**: 检查 train/val/test artifact 使用边界；确认 threshold/model selection 只用 validation；说明 test rationale/logits 是否仅用于 evaluation，不参与模型选择。

### P2 — 工作量小
- ☐ **E9** Table 5 补 Lexical+Scenario、Canonical+Scenario 两行。
- ☐ **E10** Table 6 K 非单调现象解释/中间 K 平滑曲线。
- ☐ **E11** Table 2 Easy 子集 FDIL<HLEG 对照与解释。

---

## 4. 写作清单

### P0 — 救新颖性与理论
- ☐ **W1 重构 Introduction**: 明确 purpose（R4#3）；批判现有方法在两类歧义上的失败（R4#2）；扩展 LLM-based 视觉意图综述（R4#4）。
- ☐ **W2 理论深度**: 信息论/特征选择视角形式化"功能解耦"——监督歧义=标签端条件熵/噪声，语义歧义=特征端类间互信息混叠，论证需正交两模块。（R2/R4#10/R1）
- ☐ **W3 Abstract**: 加量化提升数字。（R3#1）

### P1
- ☐ **W4 扩写 Discussion**: main findings + 优点 + 失败模式。（R4#9；现有 `05_discussion.tex` 偏薄）
- ☐ **W5 §3.3.1**: 场景代表性/覆盖性论证 + 未覆盖情况预期。（R3#2）
- ☐ **W6 Notation 表**。（R4#5）

### P2 — 文字/格式
- ☐ **W7** Figure 2 caption 与图内 A/B 对齐。（R5#1, `03_method.tex:4`）
- ☐ **W8** Appendix Table B.1/B.2 "FDIR"→"FDIL"（`07_appendix.tex:274,288`，全文搜残留）。（R5#4）
- ☐ **W9** 全文 proofread + 参考文献格式（[7] 期刊缩写）。（R4#1）
- ☐ **W10** 参考文献整改: 补 2025/2026 SOTA、arXiv→正式版、拆群引逐条点评、引 PR 领域近作、控制 35–55 条。（EiC a/b）
- ☐ **W11** 页数 ≤35、double-spaced single column。（EiC c）
- ☐ **W12** 复现性: validation-only 选择协议 + artifact 释放方案（"on request" 不够）。（R1）
- ☐ **W13 Response letter**: 逐条 response、修改位置索引、改前/改后摘要、给每位 reviewer 的 opening paragraph；所有新增表图与 response 引用保持一致。
- ☐ **W14 Artifact/Data availability 清单**: scenario priors、rationale jsonl、BGE/CLIP text features、threshold files、train/val/test logits、prompt templates、model seeds、VLM version、hash/checksum、生成失败/过滤规则；Data availability 从 "available on request" 改为具体 repository/supplementary statement。
- ☐ **W15 Cross-reference / PDF audit**: 新增表图后统一检查 Table/Figure 编号、appendix label、response letter 引用、LaTeX warnings、页数限制。
- ☐ **W16 Claim downgrade variants**: 准备两套 abstract/conclusion 话术。A 档: 公平协议显著领先，保留 SOTA；B 档: 公平协议净增益中等，主打 controlled ambiguity decomposition、hard-category robustness、低 inference overhead，弱化 SOTA。

---

## 5. 推进顺序

1. **E1 + E2 先行**（诚信底线 + 摸清 FDIL 真实净增益；决定后续话术）。若 E1 显示公平协议下增益很小 → 全篇 claim/卖点需调整。
2. 同步起 **E4（EMOTIC）**——跑得久，尽早排 GPU；但先作为泛化候选验证，不预设一定进主文。
3. **E3、E5、E13、E14** 跟上（新颖性正面证据 + 大模型对比 + threshold-free/negative-control 安全垫）。
4. 并行写 **W1/W2/W3/W12/W14**，用 E1 真实数字和 E15 protocol audit 结果。
5. 最后清扫 P2 全部小修（约 1 天）+ **W13 response letter** + **W15 cross-reference/PDF audit**。

---

## 6. 风险点

- **E1 反转风险**: 公平协议下 FDIL 净增益若过小，需把卖点从"绝对 SOTA"转向"在受控条件下解耦带来稳定增益 + 理论解释 + 跨数据集泛化"。
- **新颖性**: 仅靠补实验难翻 R2，必须有 W2 的理论升格 + E3 的正面证据双管齐下。
- **E4 数据集适配风险**: EMOTIC 是 subjective multi-label affect/emotion recognition，不是 visual intent 原生 benchmark。若结果不正或任务迁移解释牵强，不放主文；最多作为 appendix/limitation 中的泛化探索，避免硬说成同任务大规模验证。
- **外部 SOTA 重跑风险**: HLEG/LabCR/PIP-Net/IntCLIP 未必都能公平替换为 CLIP ViT-L/14 特征。E1b 失败时不要卡住修稿，优先用 E1a 的 controlled in-house baselines 回应公平性。
- **artifact/leakage 风险**: 离线 VLM rationale、scenario prior、val/test logits 必须有清楚边界；若 test artifacts 曾被用于选模，需要改流程或在最终稿中完全避开相关结果。
- **页数**: 大量补实验 + 重写可能超 35 页，注意把次要消融移入 appendix。
