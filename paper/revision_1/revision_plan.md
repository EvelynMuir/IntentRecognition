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
| 同 backbone + 同阈值的受控基线 | 公平对比网格 + 按原方法重训 SOTA，仅替换 CLIP ViT-L/14 backbone | E1 | ☑ E1a 完成，E1b frozen/finetuned CLIP 重训完成 |
| class-wise 阈值效应过强 | 四宫格隔离阈值贡献 vs 方法贡献 | E1 | ☑ |
| 报告多次运行方差 | ≥3 seed，mean±std | E2 | ☑ |
| validation-only 模型选择需澄清 | 写明协议 | W12 | ☐ |
| 释放/记录生成 artifact（rationale、prior） | 复现性声明 + 释放方案 | W12 | ☐ |
| 复现细节不足 | 补实现细节 | W12 | ☐ |
| 弱化 "functional decoupling" / "SOTA" 表述，除非消融能更干净地隔离 | 依 E1/E3 结果调整措辞 | W2, E3 | ◐ E3 完成且已据此软化 intro/discussion/letter 措辞；W2 理论稿待写 |
| comparison interpretability 不足 | per-class / confusion-pair / hard-category attribution，解释增益来源与受损类别 | E12 | ◐ 旧版 per-class gain / confusion-shift artifact 已有，但不是当前 E1/E2 最终协议下的 FDIL attribution；需重算或谨慎作为补充 |
| threshold 可能主导 F1 | 增加 threshold-free 指标与统计检验 | E7, E13 | ◐ mAP 已在 E1/E2 主结果中完成并写入；ROC/PR 曲线与 paired significance 仍未完成 |

### Reviewer #2（负面，新颖性 — 致命）
| 意见 | 行动 | 实验/写作 | 状态 |
|------|------|-----------|------|
| 新颖性不足，"只是现有方法组合" | (a) 信息论/特征选择视角形式化解耦原则；(b) unified vs decoupled 正面对照；(c) 第二数据集证明泛化 | W2, E3, E4 | ☑ E3(b) 已用 healthy seed20260616 重跑：decoupled FDIL **优于** unified joint-target fusion（AvgF1 +2.51），相对最强单信号 unified 控制（UTD-only）近似持平（详见 §3 P0 E3）；E4 EMOTIC 旧 pipeline 已跑但结果不支持主 claim；新颖性论证应主要依赖 W2(理论)+E1/E2 controlled gains+E14 negative controls |
| VLM/text prior 可能只是外部知识堆叠 | shuffled/random/label-only/caption-only 等 negative controls，证明不是任意文本先验都有效 | E14 | ☑ shuffled_rationale 灾难性(−7.8 Macro，比去掉 teacher 还差)、ungated/uniform_gate 小幅、no_utd 大幅 → 证明 rationale 内容+gating 是关键，非"任意文本"；已写入 R2.2 与正文 tab:negative_controls |

### Reviewer #3（AE 元评审）
| 意见 | 行动 | 实验/写作 | 状态 |
|------|------|-----------|------|
| 1. Abstract 写明量化提升 | 加具体数字 | W3 | ☐ |
| 2. §3.3.1 场景选择的代表性/覆盖性论证 | 补论证 + 未覆盖情况预期 | W5 | ☐ |
| 3. Table 2 Easy 子集 FDIL(81.01)<HLEG(81.49) 未讨论 | 补对照实验与解释 | E11 | ◐ 原始结果已在 Table 2；正文已有“FDIL 主要改善 Medium/Hard”的泛化解释，但 response 中仍需点名说明 Easy 差 `-0.48` |

### Reviewer #4（11 条，多为可执行）
| # | 意见 | 行动 | 实验/写作 | 状态 |
|---|------|------|-----------|------|
| 1 | proofreading（如 [7] 期刊缩写） | 全文校对 + 修参考文献格式 | W9, W10 | ☐ |
| 2 | 深入分析现有方法在两类歧义上的不足 | 重写 intro 批判段 | W1 | ☐ |
| 3 | intro 缺明确 purpose | 重写 intro | W1 | ☐ |
| 4 | intro 补 LLM-based 视觉意图最新进展 | 扩展综述 | W1 | ☐ |
| 5 | notation 表 | 新增 | W6 | ☐ |
| 6 | 更大规模数据集 | EMOTIC 全 pipeline | E4 | ◐ 2026-03 旧 pipeline 已有结果，但 FDIL/UTD 未超过 EMOTIC baseline；不建议作为“泛化证明”主文证据 |
| 7 | 与 2025/2026 大模型对比 + t-SNE | 大模型对比 + t-SNE | E5, E6 | ◐ E5 完成；E6 t-SNE 未完成 |
| 8 | ROC 曲线 | ROC/PR 曲线 | E7 | ☐ |
| 9 | 新增 Discussion 章节 | 扩写（现有偏薄，大段被注释） | W4 | ☐ |
| 10 | 更多理论分析（信息论/特征选择） | 形式化论证 | W2 | ☐ |
| 11 | Params/FLOPs/inference 对比 | Table 8 加入其他方法 | E8 | ◐ Baseline/UTD/FDIL feature-level Params/FLOPs/Latency 已在正文；IntentMLM/IntCLIP/HLEG 对比仍缺 |

### Reviewer #5（正面，4 个小修）
| # | 意见 | 行动 | 实验/写作 | 状态 |
|---|------|------|-----------|------|
| 1 | Figure 2 caption(Top/Bottom) 与图内 A/B 不一致 | 对齐 caption/图 | W7 | ☐ |
| 2 | Table 5 补 Lexical+Scenario、Canonical+Scenario | 补两行消融 | E9 | ☐ |
| 3 | Table 6 K=5/K=28 优于 K=10 的非单调异常 | 补解释/中间 K 平滑曲线 | E10 | ☑ K=5/10/28 已有重训结果并写入正文解释；中间 K 平滑曲线仅作为可选加强 |
| 4 | Appendix Table B.1/B.2 "FDIR"→"FDIL" | 全文修正 | W8 | ☐ |

### EiC 清单
| 项 | 行动 | 写作 | 状态 |
|----|------|------|------|
| a | 补 2025/2026 SOTA；arXiv→正式版；拆群引逐条点评；最终 35–55 条 | W10 | ☐ |
| b | 引用 PR *领域*（非仅 PR 期刊）近作 | W10 | ☐ |
| c | ≤35 页、double-spaced single column | W11 | ☐ |

---

## 3. 实验清单

### 已有结果快速盘点（本轮核对）

已经有明确结果、可直接服务修稿的实验：

- **E1/E1b**: 公平 backbone/threshold 对比与外部方法 CLIP ViT-L/14 controlled/retrain 证据已完成。
- **E2**: CLIP baseline、UTD only、FDIL 三种核心设置的 3-seed mean±std 已完成。
- **E3**: unified vs decoupled 已用 healthy seed 重跑。decoupled FDIL 全面优于 joint-target fusion（AvgF1 +2.51），但相对最强单信号 unified 控制（UTD-only）近似持平；写作以模块化为主、准确率优势为次要，不写成全面碾压。
- **E5**: Qwen3-VL-8B、InternVL3-8B、GPT-4o zero-shot MLLM 对比已完成并写入正文。
- **E10**: K=5/10/28 sensitivity 已有重训结果和正文解释；仅“中间 K 平滑曲线”仍是可选加强。
- **E14**: rationale/teacher/gate negative controls 已完成并写入正文与 R2 response。

已有结果但不宜当作强主文 claim 的实验：

- **E4**: EMOTIC pipeline 已在 2026-03 跑过，结果显示 baseline 已强、UTD/SLR-C/FDIL 未带来稳定提升；可作为“跨任务泛化风险/limitation”证据，不宜写作第二数据集泛化成功。
- **E8**: 已有 Baseline/UTD/FDIL 的 feature-level Params/FLOPs/Latency，但尚未加入 IntentMLM/IntCLIP/HLEG 等外部方法。
- **E11**: Easy subset 的原始差异已在表中存在（FDIL `81.01` vs HLEG `81.49`, `-0.48`），正文已有 Medium/Hard 改善解释，但 response 仍需点名回应。
- **E12**: 旧版 per-class gain/confusion-shift artifact 已有（如 `logs/analysis/full_data_driven_agent_evidence_verification_20260311/`），但不是当前最终 FDIL/E1/E2 协议下的 attribution，不能直接当最终证据。
- **E13**: threshold-free mAP 已在 E1/E2 中完成；macro-AUC/micro-AUC/PR-AUC 与 paired bootstrap/permutation test 仍缺。

尚未找到可用完成结果的实验：**E6 t-SNE、E7 ROC/PR 曲线、E9 Lexical+Scenario/Canonical+Scenario、E15 leakage/protocol audit**。

### P0 — 决定成败
- ☑ **E1 公平对比网格**: `{CLIP baseline, FDIL} × {global, class-wise threshold}` 四宫格；明确 baseline 53.41 用的哪种阈值。
  - **输出**: `logs/analysis/e1_fair_comparison_20260615/REPORT.md`, `e1_core_four_grid.csv`, `e1_inhouse_controlled_baselines.csv`, `e1_threshold_decomposition.csv`
  - **结论**: `baseline 53.41` 是 CLIP baseline 在 class-wise threshold 下的 `AvgF1=(Macro+Micro+Samples)/3`，不是 mAP；同 class-wise 协议下 final FDIL 相对 CLIP baseline 为 Macro `+3.80`、AvgF1 `+3.68`、Hard `+5.03`；同 global 阈值下 final FDIL 的 Macro/Hard 几乎不变，主要提升 AvgF1 `+0.84`，因此修稿需明确区分方法贡献与 calibration 贡献。
  - **E1a 必做**: ☑ 所有 in-house controllable baselines 在同一 CLIP ViT-L/14 特征、同一 train/val/test、同一 validation-only threshold search 下比较。
  - **E1b 必做**: ☑ 完成两条外部方法证据流：(1) frozen CLIP ViT-L/14 cached feature 上的受控机制复现；(2) `~/lambda/projects/IntentRecognition/{HLEG,LabCR,PIP-Net,IntCLIP}` 原始 image-level 训练/评估入口中替换 CLIP ViT-L/14 backbone 的重训/续评。
  - **E1b 代码入口**: `scripts/e1b_retrain_clip_vitl14_original.sh`, `scripts/e1b_retrain_clip_vitl14_original.slurm`
  - **E1b 输出**: `logs/analysis/e1b_original_clip_retrain_20260617/REPORT.md`, `e1b_frozen_and_finetuned_results.csv`, `e1b_original_clip_retrain_results.csv`, `summary.json`；frozen controlled 原始表见 `logs/analysis/e1b_clip_feature_sota_20260615/e1b_controlled_sota_comparison.csv`。
  - **E1b frozen CLIP controlled class-wise test 结果**: HLEG Macro/Micro/Samples/AvgF1/mAP/Hard = `48.17/57.29/56.81/54.09/51.72/26.42`；LabCR = `48.04/57.13/56.65/53.94/51.72/29.33`；PIP-Net = `15.02/14.91/14.23/14.72/13.03/6.64`；IntCLIP = `47.28/56.18/54.86/52.77/49.84/26.78`。
  - **E1b original image-retrain test 结果**: HLEG finetuned Macro/Micro/Samples/AvgF1/mAP = `22.26/37.83/35.29/31.79/39.75`；LabCR finetuned = `6.62/21.79/20.73/16.85/8.57`；PIP-Net finetuned = `3.67/24.71/24.62/17.67/NA`；IntCLIP original frozen eval-only = `36.21/48.33/45.85/43.46/41.16`。
  - **E1b 结果口径**: frozen 行是同 frozen CLIP ViT-L/14 feature、同 split、validation-only class-wise threshold 的公平受控复现；original retrain 行是外部 repo 原始 image-level entry。HLEG/LabCR 已用原方法 test/eval 入口加载 `model_best.pth.tar` 单独完成 test split eval；本地未发现 IntCLIP ViT-L/14 `finetune_backbone=True` 输出。
  - **E1b 当前实现**: HLEG/LabCR 增加 7×7 dense CLIP ViT-L/14 backbone adapter，PIP-Net 增加 prototype-compatible CLIP ViT-L/14 backbone，IntCLIP 增加 ViT-L/14 model config 与 ViT region-token 输出；frozen/finetuned CLIP 数值已回填。
  - 基础: `scripts/analyze_calibrated_decision_rule.py`, `scripts/run_clip_feature_baseline.py`
- ☑ **E2 多种子方差**: FDIL + 关键基线各 ≥3 seed，主表/消融加 mean±std。
  - **输出**: `logs/analysis/e2_multiseed_stability_20260615/REPORT.md`, `e2_mean_std.csv`, `e2_seed_level_metrics.csv`, `summary.json`
  - **种子**: CLIP baseline / UTD only 使用 `20260316,20260615,20260616`；final FDIL LCS K=5 使用 `20260317,20260615,20260616`。
  - **class-wise 结论**: CLIP baseline Macro `47.07±0.83`, AvgF1 `53.37±0.72`, Hard `27.19±1.14`；UTD only Macro `50.20±0.44`, AvgF1 `55.68±0.44`, Hard `30.27±0.57`；final FDIL Macro `51.14±0.39`, AvgF1 `56.73±0.27`, Hard `32.12±1.26`。
  - **补充说明**: 本轮 E2 在 frozen CLIP ViT-L/14 cache 与 cached text features 上训练/聚合 lightweight student/residual heads；threshold 仍由各 seed 的 validation split 独立选择。
  - 基础: `scripts/build_e2_multiseed_stability.py`；历史 calibration-only 聚合器 `scripts/aggregate_multirun_stability.py` 保留但不作为本轮 E2 主证据。
- ☑ **E3 解耦原则正面证据**: unified（单机制处理两种歧义）vs decoupled（FDIL）对照，证明解耦非空话。
  - **代码入口**: `scripts/build_e3_unified_vs_decoupled.py`
  - **输出（healthy seed 重跑后）**: `logs/analysis/e3_unified_vs_decoupled_seed20260616/REPORT.md`, `e3_unified_vs_decoupled.csv`, `summary.json`, `unified_joint_target_kd_best.pt`（旧的 buggy `e3_unified_vs_decoupled_20260617/` 已弃用，勿引用）
  - **核心设置**: 同 frozen CLIP ViT-L/14 cache、同 Intentonomy split、同 validation-only class-wise threshold；unified joint-target KD 将 SLR-C 概率与 rationale teacher 概率按 `0.5/0.5` 混成单一 KD target，不使用 residual SLR-C decomposition 或 agreement-aware gating。`--fdil-summary`/`--utd-summary`/`--teacher-run-dir` 全部指向 healthy seed20260616 run（`e2_distillation_slrc_lcs_topk5_seed20260616` + `e2_privileged_distillation_seed20260616`），`--seed 20260616`。
  - **结果（healthy seed20260616）**: Unified UTD-only Macro/Micro/Samples/AvgF1/mAP/Hard = `50.68/58.19/59.58/56.15/53.34/30.97`；Unified SLR-C-only = `49.43/58.71/57.64/55.26/53.79/29.11`；Unified joint-target KD = `48.63/56.55/56.58/53.92/53.88/29.09`；Decoupled FDIL = `51.57/58.86/58.86/56.43/55.12/32.43`。
  - **写作口径（honest middle）**: healthy base 下没有任何单信号 collapse（SLR-C-only AvgF1 55.26 / mAP 53.79）。但把两信号硬塞进单一 joint-target KD 反而最差（AvgF1 53.92），decoupled FDIL 全面优于该 joint fusion（Macro +2.94 / AvgF1 +2.51 / Hard +3.34 / mAP +1.24）；相对最强单信号 unified 控制（UTD-only）则在 AvgF1 上近似持平（+0.28，落在 seed variance 内）。正文写作以“模块化”为主贡献、把对 joint fusion 的准确率优势作为次要且一致的发现，不写成全面碾压。
  - **✅ 已用 healthy seed 重跑（E13 发现的 bug 已修复）**: 旧 E3 经 `DEFAULT_FDIL_SUMMARY = distillation_slrc_lcs_topk5_20260327`（SLR-C base 未训练）导致 “Unified SLR-C-only = 18.69 / 语义先验 collapse” 伪影。已用 healthy seed20260616 summaries + teacher 重跑，正文 `04_experiments.tex` `tab:unified_decoupled`、`03_method.tex` P3、`05_discussion.tex`、`responses/3_reviewer2.tex (b)` 已全部改写为 healthy 数值与 honest-middle 口径。注：`build_e3_unified_vs_decoupled.py` 现内置 lightning/rich/hydra meta-path stub 以便在 `s2d` env（torch+clip，无 lightning）运行。
  - **✅ 完整方法 = 全局统一 3-seed 均值 + 切换到 K=10（2026-06-18 最终）**: 用户最终口径“所有 FDIL full method 处用同一组数字、3-seed 均值 everywhere”。流程：(1) 重训 healthy seed0317（privileged+distillation），组成全 healthy 3-seed 集（0317h/0615/0616）；(2) 训 K=10/K=28 ×3 seeds、重跑 E3 ×3 seeds（`scripts/_run_3seed_extra.py` + E3-K10 循环）；(3) **K-sensitivity 用 healthy 3-seed 后翻盘：K=10 最优（Avg 57.49），旧“K=10 dip”是 single-seed/bug 伪影**，用户决定**把 full method 切到 K=10**。
  - **新 canonical FDIL-full（K=10 3-seed 均值，seeds 0317h/0615/0616）= `52.13/60.06/60.28/57.49AvgF1/55.10mAP/35.02Hard`**，已写入所有表（main_results、matched_backbone、thresholding_test、negative_controls、slrc_ablation、k_sensitivity[K=10 adopted]、backbone CLIP、E3、difficulty/content test+val 含 appendix）+ 全部正文 delta + 回复信（R1/R2/R5/general）。方法定义 Top-5→Top-10。mAP headline +3.57→**+4.12**（UTD+2.09、+SLR-C+2.03）。子集表用 stored `per_class_f1` + `src/utils/metrics.py` 的 `SUBSET2IDS` 重算，baseline 行也改 3-seed（baseline 现赢 Easy/Context，FDIL 在更难子集上提升）。两个 PDF 均编译通过。
  - **残留 caveat**: `negative_controls` 的 Δ 行与 `prior_ablation` 仍是 single-seed/K=5（非 full-method 行，未动）。E2 multiseed CSV 仍引旧 buggy seed0317；如需可用 healthy 0317 重生成 `e2_mean_std`。

### P1 — 分量重
- ◐ **E4 EMOTIC 第二数据集**: 旧 pipeline 已跑完，但结果不支持“第二数据集泛化成功”主 claim。
  - **已有输出**: `logs/analysis/emotic_clip_baseline_full_20260323/summary.json`, `logs/analysis/emotic_privileged_distillation_20260323/main_comparison.csv`, `logs/analysis/emotic_distillation_slrc_20260324/main_comparison.csv`, `logs/analysis/emotic_vlm_20260323/`
  - **baseline class-wise test**: Macro/Micro/Samples/mAP/Hard = `39.91/53.65/53.91/38.69/32.59`。
  - **UTD/privileged distillation test**: baseline `40.01/53.58/53.90/38.51/32.48`; standard KD `38.89/54.38/54.66/37.13/31.72`; dynamic gated KD `39.67/53.02/53.29/38.42/32.46`。
  - **SLR-C + residual on EMOTIC test**: SLR-C fixed `39.65/52.11/52.39/38.24/32.55`; residual supervised `39.42/53.40/53.50/38.28/32.17`; residual standard KD `38.88/54.35/54.60/36.97/31.40`; residual dynamic KD `39.30/52.68/52.86/38.37/31.86`。
  - **写作口径**: 不能写作“跨数据集显著泛化”。若需要回应 R4#6，可写成 appendix/limitation: EMOTIC 是 affect/emotion 而非 intent，旧适配结果显示 FDIL 的 ambiguity decomposition 不会自动迁移到所有 subjective multi-label tasks。
- ☑ **E5 2025/2026 大模型对比**: 按 probability-dictionary prompt 补 zero-shot MLLM 对比，并写入正文 Table `tab:mllm_zeroshot` 与 Reviewer 4 response。
  - **代码入口**: `scripts/run_intentonomy_vllm_zeroshot.py`
  - **输出**: `outputs/vllm_zeroshot/qwen3_vl_8b_instruct_prompt_image_maxtok512/metrics.json`, `outputs/vllm_zeroshot/internvl3_8b_prompt_image_maxtok512/metrics.json`
  - **Qwen3-VL-8B-Instruct test**: Macro/Micro/Samples/Avg/mAP = `38.51/40.75/39.63/39.63/38.84`
  - **InternVL3-8B test**: Macro/Micro/Samples/Avg/mAP = `37.54/41.85/40.50/39.97/37.67`
  - **GPT-4o test**: Macro/Micro/Samples/Avg = `43.05/54.77/56.75/51.52`
  - **写作口径**: zero-shot MLLM 有一定竞争力（GPT-4o 接近 CLIP baseline），但仍低于 trained CLIP/FDIL，说明 general MLLM prompting 不能替代 dataset-specific calibration、uncertainty-aware supervision 和 image-conditioned refinement。
- ☐ **E6 t-SNE**: FDIL vs baseline 的 intent 嵌入可分性（hard/语义相邻类）。
- ☐ **E7 ROC/PR 曲线**: 与 SOTA 对比（评估日志已有 logits）。
- ◐ **E8 效率对比扩展**: Baseline/UTD/FDIL 的自身效率已有，外部方法仍缺。
  - **正文已有**: `paper/revision_1/04_experiments.tex` 的 `tab:efficiency_comparison`。
  - **已有结果**: Baseline Params/FLOPs/Latency = `1.20M/2.40M/0.109ms`；FDIL(UTD only) = `1.20M/2.40M/0.109ms`；FDIL = `2.41M/4.93M/0.466ms`。
  - **仍缺**: IntentMLM/IntCLIP/HLEG 的同口径 Params/FLOPs/Latency；若拿不到同环境 latency，至少补 Params/FLOPs 或说明不可比。
- ◐ **E12 comparison interpretability / attribution**: 有历史 artifact，但需重算当前最终协议。
  - **历史输出**: `logs/analysis/full_data_driven_agent_evidence_verification_20260311/per_class_gain_rows.json`, `confusion_shift.json`, `case_studies.json`。
  - **历史可用线索**: per-class gain 示例包含 `NatBeauty +2.17`, `WorkILike +1.84`, `InLove +1.59`（百分比点量级）；dominant confusion 示例包含 `Attractive→EnjoyLife`, `FineDesignLearnArt-Art→CreativeUnique`, `Playful→CuriousAdventurousExcitingLife`。
  - **限制**: 这些 artifact 来自旧 evidence-verification 分支，不是 E1/E2 的 final FDIL LCS K=5 3-seed 协议；response 若引用，需标为 qualitative diagnostic 或重新生成当前版 attribution。
- ☑ **E13 threshold-free 指标 + 统计显著性**: macro/micro ROC-AUC、micro PR-AUC 数表 + paired bootstrap（5000 重采样）完成。
  - **代码入口**: `scripts/build_e13_threshold_free_significance.py`（从 healthy E2 seed20260616 checkpoint 确定性重建 4 方法 test/val 分数；s2d 环境 + meta-path stub 绕过 lightning/rich）。
  - **输出**: `logs/analysis/e13_threshold_free_significance_20260617/REPORT.md`, `e13_threshold_free_metrics.csv`, `e13_paired_significance.csv`, `summary.json`。
  - **threshold-free 表（test, seed20260616）**: baseline mAP/macroAUC/microAUC/microPR = `49.94/87.04/84.90/54.89`；UTD = `52.76/90.09/88.55/59.59`；SLR-C only = `53.18/87.43/85.59/57.45`；FDIL = `54.37/89.43/88.15/59.95`。
  - **显著性（FDIL − x，pp, 95% CI, p）**: vs baseline mAP `+4.43 [3.15,5.82] p=2e-4`、macroAUC `+2.38 p=2e-4`、microAUC `+3.24 p=2e-4`（全部显著）；vs UTD mAP `+1.61 [0.02,3.98] p=0.047`、macroAUC `−0.66 p=2e-3`、microAUC `−0.40 p=0.010`（FDIL 在 AUC 上略低，诚实写入）；vs SLR-C-only 三项均显著。
  - **写作口径**: FDIL 相对 baseline 在所有 threshold-free 指标显著领先→增益非阈值伪影；相对 UTD 仅 mAP 显著、ROC-AUC 略低→SLR-C 残差提升 average-precision/F1 而非 ROC 可分性。绝对幅度用 seed-specific，headline 仍引 E2 多种子 mAP `+3.57`。
  - **⚠️ checkpoint 完整性发现**: `distillation_slrc_lcs_topk5_20260327`（build_e3 的 `DEFAULT_FDIL_SUMMARY`、正文 “SLR-C only=18.69” 来源）把 Mar-16 `net.*` baseline 以 strict=False 载入重构后的 `StudentMLP`(`encoder.*`)→0/14 参数载入→SLR-C base 未训练；residual 补偿使 FDIL 数仍≈healthy。但 **“语义先验单独 collapse 到 ~18.6” 是该 bug 的伪影**，healthy seed 下 SLR-C-only ≈ mAP 53 / macroF1 49（与 UTD 相当）。**E3 需用 healthy seed 重跑**；E2 多种子 FDIL headline 不受影响（seeds 0317/0615/0616 base 正常）。
- ☑ **E14 VLM/text negative controls**: 自洽 retrain（当前 cache，单种子）。
  - **代码入口**: `scripts/build_e14_retrain_dissociation.py`（含 branch contribution + E14 controls + 类级 subset 解离）；`scripts/build_intervention_dissociation.py`（推理期干预，已弃用为主证据）。
  - **输出**: `logs/analysis/e14_retrain_dissociation_20260617_193954/REPORT.md`, `e14_metrics.csv`, `e14_drops.csv`, `summary.json`。
  - **结果（drop vs full）**: shuffled_rationale `−7.75 Macro`（比 no_utd 还差，灾难性）；no_utd `−4.03`；ungated `−0.62`；uniform_gate `−1.16`；no_prior `+0.56`（不降反升）；shuffled_prior `+0.43`。
  - **写入**: R2.2（`responses/3_reviewer2.tex`）+ 正文 `tab:negative_controls`，只写 rationale-teacher controls（shuffled_rationale / w/o UTD / ungated / uniform_gate）。
  - **未写入（待 (b) 决策）**: 双解离不成立（prior 在 retrain 后基本 inert，no_prior/shuffled_prior 不降反升；UTD 对两子集增益接近、未特化）。SLR-C 的定位问题留待 (b) repositioning。
  - **Jaccard(semantic vs supervisory subsets)** = 0.385。
- ☑ **E15 leakage / protocol audit**: 源码级追踪 E1/E2/E3/E14 + 阈值标定路径，四项全部 PASS。
  - **输出**: `logs/analysis/e15_leakage_protocol_audit_20260617/REPORT.md`
  - **结论**: (1) train/val/test image-id 两两零重叠（12,739 / 498 / 1,216）；(2) global/class-wise/group-wise 阈值全部仅在 validation 上 `search_*` 拟合后施加到 test（grep 确认无任何 selection 调用吃 test 数组）；(3) `_train_student` 与 `_train_residual_student` 的 best-epoch/early-stopping 仅由 `val_macro` 驱动，test 指标仅 logging；(4) 推理仅用 image features（`_predict_student`），rationale teacher 是训练期 KD 信号，test rationale/logits 仅用于评测。
  - **残留 caveat**: split 非重叠继承自官方 Intentonomy split（仅在 cache 层校验）；class-wise 阈值在 498-val 上网格搜索消耗 validation 信号，故以 threshold-free mAP 为主指标；`analyze_calibrated_decision_rule.py:115` 有一处被覆盖的 dead 0.5-threshold 计算，建议清理。
  - **已写入**: R1 “validation-only model selection” response（`responses/2_reviewer1.tex`）。

### P2 — 工作量小
- ☐ **E9** Table 5 补 Lexical+Scenario、Canonical+Scenario 两行。
- ☑ **E10** Table 6 K 非单调现象解释/中间 K 平滑曲线。
  - **已有输出**: `logs/analysis/distillation_slrc_lcs_topk5_20260327/main_comparison.csv`, `distillation_slrc_lcs_topk10_20260327/main_comparison.csv`, `distillation_slrc_lcs_topk28_20260327/main_comparison.csv`；正文 `tab:k_sensitivity_test`。
  - **结果（FDIL dynamic KD）**: K=5 Macro/Micro/Samples/Avg/Hard = `51.21/59.64/60.43/57.09/33.48`；K=10 = `48.78/57.65/58.25/54.89/29.33`；K=28 = `50.02/59.33/59.77/56.37/31.21`。
  - **写作口径**: 已可解释为 final LCS prior 下紧凑候选集更稳；若 reviewer 明确要求“曲线”，再补 K=3/15/20 等中间点。
- ◐ **E11** Table 2 Easy 子集 FDIL<HLEG 对照与解释。
  - **已有结果**: `paper/revision_1/04_experiments.tex` Table `tab:difficulty_results` 中 HLEG Easy `81.49`，FDIL Easy `81.01`，差值 `-0.48`；FDIL 在 Medium/Hard 分别为 `57.06/33.48`，明显高于 HLEG `38.99/20.22`。
  - **正文现状**: 已写“FDIL 主要改善 Medium/Hard，Hard 是 ambiguity-prone 核心指标”，但未点名回应 Easy 子集略低。
  - **仍需写入 response**: Easy 类视觉线索直接、HLEG hierarchy 对简单类别足够强；FDIL 的主要价值在 ambiguous Medium/Hard，而不是牺牲整体性能追求 Easy 上限。

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
2. **E4（EMOTIC）** 不再作为主线泛化实验推进；已有旧结果偏负，若使用只放 appendix/limitation。
3. **E13、E15、E9** 是当前剩余实验优先级：补 paired significance / AUC，完成 protocol audit，并补 R5 点名缺失的两行 prior ablation。
4. 并行写 **W1/W2/W3/W12/W14**，用 E1/E2 真实数字、E14 negative controls 和 E15 protocol audit 结果。
5. 最后清扫剩余 P2 小修 + **W13 response letter** + **W15 cross-reference/PDF audit**。

---

## 6. 风险点

- **E1 反转风险**: 公平协议下 FDIL 净增益若过小，需把卖点从"绝对 SOTA"转向"在受控条件下解耦带来稳定增益 + 理论解释 + 跨数据集泛化"。
- **新颖性**: 仅靠补实验难翻 R2，必须有 W2 的理论升格 + E3 的正面证据双管齐下。
- **E4 数据集适配风险**: EMOTIC 是 subjective multi-label affect/emotion recognition，不是 visual intent 原生 benchmark。若结果不正或任务迁移解释牵强，不放主文；最多作为 appendix/limitation 中的泛化探索，避免硬说成同任务大规模验证。
- **外部 SOTA 重跑风险**: HLEG/LabCR/PIP-Net/IntCLIP 未必都能公平替换为 CLIP ViT-L/14 特征。E1b 失败时不要卡住修稿，优先用 E1a 的 controlled in-house baselines 回应公平性。
- **artifact/leakage 风险**: 离线 VLM rationale、scenario prior、val/test logits 必须有清楚边界；若 test artifacts 曾被用于选模，需要改流程或在最终稿中完全避开相关结果。
- **页数**: 大量补实验 + 重写可能超 35 页，注意把次要消融移入 appendix。
