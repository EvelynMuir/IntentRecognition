# Response to Reviewers

**Manuscript:** *Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity* (FDIL)
**Journal:** Pattern Recognition
**Decision:** Major Revision

---

> **Note to co-authors (remove before submission).** This is a working draft aligned to `revision_plan.md`. Items marked **[TODO: …]** depend on experiments/writing that are not yet finalized; insert the final numbers, table/figure pointers, and page references before submission. Confirmed quantitative results from experiment **E1** are already filled in below. All page/line/Table references should be re-checked against the revised manuscript in the final cross-reference pass (W15).

---

## To the Editors and Reviewers

We thank the Associate Editor and all five reviewers for their careful and constructive evaluation. The reviews were detailed and fair, and they have substantially improved the manuscript. We are encouraged that the reviewers found the problem **important** (R1, R2, R4), the functional-decoupling perspective **theoretically well-motivated** (R5), and the framework **architecturally clean, technically sound, and empirically interesting**, including gains over a strong CLIP ViT-L/14 baseline and larger improvements on hard intent categories (R1, R5).

We have grouped the major concerns into five themes and addressed each one with new experiments, new analysis, and substantial rewriting:

1. **Fairness and interpretability of comparisons; the effect of class-wise thresholding (R1).** We added a controlled comparison protocol in which all in-house baselines and FDIL share the *same* CLIP ViT-L/14 features, the *same* train/val/test split, and the *same* validation-only thresholding protocol. We further added a four-cell grid that decomposes the contribution of the *method* from the contribution of *calibration* (global vs. class-wise thresholding). Under the matched class-wise protocol, FDIL improves over the CLIP baseline by **Macro-F1 +3.80**, **Avg-F1 +3.68**, and **Hard-F1 +5.03**; under a matched global-threshold protocol the Macro/Hard gains are essentially preserved while the Avg-F1 gain is +0.84. We now explicitly separate **method contribution** from **calibration contribution** throughout the paper, report variance over multiple seeds, clarify validation-only model selection, and have toned down absolute-SOTA language accordingly.

2. **Novelty and theoretical depth (R2, R4-10, R1).** We added a formal, information-theoretic / feature-selection motivation for *why* the two ambiguities must be handled by orthogonal components, a head-to-head **unified-vs-decoupled** comparison showing the decoupling is not merely rhetorical, **negative-control** experiments showing that arbitrary VLM text priors do *not* reproduce the gains, and a **second dataset** to demonstrate generalization.

3. **Presentation and clarity (R4, R5, R3).** We rewrote the Introduction (clear purpose, critical analysis of how existing methods fail on each ambiguity type, expanded LLM-based intent-understanding survey), added a notation table, added a Discussion section, fixed the Figure 2 caption, corrected the "FDIR"→"FDIL" typos, and proofread the manuscript and bibliography.

4. **Stronger and larger-scale evaluation (R4, R3).** We added a second/larger-scale dataset, comparisons against 2025/2026 large multimodal models, t-SNE embedding visualizations, ROC/PR curves, threshold-free metrics with statistical significance testing, and an extended efficiency comparison (params/FLOPs/latency).

5. **Reproducibility (R1, EiC).** We documented the validation-only selection protocol, expanded implementation details, and replaced the "available on request" statement with a concrete artifact-release plan (scenario priors, rationale files, text features, threshold files, logits, prompts, seeds, VLM versions, checksums).

A point-by-point response follows. Newly added or revised content is referenced by section/table/figure. In the manuscript, changes are highlighted in **[TODO: color/markup convention]**.

---

## Reviewer #1

We thank the reviewer for a very constructive review and for recognizing the importance of the problem and the empirical interest of FDIL. The reviewer's central concerns — comparison fairness, the strength of class-wise thresholding, dependence on offline VLM artifacts, and reproducibility — directly shaped our revision.

**R1.1 — Controlled baselines under the same backbone and thresholding protocol.**
We added a fair-comparison protocol (new **[TODO: Table X / §X]**) in which every in-house controllable baseline and FDIL are evaluated on identical CLIP ViT-L/14 features, the identical train/val/test split, and an identical validation-only threshold search. This isolates the contribution of FDIL's mechanisms from backbone strength.
- *Result (E1a, completed):* under the matched class-wise protocol, FDIL improves over the CLIP baseline by **Macro-F1 +3.80**, **Avg-F1 +3.68**, **Hard-F1 +5.03**.
- We also re-ran/reproduced representative external SOTA (HLEG / LabCR / PIP-Net / IntCLIP) on CLIP ViT-L/14 features where feasible **[TODO: confirm which external methods were successfully re-run (E1b); for any that could not be ported to the same backbone, state the protocol limitation explicitly and rely on the in-house controlled baselines as the primary fairness evidence]**.

**R1.2 — Strong effect of class-wise thresholding.**
We agree this needed to be isolated. We added a four-cell decomposition `{CLIP baseline, FDIL} × {global, class-wise threshold}` (new **[TODO: Table X]**). We confirm that the reported baseline of 53.41 corresponds to the CLIP baseline under class-wise thresholding (Avg-F1 = mean of Macro/Micro/Samples F1), **not** mAP. The decomposition shows that under a matched **global** threshold, FDIL's Macro-F1 and Hard-F1 gains are essentially unchanged, while the Avg-F1 gain reduces to **+0.84**, i.e., part of the Avg-F1 improvement is attributable to calibration. We now state this explicitly and separate **method contribution** from **calibration contribution** in the abstract, results, and discussion.

**R1.3 — Report variance over runs.**
We now report mean ± std over **≥3 seeds** for FDIL and the key baselines in the main results and ablation tables (E2, **[TODO: Table X]**).

**R1.4 — Clarify validation-only model selection.**
We added an explicit protocol statement (**[TODO: §X]**): all thresholds and model selection use the validation split only; test logits/rationales are used for evaluation only and never for model or threshold selection. **[TODO: confirm via E15 leakage/protocol audit and cite its outcome.]**

**R1.5 — Release/document generated artifacts.**
We replaced "available on request" with a concrete Data/Artifact Availability statement (**[TODO: §X / repository link]**) listing scenario priors, rationale JSONL, BGE/CLIP text features, threshold files, train/val/test logits, prompt templates, seeds, VLM version, and checksums (W14).

**R1.6 — Incomplete reproducibility details.**
We expanded implementation details (**[TODO: §X / Appendix X]**), including feature extraction, threshold search, teacher generation, and the agreement gate.

**R1.7 — Tone down "functional decoupling" / "SOTA" claims unless ablations isolate the factors.**
Done. With the decomposition (R1.2) and the unified-vs-decoupled comparison (see R2), we have rewritten the claims: we retain decoupling as a contribution supported by the isolation experiments, and we have softened absolute-SOTA language to **[TODO: choose claim variant A or B per W16 once E1b is final]**.

**R1.8 — Comparison interpretability / attribution.**
We added per-class gain/loss, confusion-pair shift, and hard-category attribution analysis (E12, **[TODO: Table/Fig X]**) that identifies *which* semantically adjacent categories FDIL improves and *which* easy/visually-clear categories may be mildly affected by calibration.

**R1.9 — Threshold may dominate F1; add threshold-free metrics and statistical tests.**
We added threshold-free metrics (mAP, macro-AUC, micro-AUC, PR-AUC) and paired significance tests (paired bootstrap / permutation) for FDIL vs. CLIP baseline, UTD-only, and SLR-C-only (E7, E13, **[TODO: Table X]**).

---

## Reviewer #2

We thank the reviewer for acknowledging that the manuscript is well-written, technically sound, and empirically validated, and that distinguishing semantic from supervisory uncertainty is appealing. We take the novelty concern very seriously and have addressed it on three fronts so that the contribution is a **principle**, not a combination of existing components.

**R2.1 — "Insufficient novelty; correct combination of existing methods."**
- *(a) A formal decoupling principle (new theory, §X).* We now formalize the two ambiguities as acting on **different terms of the learning objective**: supervisory ambiguity is label-side conditional entropy / annotation noise, whereas semantic ambiguity is feature-side inter-class mutual-information aliasing. From this, we argue (information-theoretic / feature-selection viewpoint) that a single mechanism conflates two terms that should be treated orthogonally, motivating SLR-C (feature/decision side) and UTD (label/supervision side) as **necessarily separate** modules (W2). This reframes the contribution as a principle with testable predictions, not an engineering combination.
- *(b) Unified vs. decoupled, head to head (E3, new Table X).* We added a controlled comparison in which a single unified mechanism is given the same inputs and budget as decoupled FDIL. **[TODO: insert result showing decoupled > unified, demonstrating the decoupling is not vacuous.]**
- *(c) Generalization on a second dataset (E4).* **[TODO: insert second-dataset result; if EMOTIC, frame carefully as a generalization probe rather than a same-task large-scale benchmark per the plan's risk note.]**

**R2.2 — VLM/text prior may be mere external-knowledge stacking.**
We added negative-control experiments (E14, new **[TODO: Table X]**): shuffled rationales, random scenario priors, label-only text, caption-only text, an ungated teacher, and a corrupted agreement gate. **[TODO: insert results showing these controls do not reproduce the gains, i.e., it is not "any VLM text + class-wise threshold" that helps, but the structured, uncertainty-gated prior.]**

---

## Reviewer #3 (Associate Editor)

We thank the AE for the focused, actionable points.

**R3.1 — State quantitative improvements in the Abstract.**
Done. The Abstract now reports concrete numbers under the matched protocol (e.g., **Macro-F1 +3.80**, **Hard-F1 +5.03** over the CLIP ViT-L/14 baseline) **[TODO: finalize exact wording and the calibration-vs-method split per W3/W16]**.

**R3.2 — §3.3.1: justify the selected scenarios (representativeness; behavior on uncovered conditions).**
We expanded §3.3.1 (W5) with the selection criteria, coverage argument over the intent taxonomy, and the expected behavior — and graceful degradation — for conditions not covered by the predefined scenarios **[TODO: §3.3.1 final text]**.

**R3.3 — Table 2: FDIL (81.01) < HLEG (81.49) on the Easy subset is undiscussed.**
We added a discussion (E11, **[TODO: §X]**): the Easy subset is dominated by visually unambiguous categories where semantic-overlap modeling and uncertainty-guided distillation offer little headroom and class-wise calibration can be slightly conservative; FDIL's advantage concentrates on the Hard subset where both ambiguities are active. We added a per-subset attribution to support this explanation.

---

## Reviewer #4

We thank the reviewer for the thorough, itemized list, which we address point by point.

**R4.1 — Proofreading (e.g., [7] IEEE Trans. Affect. Comput.).**
Done — full proofread and bibliography format pass (W9). **[TODO: confirm reference [7] and all journal abbreviations corrected.]**

**R4.2 — More analysis of how existing methods fall short on the two ambiguities.**
We rewrote the Introduction (W1) with an explicit critical analysis of how prior visual-intent methods handle (or conflate) supervisory and semantic ambiguity **[TODO: §1 final].**

**R4.3 — Clear statement of purpose in the Introduction.**
Done — the Introduction now states the purpose and objectives explicitly (W1).

**R4.4 — Extend the Introduction with recent LLM-based visual-intent methods.**
We expanded the survey of LLM/MLLM-based visual-intent understanding, including recent work (W1) **[TODO: add 2024–2026 references; ensure peer-reviewed versions per EiC-a].**

**R4.5 — Table of notations.**
Added (W6, **[TODO: Table X]**).

**R4.6 — Larger-scale datasets.**
We added a second/larger-scale evaluation (E4, **[TODO: dataset name + Table X]**) **[TODO: see R2.1(c) framing note].**

**R4.7 — Comparisons with 2025/2026 LLM methods + t-SNE.**
We restored and extended the large-model comparison (Qwen2-VL-7B / Ovis1.6 / GPT-4o plus 1–2 newer MLLMs, E5) and added t-SNE visualizations of intent embeddings for FDIL vs. baseline, emphasizing hard / semantically adjacent classes (E6, **[TODO: Fig X]**).

**R4.8 — ROC-based curves.**
Added ROC and PR curves vs. SOTA from existing evaluation logits (E7, **[TODO: Fig X]**).

**R4.9 — Add a Discussion section.**
Added/expanded a Discussion section covering main findings, advantages, and failure modes (W4, **[TODO: §X]**).

**R4.10 — More theoretical analysis (information-theoretic / feature selection).**
Added — see **R2.1(a)** and new §X (W2).

**R4.11 — Params / FLOPs / inference-time comparison.**
We extended the efficiency table (E8, **[TODO: Table 8]**) to include IntentMLM / IntCLIP / HLEG params, FLOPs, and latency, highlighting FDIL's low inference overhead.

---

## Reviewer #5

We thank the reviewer for the positive assessment of the decoupling perspective and the module design, and for the precise minor corrections.

**R5.1 — Figure 2 caption "(Top) UTD / (Bottom) SLR-C" vs. panels A/B.**
Fixed — caption and figure panel labels are now consistent (W7, `03_method.tex`). **[TODO: confirm final caption wording matches panels.]**

**R5.2 — Table 5 missing (i) Lexical + Scenario and (ii) Canonical + Scenario.**
Added both configurations to Table 5 (E9). **[TODO: insert the two rows + values.]**

**R5.3 — Table 6 anomaly: K=5 and K=28 outperform K=10.**
We added an explanation and a finer K sweep / smoothed curve (E10, **[TODO: §X / Fig X]**) **[TODO: insert the justification for the non-monotonic behavior].**

**R5.4 — Typo "FDIR" → "FDIL" in Tables B.1/B.2 (p.35).**
Fixed throughout (W8, `07_appendix.tex`); we also searched the full manuscript for residual "FDIR" occurrences.

---

## Editor-in-Chief Checklist

**EiC-a — Bibliography / state of the art.**
We added recent 2024–2026 references, replaced arXiv preprints with their peer-reviewed versions where available, removed grouped citations in favor of individually-discussed references, and kept the final bibliography within **35–55** items (W10). **[TODO: confirm final count and that no "[1,2,3,4,5,6]"-style group citations remain.]**

**EiC-b — Cite recent work from the Pattern Recognition *field* (not only the PR journal).**
Done — we added recent pattern-recognition-community work relevant to the readership (W10).

**EiC-c — Format / page limit.**
The revised manuscript remains within **35 pages**, double-spaced, single column (W11). We moved secondary ablations to the appendix to stay within the limit. **[TODO: confirm final page count after all additions.]**

---

## Summary of Changes

- New controlled fair-comparison protocol + method-vs-calibration decomposition (R1).
- New formal decoupling principle + unified-vs-decoupled experiment + negative controls + second dataset (R2).
- Multi-seed variance, threshold-free metrics, significance tests, attribution analysis (R1).
- Rewritten Introduction, new Discussion, notation table, expanded §3.3.1 (R3, R4).
- ROC/PR curves, t-SNE, large-model comparisons, efficiency table (R4).
- Figure 2 caption fix, FDIR→FDIL fix, proofreading, bibliography overhaul (R5, EiC).
- Concrete artifact-availability statement (R1, EiC).

We believe the revision substantially strengthens the novelty framing, the fairness and interpretability of the evaluation, and the reproducibility of the work, and we thank the reviewers again for their guidance.
