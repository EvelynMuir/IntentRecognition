Reviewers' comments:

Based on the detailed feedback from reviewers, the manuscript requires major revision. The authors should address the following major concerns: insufficient novelty, incomplete experimental validation (including unfair comparisons, missing baselines, unexplained anomalies, and lack of large-scale evaluation), presentation and clarity issues (including the introduction, figures, and proofreading), limited theoretical depth, and reproducibility.

Reviewer #1: This manuscript studies visual intent recognition under two forms of ambiguity: semantic overlap among intent categories and supervisory ambiguity from annotator disagreement. I find the problem important and the overall direction promising. The proposed FDIL framework is also empirically interesting, particularly because it improves over a strong CLIP ViT-L/14 baseline and gives larger gains on hard intent categories. However, the current version needs substantial revision before it can be judged ready for publication. The main concerns are the fairness and interpretability of the comparisons, the strong effect of class-wise thresholding, the dependence on offline VLM-generated rationales and semantic priors, and incomplete reproducibility details. Please provide controlled baselines under the same backbone and thresholding protocol, report variance over runs, clarify validation-only model selection, release or document generated artifacts, and tone down claims about functional decoupling and SOTA performance unless the ablations isolate these factors more cleanly.


Reviewer #2: The manuscript addresses an important and difficult issue in visual intention recognition and suggests an interesting framework combining semantic priors with uncertainty-driven distillation. The manuscript is well-written, technically sound, and experimentally validated on the Intentonomy dataset.

However, the major drawback is related to insufficient novelty of the technical contribution. Most of the contributions proposed in the paper relate to previously discussed topics within vision-language modeling, semantic alignment, and uncertainty-aware learning. Although the idea of distinguishing between semantic uncertainty and supervision uncertainty is quite appealing, it does not constitute a new theoretical principle or algorithmic idea.

It appears that the results are based mainly on the correct combination of existing methods rather than the development of a novel approach itself. As a result, the paper fails to satisfy the novelty criterion required for Pattern Recognition publication.


Reviewer #3: (To be filled in by the Associate editor) - Metareview from the Associate Editor (minimum 60 words):
Notes to the Reviewer : This field is mandatory. Please put here your comments explaining your ratings of the paper and suggesting improvements

1. The authors are encouraged to clearly state, in the Abstract, the quantitative improvements achieved by the proposed FDIL method compared to existing approaches.

2. In Section 3.3.1, Semantic Prior Construction, the authors should provide a more detailed justification for the selected scenarios, including their representativeness and the expected outcomes in conditions not covered by the presented scenarios.

3. Table 2 shows that FDIL achieves lower performance than the conventional HLEG method on the Easy subset, yet this observation is not discussed. The authors should explain the possible reasons for this result and clarify whether certain characteristics of the Easy subset contribute to the similar performance between the two methods. Further discussion of the key observations in the table is recommended.


Reviewer #4: This paper proposes a FDIL (Functionally Decoupled Intent Learning) for visual intention understanding. The experiment results show the efficacy of the proposed algorithm compared with other deep learning algorithms.
Besides, I have the following questions for authors to improve the manuscript.
1. The manuscript must require careful proofreading, such as [7] IEEE Trans. Affect. Comput., and so on.
2. The paper is not very clear but only claims that " A central difficulty of this task lies in two complementary forms of ambiguity: supervisory ambiguity, caused by subjective annotation over plausible intentions, and semantic ambiguity, caused by semantically adjacent and overlapping intents." Please give more analysis about these disadvantages in the existing visual intention understanding.
3. The clear purpose of work the proposed work is lacking in the introduction section.
4. The introduction should be re-discussed and extended. I think that more Large Language model-based visual intention understanding methods should be mentioned. In the introduction, this section is hard for readers to understand its latest development.
5. You may consider including a table of notations.
6. There should be larger-scale datasets to better evaluate the proposed methods.
7. Experimental evaluation is not sufficient. More comparisons with state-of-the-art Large Language Model methods (such as 2025 and 2026) and t-SNE are needed.
8. Please use some ROC-based curves for a clear evaluation of the proposed method with other state-of-the-art methods.
9. A new section called "Discussion" could be added. The authors can discuss the main findings and also the important advantages and disadvantages of the proposed model.
10. It would add more value to the paper if more theoretical analysis were conducted, e.g., from information-theoretical or feature selection perspectives.
11. A comparative analysis of model parameters, FLOPs, or inference times is necessary.



Reviewer #5: This manuscript presents a novel perspective on visual intention recognition by functionally decoupling two complementary sources of ambiguity: semantic ambiguity (inter-class similarity and overlapping intent boundaries) and supervisory ambiguity (annotator disagreement and subjective labels). This functional decoupling is theoretically well-motivated. The authors argue that addressing both types of ambiguity through a single mechanism is suboptimal, as they interfere with distinct functional components of the system. The proposed modules, Semantic Local Reranking and Calibration (SLR-C) and Uncertainty-guided Text Distillation (UTD), are architecturally clean and complementary. The residual student design built upon semantic priors is elegant and avoids heavy architectural overhead.

Weakness
1. Figure 2 caption inconsistency: The caption refers to "(Top) UTD" and "(Bottom) SLR-C", but the figure shows "A" as the top and "B" as the bottom. Please revise the caption or the figure to ensure consistent correspondence.
2. Missing experimental settings in Table 5:: Two additional configurations should be included: (i) Lexical + Scenario, and (ii) Canonical + Scenario.
3. Unexplained anomaly in Table 6: Provide a justification for why K=5 K=28 yield better performance than K=10.
4. Typo in Appendix tables: In Table B.1 and Table B.2 (page 35), "FDIR" should be corrected to "FDIL".



%ATTACH_FOR_REVIEWER_DEEP_LINK INSTRUCTIONS%

%REVIEW_QUESTIONS_AND_RESPONSES%

EiC: While you are revising your paper, here is a list of points worth checking, which we find author's overlook. I will check that these are adhered to before your paper is approved for publication, assuming the revision satisfies the Associate Editor and Reviewers.

a) Take a careful look at your bibliography and they cover the state of the art. Missing references from last and current year most probably would mean you are missing the state of the art and the revision process can be delayed being asked to update it. Please do not make excessive citation to arXiv papers, but substitute them with their peer-reviewed versions, or papers from a single conference series. Do not cite large groups of papers without individually commenting on them. So we discourage " In prior work [1,2,3,4,5,6] …". Your bibliography in the final version after the revision still should be between 35-55 items.

b)  Please make sure the revised version is relevant to the readership of the Pattern Recognition field. To this end, please make sure you cite RECENT  work from the field of pattern recognition not only the Pattern Recognition journal. 

c) Although the revision could lead to extending your article, it still can not exceed the page limits or violate the format, i.e. double spaced SINGLE column with a maximum of 35 pages for a regular paper and 40 pages for a review.