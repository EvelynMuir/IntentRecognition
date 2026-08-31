AE: Major revision is required, with particular attention to strengthening the theoretical justification, improving statistical validation, and adding comparisons with new baseline methods.
Reviewer #2: The manuscript describes an important and relevant problem of subjective visual understanding by explicit modeling of supervisory and semantic ambiguity in functionally decoupled manner. Compared with the original version of the manuscript, the current version demonstrates some improvements in methodological detail, experiment evaluation, metrics, presentation, and discussion. Integration of Semantic Local Reranking and Calibration with Uncertainty-guided Text Distillation is quite an interesting approach for ambiguity-aware visual intent recognition.

Nevertheless, there are still some serious scientific concerns. There is still a lack of theoretical justification of functionally separate two types of ambiguity. In spite of the inclusion of new experiments, statistical validation is still missing. Confidence intervals, hypothesis testing, and robustness to multiple training runs are necessary to prove that reported improvements are statistically significant. Moreover, the evaluation is limited to the single benchmark dataset that makes it impossible to assess generalization ability of the approach. The analysis of computational efficiency, visualization of failures, and discussion of deployment are needed to make the contribution stronger.

To summarize, the manuscript shows promising technical merit and has been improved with the revision. However, further experimental validation and scientific justification are required before reaching the standards of Pattern Recognition.


Reviewer #3: (To be filled in by the Associate editor) - Metareview from the Associate Editor (minimum 60 words):
Notes to the Reviewer : This field is mandatory. Please put here your comments explaining your ratings of the paper and suggesting improvements

Although the manuscript has been revised in response to the previous reviewer comments, the Abstract still does not fully address the main concern. My previous comment was not intended to encourage the authors to simply present more numerical values. Rather, I expected a quantitative comparison with existing methods, such as clearly stating the percentage improvement over the baseline. Presenting objective comparative metrics is essential for highlighting the significance of the proposed method and attracting readers' interest. Therefore, I believe this part still requires further revision.


Reviewer #4: While the manuscript has been revised, the current version still fails to address fundamental concerns. Besides, I have the following questions for authors to improve the manuscript.
1. There must be larger-scale datasets to better evaluate the proposed methods.
2. More comparisons with state-of-the-art Large Language Model methods (such as 2026) are needed..
3. The paper lacks theoretical analysis or deeper learning insights. It must add more value to the paper if more theoretical analysis were conducted, e.g., from information-theoretical or feature selection perspectives.
4. The overall framework is over-complicated and lacks clarity in design justification. Multiple modules are introduced, but their individual contributions and interactions are not clearly disentangled. The ablation studies, while present, are not sufficient to demonstrate that each component is necessary rather than redundant, making the method appear over-engineered.
5. The manuscript must require careful proofreading, such as [7] The manuscript must require careful proofreading and [9] Computer Vision -ECCV 2024.



Reviewer #5: The author has comprehensively addressed my concerns, leaving no further questions.


Reviewer #6: The paper presents a functionally decoupled framework for visual-intent recognition that addresses supervisory and semantic ambiguity using structured semantic priors and uncertainty-aware distillation. The manuscript is well written, clearly structured, and the proposed contribution is reasonably good and relevant to the field. However, the performance improvement over comparable state-of-the-art methods is relatively small, which limits the strength of the empirical contribution. Although the authors report results using mean ± standard deviation over three runs, this alone is not sufficiently convincing for establishing statistical superiority. An explicit paired t-test, applied consistently against the strongest state-of-the-art methods, should therefore be provided to demonstrate that the observed improvements are statistically significant rather than caused by random variation.



%ATTACH_FOR_REVIEWER_DEEP_LINK INSTRUCTIONS%

%REVIEW_QUESTIONS_AND_RESPONSES%

EiC: While you are revising your paper, here is a list of points worth checking, which we find author's overlook. I will check that these are adhered to before your paper is approved for publication, assuming the revision satisfies the Associate Editor and Reviewers.

a) Take a careful look at your bibliography and they cover the state of the art. Missing references from last and current year most probably would mean you are missing the state of the art and the revision process can be delayed being asked to update it. Please do not make excessive citation to arXiv papers, but substitute them with their peer-reviewed versions, or papers from a single conference series. Do not cite large groups of papers without individually commenting on them. So we discourage " In prior work [1,2,3,4,5,6] …". Your bibliography in the final version after the revision still should be between 35-55 items.

b)  Please make sure the revised version is relevant to the readership of the Pattern Recognition field. To this end, please make sure you cite RECENT  work from the field of pattern recognition not only the Pattern Recognition journal. 

c) Although the revision could lead to extending your article, it still can not exceed the page limits or violate the format, i.e. double spaced SINGLE column with a maximum of 35 pages for a regular paper and 40 pages for a review.


At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_PR#

To ensure this email reaches the intended recipient, please do not delete the above code
