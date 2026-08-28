1. 核对正文及response的统一性

3. **不要叫 Table 7 “interaction experiment”。**
   你现在做的是 2×2 stratification + 每个 cell 内的 paired test，并没有正式检验 supervisory × semantic 的 statistical interaction。因此正文 caption 的 “2 × 2 ambiguity interaction”和 response 里大量的 “interaction experiment” 容易被统计审稿人抓住。

   推荐统一成 **“2×2 ambiguity-stratified analysis”**。同时 response 里这句建议直接删掉：

   > “Two components that were redundant with each other could not respond to orthogonal stratifications of the data in this way.”

   这句话逻辑上并不成立，而且两个 ambiguity axis 也没有证明是“orthogonal”。比较稳妥的是：

   > “These patterns are consistent with distinct functional sensitivities of the two modules.”

   这已经足够回答 reviewer。 

4. **理论部分本身现在是成立的，但 response 把理论意义说过头了。**
   Eq. (15) 的 identity 没问题；问题在于你从“different functionals”跳到了：

   > “therefore require different learning treatments rather than one shared mechanism”
   > 以及正文：
   > “pose different learning problems rather than admitting a single shared mechanism.”

   信息分解只能**motivate** functional decoupling，不能证明任何 unified mechanism 都不可能同时处理它们。建议全部把 `require` / `rather than admitting` 改成：

   > “motivate treating them with distinct mechanisms”

   然后用 Table 6 说“在我们测试的 unified control 下，decoupling 更好”。理论负责 justification，实验负责 empirical support，不要让理论承担“证明 necessity”的任务。

   还有 response 中：

   > “the required non-redundancy is visible in the prior ablation (Table 8), where no single source dominates.”

   这个逻辑不对。“没有单一 prior source dominate”证明的是不同 prior source 有互补性，**不是**证明 (S) 在给定 (B) 后 non-redundant。这里最好改成“SLR-C 相对 baseline 的性能提升与 (S) 提供额外 label-relevant information 的解释一致”。

5. **效率部分要明确是 feature-level/head-level cost。**
   response 的 Table 17 caption 其实写得更严谨：“Feature-level inference cost on frozen CLIP features”；但正文 Table 17 现在只写：

   > “Inference efficiency comparison on an RTX 4090.”

   这样很容易让人以为 0.466 ms 是完整 image→prediction latency。Discussion 里又据此说适合 high-throughput，会进一步放大这个误解。

   建议正文 caption 恢复：

   > “Feature-level inference cost on pre-extracted CLIP ViT-L/14 features...”

   同时删掉：

   > “HVU-CLIP releases no code and we expect a comparable cost”

   `we expect` 完全没有必要，也没数据支持。IntentMLM/CoT4Intent 的 “orders of magnitude more expensive” 如果没有统一测量，最好也改成：

   > “require MLLM inference and are therefore not included in the measured feature-level comparison.”

   response 里还有一句：

   > “At inference UTD contributes nothing”

   这也不准确——UTD 的训练作用保留在 student 参数里，只是**不增加 inference-time computation**。应该这么写。 

6. **ResNet101 那句归因没有证据。**
   正文：

   > “FDIL is better on Macro, Samples, and Avg. F1 and lower on Micro F1, which we attribute to its handling of the highly ambiguous hard categories.”

   但 HVU-CLIP 根本没有 Hard F1，所以你不能根据这张表把两者差异归因到 hard categories。直接停在事实即可：

   > “FDIL is higher on Macro, Samples, and Avg. F1, while HVU-CLIP is higher on Micro F1.”

   会显得更专业。

7. **Case Study 最后一句很像在为 failure 辩护。**
   现在是：

   > “illustrating the intrinsic uncertainty of the task and the rationality of the model’s decisions.”

   “模型虽然错了但它是 rational 的”特别像防御性解释。建议：

   > “illustrating that low-agreement samples can admit semantically plausible alternative predictions.”

   客观描述现象即可。

8. **Discussion/Conclusion 对 transfer 的总结还是太用力。**
   比如：

   > “Generalization beyond the intent taxonomy is likewise bounded in magnitude rather than in kind.”
   > “What does carry across all three transfers is the functional decomposition itself...”
   > “therefore calls for priors written for that space rather than for a different division of labour.”

   这些句子写得很漂亮，但数据其实没有这么确定：LDL 的 effect 很小、metric-dependent，而且 Flickr 和 Emotion6 对 full FDIL 的显著指标并不一致。这里最好写成：

   > “Across the transfer studies, the component-level tendencies are broadly consistent, although the overall gains are small and metric-dependent.”

   最后一段：

   > “the key challenge ... lies in resolving supervisory and semantic ambiguity rather than in extracting low-level visual features”

   也太宽泛，而且你自己的 ResNet101 实验其实证明 backbone quality 仍然重要。建议改成：

   > “These results suggest that, given strong visual representations, explicitly modeling supervisory and semantic ambiguity can further improve subjective intent recognition.”

   这样几乎没有攻击面。

9. **response 里最需要删的不是科学 caveat，而是“辩护式措辞”。**
   我会删/改这些表达：

   * “new experiments ... **rather than with rewriting**”
   * “We **deliberately** scope the claim ... which we report **rather than obscure**.”
   * “There are two modules, **not many**”
   * “the **largest benchmark we could use**...”
   * “compatible ... benchmarks are **scarce**”
   * “We implemented **exactly** the requested test”
   * “This **establishes statistical superiority**...”
   * “the **sole non-trivial regression**...”
   * closing 的 “a **demanding** and constructive second round”

   这些都不是错，但组合起来会让 response 有一种“我提前预判你会攻击我，所以我先解释”的感觉。尤其 **“rather than obscure”** 我强烈建议删掉，完全没有收益。

   同时“35-page limit”目前被重复解释了很多次。General Response 说一次、EiC page-limit comment 再说一次就足够了。其他 reviewer 下直接说“full statistics are provided in Table R1”即可。

10. **LDL response 里有两个不严谨的表述。**
    一是：

> “SLR-C moves the pointwise-fit metrics (KLD, cosine, µ) **and only those**”

Table R2 中其他指标也有数值变化，只是显著性/主要趋势不同，所以 `and only those` 应改成 `primarily affects`。

二是：

> 四个 LDL objective “moving every metric by at most (7\times10^{-4})”

这句话和表里的 µ(%) 不严格对应，而且完全没必要精确到这个程度。直接写：

> “produce only negligible changes relative to the matched baseline”

更安全也更清楚。

11. **还有几个小但应该修的 proofreading 问题。**
    首页单位应为 **Beijing University of Posts and Telecommunications**，不是 “Post and Telecommunications”；Sec. 4.5.2 有 `distribution.The` 缺空格；Keywords 的大小写建议统一；`source benchmark` 用在 EMOTIC 上也不太自然，改成 `transfer benchmark`。另外 Figure 3 的 silhouette 如果是在 **t-SNE 2D embedding** 上计算的，我建议不要作为定量证据；如果是在原始 decision-score space 算的，应明确写出来。

总体判断是：**现在不需要再增加实验了，主要需要“收口”。** 目前最大的风险不是 reviewer 觉得你没回答，而是你为了把每个质疑都堵死，反而把一些原本只需要“evidence supports”写成了“establishes / requires / could not / must”。这会给 reviewer 新的逻辑攻击点。

如果按优先级，我会先改 **Abstract → Table 5 → interaction措辞 → theory中的 require/necessity → efficiency → Discussion/Conclusion**。Response 则主要做减法，特别是 Reviewer 4 Comment 4 那三页，完全可以少掉大约 25% 的自我辩护文字，论证反而会更有力量。
