| 二审问题                                 | 评价                      | 目前风险  |
| ------------------------------------ | ------------------------- | ----- |
| R2：缺乏两种 ambiguity 的理论依据              | **基本回答，但理论映射有漏洞**         | **高** |
| R2/R6：CI、t-test、多次训练                 | **回答得很好**                 | 低     |
| R2：single benchmark/generalization   | **回答了**                   | 中     |
| R4：larger-scale dataset              | **严格说没有回答**               | **高** |
| R4：2026 / recent SOTA                | **回答充分**                  | 低     |
| R4：framework 过复杂、模块必要性               | **明显增强，但“necessity”仍稍过度** | 中     |
| R4：efficiency / failure / deployment | **基本充分**                  | 低     |
| AE：Abstract 要相对提升                    | **回答了，但有内部矛盾**            | 中高    |
| EiC：recent bibliography / 35–55 refs | **基本满足**                  | 低     |
| Proofreading                         | **仍有几处明确错误**              | 中     |

### 最需要改的 7 个地方

1. **理论部分最大的逻辑问题：SLR-C 不可能补回 (I(Y;X\mid Z))。**

你现在 Eq. (15) 写：

[
H(Y_c|Z)=H(Y_c|X)+I(Y_c;X|Z),
]

然后把第二项解释成 “information discarded by (\phi)”，并进一步把 SLR-C 和这一类 semantic insufficiency 联系起来。公式本身完全正确。

但你的 SLR-C prior 实际是

[
s_c^{prior}=\operatorname{sim}(f_{\rm img}(x),e_c),
]

而你定义的 (Z=f_{\rm img}(X))。换句话说，**SLR-C 的 prior (S) 也是 (Z) 的函数**。

所以如果某些信息已经被 (Z) 丢掉了，也就是 (I(Y;X|Z))，**SLR-C 不可能重新获得这些信息**。这是目前理论中最容易被理论 reviewer 一眼指出的地方。

真正与你方法一致的是：

[
H(Y|B)=H(Y|Z)+I(Y;Z|B),
]

因为 baseline logits (B) 是 (Z) 的压缩，而 SLR-C 可以利用 **(Z) 中存在、但 baseline logits (B) 没有充分利用的信息**。

甚至可以写成一个更完整的 exact identity：

[
H(Y|B)
======

H(Y|X)
+
I(Y;X|Z)
+
I(Y;Z|B).
]

你的 frozen backbone 不处理第二项；**SLR-C 处理的是最后一项——decision/readout insufficiency**。这样就完全自洽了。

同理，UTD 也不能“降低” (H(Y|X))，因为你自己已经把它定义成 irreducible disagreement。UTD 能做的是：**在这种不可约主观性存在时，减少有限标注造成的 target-estimation variance / 避免对不可靠 hard target 过拟合**。所以 response 里“two deficits ... admit corrective mechanisms”最好改成“require different learning treatments”，不要暗示两个 entropy term 都能被方法减少。

还有一句尤其建议删掉：

> “at (\omega=1), where supervision is effectively noiseless”

(\omega=1) 只是有限 annotator 全部一致，不等于真实 population supervision 无噪声。并且你的 (g^*) 只有在额外假设 (\sigma_v^2(1)=0) 时才真的等于 0。把它改成 **“maximally reliable observed supervision”** 会安全很多。

---

2. **Reviewer 4 要的是 “larger-scale datasets”，而你加的两个数据集其实都更小。**

Intentonomy 约 14.4k images；你 response 自己写：

* Flickr-LDL：11,150
* Emotion6：1,980

所以这不能严格回答：

> “There must be larger-scale datasets.”

现在 response 是：

> “We agree ... Flickr-LDL (11,150) and Emotion6 (1,980) ... We chose these to vary a different axis rather than simply to add images.”

这个理由能回答 **generalization breadth**，但不能回答 **larger-scale**。

这是我认为除理论外最大的审稿风险。

如果版面和实验条件已经不允许再跑真正更大的数据集，我反而建议**不要假装它满足了 larger-scale**。改成更诚实的：

> Although these benchmarks are not individually larger than Intentonomy, they directly address evaluation breadth by testing the framework under a distinct subjective label-distribution formulation...

然后解释 compatible large-scale subjective soft-label benchmarks 很有限。

这样至少 reviewer 不会觉得你在“换概念”。目前的 “We agree” → 紧接着两个更小数据集，会很显眼。

---

3. **LDL transfer 的方法协议在正文里写得不够，甚至有 formulation gap。**

正文先明确说 LDL 输出位于 probability simplex，不再做 threshold。 但原 FDIL 方法里：

* UTD 是 Bernoulli KL；
* (y^{hard}) 是 multi-label binary；
* (\omega) 是 positive labels 的 minimum agreement；
* 每一类是独立 Bernoulli。

这些和 LDL 的 simplex distribution **不是同一个输出 geometry**。

Response 甚至进一步说用了：

> “out-of-fold teacher predictions”

但我在正文的 Flickr-LDL / Emotion6 protocol 里没有看到具体说明这一套是怎么适配的。

至少正文应该用很短的一段写清：

* LDL 中 (\omega) 到底怎么从 distribution / annotator votes 得到；
* UTD 的 student/teacher loss 是 Bernoulli KL 还是 categorical KL；
* 是否仍有 hard label；
* SLR-C 如何作用于 distribution logits；
* train/val/test 怎么分；
* OOF teacher 怎么生成；
* 哪些 hyperparameters 保持 Intentonomy 不变。

否则 reviewer 很容易问：**“你说 prediction geometry 变了，但你的 method definition 没跟着变。”**

---

4. **Abstract 的 “28.0% hard-subset SOTA” 现在和你自己的 GPT-5.6 表存在内部冲突。**

Abstract 写：

> “FDIL leads the state-of-the-art by 6.1% relative average F1 and 28.0% on the hard subset.”

但 Table 2 里：

* FDIL Hard = **35.02**
* GPT-5.6 Sol Hard = **36.19**

也就是说，你自己的新增 comparator 在 Hard 上已经超过 FDIL。

你这里的 28.0% 实际应该是相对**此前 task-specific / published intent method**（例如 IntentMLM 27.39）而言。

所以一定要限定：

> **“FDIL improves over the strongest prior task-specific method by 6.1% relative Avg. F1 and 28.0% relative Hard F1.”**

或者：

> “Among prior task-specific intent-recognition methods, ...”

否则“state-of-the-art on hard subset”已经被自己的 Table 2 否定。

另外 response 说：

> “Both percentages are computed directly from the main comparison table.”

也不准确。Avg 来自 Table 1，而 Hard 的 prior comparison 是另一张 difficulty table，不是 Table 1。

---

5. **Response 里 HVU-CLIP 的 “best on four of five columns” 是明确错误。**

你写：

> “FDIL is best on four of the five reported columns”

然后自己列出来：

* Macro：FDIL 赢
* Micro：FDIL **输**
* Samples：FDIL 赢
* Avg：FDIL 赢
* Hard：HVU-CLIP **没有报告**

所以可比较的其实只有 4 项，FDIL 是 **3/4 胜**，而不是 4/5。

建议直接改：

> “FDIL outperforms HVU-CLIP on three of the four commonly reported metrics (Macro, Samples, and Avg. F1), while trailing on Micro F1; HVU-CLIP does not report Hard F1.”

这个小错误很值得现在修，因为 reviewer 一眼就能算出来。

---

6.

Table 12：你说 flattening/removing gate “degrades consistently”，但 uniform gate 的 Hard 是 **+1.13**，所以不能说所有指标都 consistently degrade。 改成：

> “degrades Avg. F1 and most metrics”

即可。

---


还有一些纯语言小问题，比如：

> “we report...” 出现在句号后仍然小写；
> “we propose the FDIL” 应该是 “we propose FDIL”；
> “intention perception ,” 多了空格。

这些不致命，但 Reviewer 4 已经明确要求 proofreading，最好别再留下这种东西。

---

### 哪些部分已经做得很好

统计验证这次基本可以放心。五 seeds、95% t-interval、paired two-sided t-test、Holm correction，而且把唯一一个不显著的 Samples F1 vs UTD-only (p_H=0.0755) 也明确报告了，这正面满足了 Reviewer 2 和 Reviewer 6 的要求。

不过 response 中的措辞最好从 **“against every compared method”** 改成 **“against every matched-protocol comparator”**。因为 CoT4Intent、HVU-CLIP、IntentMLM 这些 published-only rows 并没有五 seed paired test，你真正检验的是 matched reproductions。

recent baselines 这块也够了：HVU-CLIP、CoT4Intent 加进主比较，又加了一组近期 zero-shot MLLM；而且你没有为了显得 FDIL 强而隐藏 GPT-5.6 在 mAP/Hard 上更高的事实，这反而是加分项。

efficiency、failure visualization 和 deployment 也都回答到了；35 页 manuscript 我看排版上没有明显的溢出、重叠、图表裁切问题。Response 里把 supplementary statistics 放进去，在正文达到 page limit 的情况下也算合理。

### 我对当前版本的最终判断

**如果不考虑上面几个问题，这次 revision 已经比上一版强很多，二审的主要批评基本都有实质性新实验回应，而不是只靠措辞。**

但我会把提交前优先级排成：

**第一优先：修理论逻辑 → 第二优先：处理 larger-scale 这个没有真正满足的问题 → 第三优先：补清 LDL adaptation protocol → 然后修 Abstract/GPT-5.6、4/5、Table 4、HLEG venue 等明确错误。**

其中**理论问题我认为必须改**。它不是“reviewer 可能不喜欢这种解释”，而是当前“(I(Y;X|Z)) 被 representation 丢弃”与“SLR-C 只读取同一个 (Z)”之间确实存在形式上的矛盾。好在这个问题**不需要重新做实验，改理论 decomposition 就能修正，而且改完反而会比现在更严谨**。