# v3 Design: Confusion-Aware Conditional Multi-Agent Ambiguity Resolution

## 0. Goal

This document is the implementation-facing design spec for the next experiment line.

The core goal is **not** to replace the current strong system (`baseline + scenario SLR-C`), but to add a **targeted ambiguity-resolution module** that:

1. only activates on hard / ambiguous cases,
2. operates inside a confusion neighborhood instead of full label space,
3. uses specialist evidence agents with real routing / interaction,
4. remains lightweight enough to justify itself experimentally.

This document is written so that an engineer / coding agent can directly use it to implement and run experiments.

---

## 1. Problem Setup

Current findings:

- The strong frozen CLIP baseline already provides strong visual representation.
- `scenario SLR-C` is the main source of performance gain.
- Remaining errors are mostly **candidate-local ambiguities** inside top-k predictions.
- These errors are concentrated on semantically similar intent pairs / groups.

Therefore, v3 should be framed as:

> A confusion-aware, conditionally triggered, multi-agent disambiguation module built on top of `SLR-C`, designed specifically for hard ambiguous candidate neighborhoods.

---

## 2. Base System (must stay unchanged first)

v3 must **reuse the current best base system** before introducing any new module.

### 2.1 Visual baseline

- frozen CLIP `ViT-L/14`
- feature = last-layer `CLS + mean(patch)`
- MLP classifier
- output baseline logits:

$$
z^{base}(x) \in \mathbb{R}^{C}
$$

### 2.2 SLR

Use `scenario` text prior to do local reranking only inside top-k:

$$
\tilde z_c =
\begin{cases}
z_c^{base} + \alpha \cdot \tilde s_c^{scenario}, & c \in \mathcal{T}_k(x) \\
z_c^{base}, & c \notin \mathcal{T}_k(x)
\end{cases}
$$

Current default:

- `top-k = 10`
- `alpha = 0.3`
- fusion = `add_norm`

### 2.3 Class-wise calibration

Keep the same class-wise threshold search on validation set:

$$
\hat y_c^{slr-c} = \mathbb{I}(\sigma(\tilde z_c) > t_c)
$$

### 2.4 Role of the base system

In v3, `SLR-C` is the **Proposal Agent**. Its job is to:

- generate high-quality top-k proposals,
- provide confidence / uncertainty signals,
- decide whether a second-stage ambiguity resolver is needed.

Important: do **not** modify base training or SLR-C itself in the first v3 stage. First prove the value of the new disambiguation framework.

---

## 3. v3 Method Design

## 3.1 Overall pipeline

```text
Image
  -> Proposal Agent (baseline + scenario SLR-C)
  -> top-k candidates + ambiguity signals
  -> Ambiguity Router Agent
       -> easy sample: output SLR-C directly
       -> ambiguous sample: dispatch specialist evidence agents
  -> Specialist Evidence Agents (object / scene / style / activity)
  -> pairwise comparative evidence over confusion neighborhood
  -> Resolver Agent
  -> local residual rerank on selected candidates
  -> class-wise thresholds
  -> final multi-label prediction
```

---

## 3.2 Agent 1: Proposal Agent

Proposal Agent = current `SLR-C`.

It must output:

1. reranked logits `\tilde z`
2. top-k candidate set `\mathcal{T}_k(x)`
3. ambiguity-related metadata, including at least:
   - top-1 / top-2 margin
   - top-k score distribution / entropy
   - candidate text similarity
   - whether the top candidates belong to known confusion pairs / groups

This agent is responsible for **proposal + escalation signals**, not final hard-case resolution.

---

## 3.3 Agent 2: Ambiguity Router Agent

This is the key structural addition in v3.

The router must make two decisions:

1. whether to trigger second-stage ambiguity resolution,
2. which candidate pairs / groups and which specialist agents should be activated.

### 3.3.1 Router inputs

Minimum router inputs:

- top-k candidates from `SLR-C`
- top-1/top-2 margin
- score entropy or similar uncertainty measure
- candidate semantic similarity
- confusion statistics computed from validation / training errors

### 3.3.2 Router outputs

The router should output:

- `trigger = {0,1}`
- selected confusion neighborhood / candidate pairs
- selected specialist agent subset

### 3.3.3 Initial implementation rule

Start with a rule-based router before trying any learned router.

Recommended MVP rules:

- trigger if top-1 / top-2 margin < `tau_margin`
- trigger if top-k contains a known confusion pair / confusion group
- optionally also trigger if candidate text similarity is above `tau_sim`
- if no trigger condition is met, directly use `SLR-C` output

### 3.3.4 Why this matters

This is what turns the method from a full heavy verifier into a **conditional targeted framework**.

---

## 3.4 Agent 3: Specialist Evidence Agents

Keep the current evidence spaces, but reinterpret them as role-specialized agents:

1. **Object Agent**
2. **Scene Agent**
3. **Style Agent**
4. **Activity Agent**

These agents should not output full-label global scores. Instead, each agent should output **comparative evidence** for selected candidate pairs.

### 3.4.1 Input

For each specialist agent:

- image `x`
- selected candidate pair or confusion group from the router
- corresponding relation profiles

### 3.4.2 Output

For a selected pair `(c_i, c_j)`, agent `a` outputs:

$$
\Delta V_a(c_i, c_j, x)
$$

interpreted as how much this agent supports `c_i` over `c_j`.

### 3.4.3 Evidence computation

Keep the current evidence extraction principle:

1. compute image activation in the specialist evidence bank,
2. use learned relation profiles,
3. compute comparative support only on activated sparse evidence.

Basic form:

$$
\Delta V_a(c_i,c_j,x)=\sum_{z \in A_a(x)} w_z(x) [R_a(c_i,z)-R_a(c_j,z)]
$$

Where:

- `A_a(x)` = activated sparse evidence set for agent `a`
- `R_a(c,z)` = relation score between class `c` and evidence `z`

### 3.4.4 Relation choice

Start from the strongest existing relation:

- default: `hard_negative_diff`

Secondary branch:

- `support_contradiction`

Do **not** reopen a broad relation-family search at the start.

---

## 3.5 Required interaction mechanism

To avoid “fake agentization”, v3 must include actual interaction logic.

At minimum, implement these three forms of interaction:

### 3.5.1 Router-to-agent dispatch

The router should choose whether all specialists are called or only a subset.

MVP comparison should include:

- full specialist call
- selective specialist dispatch

If possible, route by ambiguity type, e.g.:

- action-heavy confusion -> prioritize Activity Agent
- scene-heavy confusion -> prioritize Scene Agent
- aesthetic / design confusion -> prioritize Style Agent

### 3.5.2 Candidate-pair-specific activation

Specialist agents should work only on router-selected pairs / groups, not uniformly on all top-k candidates.

### 3.5.3 Resolver-mediated aggregation

Final evidence should not be produced by simple naive concatenation or unconditional score summation.

---

## 3.6 Agent 4: Resolver Agent

The Resolver Agent aggregates specialist outputs and converts them into local candidate reranking signals.

### 3.6.1 Inputs

- reranked logits `\tilde z`
- selected candidate pairs / neighborhoods
- outputs from selected specialist agents

### 3.6.2 Pairwise aggregation

For pair `(c_i, c_j)`:

$$
S(c_i,c_j,x)=\sum_{a \in \mathcal{A}(x)} \lambda_a(x,c_i,c_j) \cdot \Delta V_a(c_i,c_j,x)
$$

Where:

- `\mathcal{A}(x)` = selected specialist set
- `\lambda_a` = resolver weight for specialist `a`

MVP can start with fixed or rule-based `\lambda_a`.

### 3.6.3 Candidate-level comparative score

Aggregate pairwise outputs inside the selected neighborhood:

$$
V^{res}(c_i,x)=\frac{1}{|\mathcal{N}(c_i)|} \sum_{c_j \in \mathcal{N}(c_i)} S(c_i,c_j,x)
$$

### 3.6.4 Final update

Only update candidates inside the selected ambiguity neighborhood:

$$
z_c^{final}=\tilde z_c + \beta \cdot g(x) \cdot V^{res}(c,x)
$$

Keep an uncertainty-aware gate `g(x)`, but in v3 it should be interpreted as part of the **router + resolver control mechanism**, not a standalone scaling trick.

---

## 3.7 Key differences from v2

Relative to v2, v3 changes the method structure in four important ways:

1. **conditional triggering instead of full verifier invocation**
2. **specialist evidence agents instead of one monolithic verifier view**
3. **router + dispatch + resolver interaction instead of only margin gating**
4. **confusion-neighborhood comparison instead of broad top-k pairwise processing**

---

## 3.8 Recommended MVP

The first implementation should be intentionally conservative.

### Step A. Keep the Proposal Agent unchanged
Reuse current `scenario SLR-C` exactly.

### Step B. Implement a rule-based router
Use margin + confusion-pair hit as the first routing rule.

### Step C. Reuse current evidence extraction backend
Do not redesign evidence extraction yet. First change the **usage pattern**.

### Step D. Implement a simple resolver
Start with fixed or rule-based specialist weights.

### Step E. Restrict re-ranking to selected confusion neighborhoods
Do not apply the resolver to all top-k candidates by default.

This MVP is enough to test whether the multi-agent formulation has real value.

---

## 4. Required Experiments

The experiments should be run in phases. Do not jump to learned router / learned resolver before proving the basic framework works.

### 4.1 Phase 1: prove the framework is meaningful

#### EXP-1 Main comparison
Compare:

1. baseline
2. `scenario SLR-C`
3. v2 best method
4. v3 MVP (rule-based router + all specialists + resolver)
5. v3 routed (rule-based router + selective specialists + resolver)

Report:

- Macro F1
- Micro F1
- Samples F1
- mAP
- Hard

Primary target:

- improve `Hard`
- preferably improve or maintain `Macro`
- avoid significant degradation elsewhere

#### EXP-2 Triggering strategy
Compare:

1. always trigger second stage
2. margin-only trigger
3. confusion-only trigger
4. margin + confusion trigger
5. margin + confusion + semantic-similarity trigger

Also report:

- trigger rate
- average number of specialists called
- additional runtime cost

#### EXP-3 Interaction ablation
Compare:

1. no routing, all specialists, direct sum
2. routing only, all specialists
3. routing + selective specialist dispatch
4. routing + selective dispatch + resolver

Purpose:

- verify that the gain comes from interaction design rather than just adding components

#### EXP-4 Hard-subset evaluation
Construct and evaluate on:

- low-margin subset
- top confusion-pair subset
- semantic-neighbor subset

Additional metrics:

- pairwise ranking accuracy
- top-2 disambiguation accuracy
- Hard

This experiment is essential because v3 is explicitly designed for hard ambiguity cases.

---

### 4.2 Phase 2: verify specialist / resolver design

#### EXP-5 Specialist decomposition
Compare:

1. unified verifier/evidence scorer
2. all four specialist agents
3. routed specialist subset

#### EXP-6 Resolver ablation
Compare:

1. average fusion
2. fixed weighted sum
3. pairwise resolver aggregation
4. pair/group-aware resolver

#### EXP-7 Router ablation
Compare routing signals:

- margin only
- confusion only
- semantic similarity only
- margin + confusion
- all signals

#### EXP-8 Relation ablation
Compare:

- `hard_negative_diff`
- `support_contradiction`
- confusion-neighborhood version of relation

---

### 4.3 Phase 3: strengthen paper readiness

#### EXP-9 Efficiency analysis
Report:

- average specialists called per sample
- trigger rate
- extra latency / runtime cost vs v2

#### EXP-10 Qualitative case studies
At least 4–6 examples showing:

- SLR-C candidates
- router trigger decision
- which specialists were called
- specialist comparative evidence
- resolver final decision

Priority cases:

- `Happy` vs `Playful`
- `EnjoyLife` vs `Happy`
- `FineDesign` vs `FineDesign-Art`

#### EXP-11 Stability / robustness
Evaluate:

- multiple random seeds
- routing thresholds
- different confusion-neighborhood sizes
- different numbers of confusion pairs

---

## 5. Implementation Priorities

Codex / implementation should follow this order:

1. **Do not touch base `SLR-C` first.**
2. Add utilities to extract ambiguity metadata from proposal outputs.
3. Implement a rule-based router.
4. Refactor evidence computation into specialist-agent style interfaces.
5. Implement pairwise neighborhood-level resolver.
6. Add experiment configs for v3 MVP.
7. Run Phase 1 experiments first.
8. Only if Phase 1 is promising, continue to Phase 2 and Phase 3.

---

## 6. Deliverables expected from the first implementation round

The first coding / experiment round should produce:

1. a runnable v3 MVP implementation,
2. config(s) for router trigger variants,
3. config(s) for all-specialist vs routed-specialist comparisons,
4. scripts or analysis outputs for hard-subset evaluation,
5. a concise experiment note summarizing whether v3 is promising enough to continue.

---

## 7. One-sentence summary

v3 should be implemented as:

> a confusion-aware, conditionally triggered, multi-agent disambiguation framework built on top of `SLR-C`, where a router selects hard ambiguity cases, specialist evidence agents provide pairwise comparative support, and a resolver performs local candidate adjudication inside confusion neighborhoods.
