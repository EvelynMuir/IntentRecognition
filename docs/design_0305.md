# Intent Concept Reasoning Network (ICRN)

## 1. Overview

ICRN predicts image intents through an intermediate **shared concept layer**, instead of directly mapping image features to intent labels.

Core assumption: intents are compositions of reusable visual-semantic concepts (e.g., social interaction, emotional cues, contextual situations).

Pipeline:

```text
Image
  -> CLIP visual encoder (frozen)
  -> visual feature v
  -> concept grounding
  -> concept activations c
  -> concept graph reasoning
  -> refined concepts c'
  -> intent composition
  -> intent logits z
```

## 2. Notation

- `D`: visual/text embedding dimension.
- `M`: number of concepts (typically `80-120`).
- `K`: number of intents (here `K = 28`).
- `v in R^D`: image embedding.
- `E_c in R^(M x D)`: concept embedding matrix.
- `c in R^M`: concept activation vector.
- `A in R^(M x M)`: concept graph adjacency/similarity matrix.
- `W in R^(M x K)`: concept-to-intent composition matrix.
- `z in R^K`: intent logits.

## 3. Step-by-Step Design

### Step 1. Visual Feature Extraction

Use a frozen CLIP ViT-L/14 visual encoder.

For each image:

```text
v = CLIP_visual(image)
```

Recommended feature construction:

```text
v = concat(CLS_token, mean(patch_tokens))
v = normalize(v)
```

复用cls_mean_patch

### Step 2. Concept Embedding

Maintain a shared concept pool:

```text
concept_list = [c1, c2, ..., cM]
```

文件在`/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json`

Concept examples:

- hugging
- group gathering
- smiling
- conversation
- collaborative work
- helping another person

Generate concept embeddings via frozen CLIP text encoder:

```text
E_c = CLIP_text(concept_list)
E_c = normalize(E_c)
```

### Step 3. Concept Grounding

Ground concepts by similarity between image and concept embeddings:

```text
c = sigmoid(E_c * v)
```

- `E_c`: `(M x D)`
- `v`: `(D)`
- `c`: `(M)`

Example interpretation:

- `c_smiling = 0.82`
- `c_hugging = 0.76`
- `c_group_gathering = 0.65`

### Step 4. Concept Graph Construction

Concepts are correlated. Build a concept similarity graph from concept embeddings:

```text
A_raw = E_c * E_c^T
A = row_softmax(A_raw)
```

`A` encodes semantic proximity among concepts.

Example relations:

- hugging <-> physical_contact
- smiling <-> joy
- conversation <-> social_interaction

### Step 5. Concept Graph Reasoning

Refine concept activations through graph propagation:

```text
c' = A * c
```

Optional nonlinearity:

```text
c' = ReLU(A * c)
```

This allows related concepts to reinforce each other.

### Step 6. Intent Composition

Model each intent as a weighted combination of concepts:

```text
z = c' * W
y_hat = sigmoid(z)
```

- `c'`: `(M)`
- `W`: `(M x K)`
- `z`: `(K)`

## 4. LLM Intent Prior (Optional but Recommended)

Use an LLM offline to estimate intent-to-concept relevance and form a prior matrix:

- `S in R^(K x M)`

Example:

```text
comforting:
  hugging: 0.9
  emotional_support: 0.8
  physical_contact: 0.7

celebrating:
  smiling: 0.9
  group_gathering: 0.85
  cheering: 0.8
```

文件在`/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json`

Initialize the composition matrix with the prior:

```text
W = S^T
```

Optional prior regularization:

```text
L_prior = ||W - S^T||^2
```

## 5. Training Objective

Base classification loss（复用Optimized Asymmetric Loss）:

```text
L_cls = ASL(z, y)
```

With prior regularization:

```text
L = L_cls + lambda_1 * L_prior
```

Optional concept sparsity (images usually activate only a subset of concepts):

```text
L_sparse = ||c||_1
L = L_cls + lambda_1 * L_prior + lambda_2 * L_sparse
```

## 6. Inference

```text
v = CLIP_visual(image)
c = sigmoid(E_c * v)
c' = A * c
z = c' * W
y_hat = sigmoid(z)
```

Obtain predicted intents by thresholding `y_hat`.

## 7. Sanity Checks

- Dimension consistency is valid across all stages.
- `W` is a full composition matrix (`M x K`), not a single intent vector.
- Use `row_softmax` for `A` to avoid ambiguity in graph normalization.
- Keep naming consistent (`CLIP_visual` during both training/inference).
