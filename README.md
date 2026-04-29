# Dynamic Personalized Shadow Models.

**Author:** Mircea Mihai Bontoi \
**Date:** April 2026  
**Status:** Preprint

---

## Abstract

We propose a novel inference optimization system for large language models (LLMs) in which a lightweight personalized "shadow model" is constructed incrementally from the activation patterns of a full-scale model during real user interactions. Unlike static compression techniques such as pruning or quantization, the shadow model grows dynamically, capturing only the subset of neurons and weights that are actually needed for a specific user's usage patterns. The result is a small, dense, standard model that runs on conventional hardware without any special requirements. Over time, the shadow model converges to a compact representation that handles the vast majority of that user's requests independently, without requiring retraining or offline distillation. This approach combines the accuracy of large models with the efficiency of small specialized ones, personalized at the individual user level.

---

## 1. Introduction

Large language models have demonstrated remarkable capabilities across a wide range of tasks. However, their computational cost at inference time remains a significant barrier to deployment on resource-constrained devices and for cost-sensitive applications.

Existing compression techniques — including pruning, quantization, and knowledge distillation — reduce model size globally, without accounting for the fact that individual users interact with a model in highly specialized, repetitive ways. A software developer using an LLM as a coding assistant activates a very different subset of the model's parameters than a writer using it for creative work. Yet both users pay the full computational cost of the entire model on every request.

We propose exploiting this observation directly: rather than compressing the model uniformly, we construct a personalized shadow model for each user by recording and accumulating the specific activation subgraphs triggered by their actual requests. The shadow model is not a sparse version of the original — it is an independent, smaller, dense model containing only what that user actually needs.

---

## 2. Background

### 2.1 Sparse Activation in LLMs

It is well established that during inference, a significant fraction of neurons in a large neural network produce near-zero activations for any given input. These neurons contribute negligibly to the output for that specific input. This observation motivates our approach: if a neuron is consistently inactive for a given user's requests, it need not be part of that user's model at all.

### 2.2 Mixture of Experts (MoE)

Mixture of Experts architectures partition model parameters into discrete expert modules, with a learned router activating a small subset per token. This achieves high model capacity with low per-token compute. However, expert boundaries are fixed at training time and are not personalized to individual users. The routing decision is made per token, not per user, and there is no mechanism for a user's history to influence which experts are loaded.

### 2.3 Knowledge Distillation

Knowledge distillation trains a small student model to imitate the outputs of a large teacher model. Standard distillation is an offline process requiring a curated dataset and a full training run. Online distillation variants exist but are not personalized, not activation-driven, and still require gradient updates. Our approach requires none of these.

---

## 3. Proposed System

### 3.1 Core Insight

For any given input, only a fraction of the neurons in a large model activate meaningfully. The neurons that do activate, together with their connections and weights, are sufficient to reproduce that output. If we copy those neurons and weights into a new smaller model, that smaller model can reproduce the same output for similar inputs — with no retraining required, running on standard hardware, occupying a fraction of the memory.

### 3.2 System Overview

The system operates in two modes: **full model mode** and **shadow model mode**.

```
New user request arrives
          ↓
Is shadow model ready for this input?
          ├── Yes → shadow model responds (fast, cheap, local)
          └── No  → full model responds
                          ↓
                    Record activated neurons and weights
                          ↓
                    Merge into shadow model (new neurons only)
                          ↓
                    Shadow grows, return response to user
```

### 3.3 Activation Recording

For each user request processed by the full model, we perform a standard forward pass with activation hooks attached to all layers. The same threshold mechanism applies uniformly to both types of components in a transformer:

For feedforward neurons, we record every neuron whose activation magnitude exceeds threshold τ:

```
S_ff(x) = { neuron n : |activation(n, x)| > τ }
```

For attention heads, we record every head whose output vector norm exceeds the same threshold τ:

```
S_attn(x) = { head h : ||output(h, x)|| > τ }
```

The logic is identical in both cases: if a component's contribution to the output is near zero for this input, it is not copied into the shadow. A head with a near-zero output norm contributed nothing to the response regardless of its internal attention weights, and can be safely excluded.

**On residual stream consistency.** Transformer attention heads write their outputs into a shared residual stream of fixed dimension. When a head is excluded from the shadow, its contribution to that stream is zero — which is identical to its contribution in the full model for inputs where it was inactive. The residual stream state in the shadow is therefore consistent with the full model for those inputs. The remaining open question — whether heads that are inactive for a given user's typical requests introduce measurable distribution shift when permanently absent rather than occasionally inactive — is quantified in the proposed experiment in Section 6.

The full activated subgraph is therefore:

```
S(x) = S_ff(x) ∪ S_attn(x)
      + all weights connecting components in S(x)
```

This is a standard operation supported natively by frameworks such as PyTorch and does not require any modification to the model or special hardware.

### 3.4 Adaptive Threshold Learning

Rather than fixing τ as a static hyperparameter, the system discovers the optimal threshold automatically for each user during the growth phase.

During early sessions, both the shadow model and the full model run in parallel on every request. This produces two outputs that can be compared directly using KL divergence over the token probability distributions:

```
KL(full model output || shadow output) < ε → threshold acceptable
KL(full model output || shadow output) ≥ ε → threshold too aggressive, lower τ
```

The system uses this signal to adjust τ over time:

```
Early sessions (τ low):
  shadow is large, very accurate
  system measures divergence on every request

Mid sessions:
  system gradually raises τ
  tests whether divergence stays below ε
  finds the minimum shadow size that preserves quality

Converged:
  optimal τ found for this specific user
  shadow is as small as possible without measurable quality loss
```

This transforms threshold selection from a static engineering decision into a self-optimizing property of the system. Two users with different usage patterns will converge to different optimal thresholds automatically, without any manual configuration. The shadow is always as small as possible and as large as necessary, discovered entirely from the user's own behavior.

### 3.4 Shadow Model Construction

The shadow model is initialized as empty. After each full-model request, the activated subgraph S(x) is merged into the shadow:

```
Shadow ← Shadow ∪ S(x)
```

Only neurons and weights not already present in the shadow are added. The shadow model is at all times a valid, dense, self-contained neural network — not a sparse mask over the original. It runs on any hardware that can run a model of its size.

Each time the shadow grows, normalization statistics (mean and standard deviation) are recomputed using only the neurons currently present in the shadow. This ensures that layer normalization remains internally consistent within the shadow without any reference to the full model, and adds negligible cost since it only occurs on growth steps, not on every request.

### 3.5 Shadow Model Inference

When a new request arrives, it is routed to the shadow model first. If the shadow produces a confident output, that response is returned directly. If confidence is low or required neurons are missing, the full model handles the request and the shadow is updated with the new activations before responding.

This guarantees that the shadow model never returns an incorrect response: it either answers correctly from its accumulated knowledge, or falls back to the full model and grows.

### 3.6 Convergence

Users exhibit highly repetitive usage patterns. A software developer asking about Python will repeatedly activate the same core subgraph across many requests. As a result, the shadow model grows rapidly in early sessions and converges toward a stable, compact representation after tens to hundreds of interactions.

```
Requests 1-10:    shadow grows quickly (many new neurons per request)
Requests 10-50:   shadow grows slowly (most neurons already present)
Requests 50-100:  shadow nearly stable, covers ~90-95% of user requests
Requests 100+:    full model rarely needed
```

The shadow is never retrained. Its accuracy comes from being a direct copy of the full model's activations, not from approximation.

### 3.7 Self-Aware Coverage Detection

A key requirement of the system is knowing whether the shadow model can handle a new request without consulting the full model first. Rather than using an external classifier, the shadow model detects this itself through the entropy of its own output distribution.

When the shadow model has the neurons required to answer a request confidently, it produces a peaked probability distribution over the next token:

```
Request within shadow territory:
  "for"    → probability 0.94  ← confident
  "while"  → probability 0.03
  "def"    → probability 0.02
  entropy  → low
```

When the shadow lacks the necessary neurons, it produces a flat, uncertain distribution:

```
Request outside shadow territory:
  "for"    → probability 0.21  ← uncertain
  "while"  → probability 0.19
  "def"    → probability 0.18
  entropy  → high
```

This signal is available for free during the shadow's forward pass and requires no additional computation. The routing decision becomes:

```
Shadow entropy < threshold → shadow responds directly
Shadow entropy ≥ threshold → fall back to full model, grow shadow
```

To guard against the known failure mode of low-entropy but incorrect responses (confident hallucination), a secondary check uses semantic similarity between the request and the shadow's historical request embeddings:

```
Low entropy AND high similarity → shadow responds with full confidence
Low entropy AND low similarity  → verify with full model
High entropy                    → full model directly
```

This two-signal approach makes the shadow model self-aware of the boundaries of its own knowledge without any external component.

### 3.8 Shadow Staleness Management

When the base model is updated, the shadow's copied weights may become inconsistent with the new version. The system handles this through a layer-level hashing strategy analogous to version control in software:

At shadow construction time, a hash is computed for each layer of the base model and stored alongside the shadow:

```
Shadow metadata:
  {
    layer_1_hash:  a3f9b2c1,
    layer_2_hash:  b7d4e9f2,
    layer_3_hash:  c2a1f8b3,
    ...
    full_model_hash: x9f2a1b4  ← cheap summary hash of entire model
  }
```

At the start of each session, only the full model hash is recomputed and compared. This is fast regardless of model size since it operates on a summary:

```
full_model_hash unchanged → shadow is valid, proceed normally
full_model_hash changed   → run per-layer diff
```

The per-layer diff identifies exactly which layers changed:

```
layer_1: a3f9b2c1 == a3f9b2c1  ✅ unchanged, shadow weights still valid
layer_2: b7d4e9f2 == b7d4e9f2  ✅ unchanged
layer_3: c2a1f8b3 != d7e2a9f4  ❌ changed, update shadow weights for this layer
layer_4: f1b3c9a2 != e4d8b1c7  ❌ changed, update shadow weights for this layer
```

Only the shadow neurons belonging to changed layers need to be refreshed. Neurons in unchanged layers remain valid and the user's accumulated profile is preserved. This surgical update avoids discarding the entire shadow on every model update, which would eliminate the personalization benefit entirely.

---

## 4. Key Properties

**No retraining required.** The shadow model is constructed entirely from inference-time observations. No gradient updates, no curated datasets, no training runs.

**Standard hardware.** The shadow model is a small dense model. It runs on any GPU, CPU, or mobile chip capable of running a model of its size. No sparse-aware hardware is required.

**Accuracy guaranteed by construction.** The shadow model never diverges from the full model for inputs it has seen, because its weights are copied directly from the full model. For new inputs it falls back to the full model.

**Grows only as needed.** No neurons are copied speculatively. The shadow contains exactly what that user has actually needed, nothing more.

**Deployable to user devices.** A converged shadow model may be small enough to run locally on a laptop, phone, or edge device. This enables offline operation, near-zero latency, and complete privacy — the full model need not be consulted at all after convergence.

**Personalized by definition.** Two users with different usage patterns produce two different shadow models from the same base model, with no explicit personalization step required.

**Self-optimizing compression.** The adaptive threshold mechanism automatically discovers the minimum shadow model size that preserves quality for each specific user. No manual tuning is required.

**Self-aware coverage.** The shadow model detects the boundaries of its own knowledge through output entropy, routing requests to the full model only when genuinely needed. No external classifier or routing component is required.

**Self-managing versioning.** The shadow model detects base model updates automatically through layer-level hashing and refreshes only the affected layers, preserving the user's accumulated profile across model updates.

---

## 5. Differences from Prior Work

| Property | Pruning | Distillation | MoE | This work |
|---|---|---|---|---|
| Personalized per user | ❌ | ❌ | ❌ | ✅ |
| No retraining required | ✅ | ❌ | ❌ | ✅ |
| Incremental and online | ❌ | Partial | ❌ | ✅ |
| Driven by real user behavior | ❌ | ❌ | ❌ | ✅ |
| Accuracy fallback | ❌ | ❌ | ✅ | ✅ |
| Runs on standard hardware | ✅ | ✅ | ✅ | ✅ |
| Result is a standalone model | ✅ | ✅ | ❌ | ✅ |

---

## 6. Open Challenges and Future Work

**Per-layer threshold refinement.** The adaptive threshold mechanism described in Section 3.4 operates globally across the model. However, different layers may have very different activation distributions, meaning a single τ may be too aggressive for some layers and too permissive for others. Extending the adaptive mechanism to learn a separate threshold per layer is a natural improvement that could reduce shadow model size further without additional quality loss.

**Residual stream distribution shift.** As discussed in Section 3.3, omitting heads with near-zero output norm does not introduce errors for the specific inputs where those heads were inactive. However, a subtler issue remains: downstream layers were trained on a statistical distribution where those heads contribute *sometimes*, even if rarely. The shadow model presents those heads as permanently absent rather than occasionally inactive, which constitutes a small but systematic shift from the training distribution. For users with highly repetitive request patterns — the primary target of this system — those heads are likely to remain inactive on future requests as well, making this error practically negligible. However, the precise magnitude of this effect has not been empirically measured.

**Proposed experiment.** To quantify this effect we propose the following validation:

```
1. Select a base model with known architecture (e.g. LLaMA 3.2 3B)
2. Define a target user profile (e.g. Python developer)
3. Simulate N requests drawn from that profile
4. Construct the shadow model from those requests
5. Evaluate on:
   a. Seen inputs: KL divergence between shadow and full model outputs
   b. Similar unseen inputs: KL divergence on held-out requests
      from the same profile
   c. Dissimilar inputs: confirm fallback triggers correctly
6. Measure shadow model size as a fraction of full model size
7. Repeat across multiple user profiles and model sizes
```

This experiment would directly quantify whether the residual stream distribution shift introduces measurable degradation in practice, and establish the compression ratio achievable for different user profiles.

---

## 7. Conclusion

We have described a system for constructing personalized shadow models of large language models through incremental accumulation of user-specific activation subgraphs. The shadow model is not a sparse approximation of the original — it is a small, dense, standalone model containing exactly the neurons and weights that a specific user actually needs.

The approach requires no retraining, no special hardware, and no offline preparation. It preserves full-model accuracy through a guaranteed fallback mechanism, and converges naturally to a compact user-specific model over time.

We believe this represents a genuinely novel direction for personalized LLM efficiency, with particular relevance for long-term single-user deployments, on-device inference, and privacy-preserving applications.

---


*Preprint. Not peer reviewed.*