# Dynamic Personalized Shadow Models

**Subtitle:** Activation-Guided User-Specific Compression for LLM Inference  
**Author:** Mircea Mihai Bontoi  
**Date:** April 2026  
**Status:** Research draft / preprint, not peer reviewed

---

## Abstract

Large language models are typically deployed as general-purpose systems, even though individual users often interact with them through highly repetitive and specialized distributions of prompts. This draft investigates whether those user-specific interaction patterns can be exploited to build compact personalized inference models, referred to here as **shadow models**.

The proposed system profiles the activation patterns of a full-scale model during real user interactions and uses those observations to select a subset of structured components, such as feedforward channels and attention heads, that are consistently important for that user's workload. A candidate shadow model is then constructed from these components and validated against the full model using divergence-based quality checks. Requests that fall outside the shadow model's reliable coverage are routed back to the full model, allowing the system to grow or recalibrate over time.

The central hypothesis is that many long-term users induce stable activation profiles that can support meaningful personalized compression without requiring a full offline distillation run. Unlike static pruning, the proposed approach is user-specific and dynamic. Unlike standard distillation, it is driven by observed internal activations rather than by training a separate student model from scratch. The goal is not to claim exact equivalence to the full model, but to explore whether activation-guided personalization can reduce inference cost while preserving quality under explicit empirical validation and fallback.

---


## 1. Introduction

Large language models have demonstrated strong general capabilities across programming, writing, reasoning, information retrieval, and interactive assistance. However, their inference cost remains a major obstacle for low-latency applications, privacy-preserving local deployment, and cost-sensitive services.

Most compression techniques reduce the model globally. Quantization, pruning, and distillation usually produce a smaller model intended to serve all users. This ignores an important practical observation: individual users often occupy narrow and recurring regions of the model's behavioral space. A developer who repeatedly asks Python and systems-programming questions, a novelist who mostly requests prose editing, and a student who primarily asks about mathematics may rely on different internal computation patterns.

This draft explores the possibility of using those recurring patterns as a compression signal. Instead of asking which parts of the full model are globally important, the proposed system asks a narrower question:

**Which components of the model are repeatedly important for this specific user's workload?**

The proposed answer is a personalized shadow model: a compact inference model or structured subnetwork derived from the full model's user-specific activation history. The shadow model is not assumed to be correct by construction. It is treated as a candidate approximation whose behavior must be validated against the full model and whose coverage must be monitored during use.

---


## 2. Background and Related Work


### 2.1 Sparse and Structured Activation in LLMs

During inference, many internal components of a neural network contribute weakly to a given input. In transformer models, this can appear as low-magnitude feedforward activations, low-norm attention-head outputs, or low contribution from specific channels across layers. This motivates activation-aware compression: components that are consistently unimportant for a workload may be candidates for removal or deactivation.

However, transformers are not simple collections of independent neurons. Their components interact through residual streams, normalization layers, attention mechanisms, and shared projections. For this reason, the proposal focuses on **structured components** such as attention heads and MLP channels rather than arbitrary individual weights. Structured selection is more likely to preserve a valid architecture and to produce practical latency or memory benefits.


### 2.2 Pruning and Activation-Aware Compression

Prior work on pruning and compression has shown that large models often contain removable redundancy. Magnitude pruning, structured pruning, SparseGPT-style methods, Wanda-style activation-aware pruning, and LLM-specific pruning approaches all attempt to reduce model size or inference cost while preserving behavior.

The distinction explored here is personalization. Most pruning methods optimize for a general evaluation distribution. The shadow-model hypothesis instead optimizes for a specific user's observed workload, and it updates the selected components as the user's interaction pattern evolves.


### 2.3 Mixture of Experts and Conditional Computation

Mixture of Experts architectures reduce per-token computation by routing each token through a subset of expert modules. They demonstrate that large-capacity models do not necessarily need to use all parameters for every token.

The proposed system is related in spirit but differs in mechanism. MoE expert boundaries are defined during training, and routing is part of the model architecture. A personalized shadow model would instead be derived after deployment by observing which parts of an existing model are repeatedly useful for a particular user.


### 2.4 Knowledge Distillation and Personalized Adaptation

Knowledge distillation trains a smaller student model to imitate a larger teacher model. This can produce efficient models, but it usually requires a training dataset, optimization, and offline evaluation. Personalized adapters and fine-tuning methods can adapt models to specific users or domains, but they still rely on gradient-based updates.

The proposed approach attempts to minimize or avoid gradient updates during the initial construction phase. It uses activation profiling and component selection first, while leaving optional calibration or distillation as future enhancements if pure activation-guided construction proves insufficient.

---


## 3. Proposed System

### 3.1 Research Hypothesis

The core hypothesis is:

**For many long-term users, the full model's activations converge toward a stable user-specific profile. This profile can be used to construct a smaller personalized inference model that handles a large fraction of that user's future requests with acceptable divergence from the full model.**

This hypothesis has three parts that must be tested empirically:

1. User-specific activation profiles must be stable enough to measure.
2. The selected components must be sufficient to approximate the full model on similar future inputs.
3. The system must reliably detect when a request falls outside the shadow model's coverage.

### 3.2 System Overview

The system operates in two main phases: a **growth and validation phase**, followed by a **shadow-assisted inference phase**.

```
New user request arrives
          ↓
Run routing check
          ↓
Can a validated shadow handle this request reliably?
          ├── Yes → shadow model responds
          └── No  → full model responds
                          ↓
                    Record activations
                          ↓
                    Update user activation profile
                          ↓
                    Rebuild or refresh candidate shadow model
                          ↓
                    Validate shadow against full model
```

During early interactions, the full model remains the source of truth. The shadow model is evaluated in parallel and is allowed to answer directly only after it satisfies predefined quality and coverage criteria.

 
### 3.3 Single-Shadow and Multi-Shadow Personalization

A user does not need to be represented by a single monolithic shadow model. If the user's workload contains distinct recurring activities, such as programming, writing, research, or administrative tasks, the system can maintain separate activity-specific shadows.

In the single-shadow case, the system learns one compact profile for the user's dominant distribution. In the multi-shadow case, the system decomposes behavior into several recurring activity profiles and maintains a separate candidate shadow for each one. A broad user may therefore still benefit from personalization if their behavior can be partitioned into stable sub-distributions.

The routing problem then has two levels:

1. determine which activity-specific shadow, if any, is relevant to the request;
2. determine whether that shadow has been validated strongly enough to answer directly.


### 3.4 Shadow-Assisted Inference

At inference time, the system decides whether a shadow model is likely to be reliable for the current request. This routing decision can combine several signals:

1. **Output entropy** from the shadow model.
2. **Similarity** between the new request and the user's historical request embeddings.
3. **Calibration history** for similar prompts.
4. **Recent fallback rate** and observed divergence.

A simple routing rule could be:

```
if entropy < H_max and similarity > S_min and calibration_status == valid:
    use shadow model
else:
    use full model and update profile
```

Entropy alone is not sufficient because models can be confidently wrong. Similarity and calibration history reduce this risk, but they do not provide a formal correctness guarantee. The intended guarantee is operational rather than absolute: uncertain or out-of-profile requests are routed to the full model, while in-profile requests are answered by a shadow model that has been empirically validated on similar inputs.

---



## 4. Shadow Construction and Calibration

### 4.1 User-Specific Activation Profiling

For each request processed by the full model, activation hooks record structured component statistics across layers. The relevant signal is not the raw internal activation of a component, but the contribution that component writes into the residual stream after its output projection.

For MLP blocks, the system records channel-level contribution after the channel has passed through the MLP output projection:

```
I_mlp(l, c) = E_x [ ||residual_contribution_mlp(l, c, x)||_2 ]
```

where `l` is the layer index, `c` is the MLP channel index, and the expectation is estimated over the user's observed requests.

For attention blocks, the system records each attention head's post-output-projection contribution to the residual stream:

```
I_attn(l, h) = E_x [ ||residual_contribution_attn(l, h, x)||_2 ]
```

where `h` is the attention-head index.

The user profile is therefore represented as a set of running post-output importance estimates:

```
P_user = {
  I_mlp(l, c),
  I_attn(l, h),
  frequency(l, c),
  frequency(l, h),
  recent_usage(l, c),
  recent_usage(l, h)
}
```

A component is considered a candidate for inclusion when its estimated post-output importance exceeds a learned threshold:

```
S_user = { component k : I_user(k) >= τ_layer(k) }
```

Layer-specific thresholds are preferable to a single global threshold because contribution distributions can vary substantially across layers.


### 4.2 Post-Output Component Attribution

The post-output contribution is an attribution signal, not a cached activation. The system does not copy the activation vectors observed for specific prompts. Instead, those vectors are used to decide which parameterized components should be retained in the shadow model.

For attention heads, attribution is measured after the head's output has been mapped through its corresponding output-projection slice and written into the residual stream. If a head is retained, the shadow retains the parameter slices needed to compute that head on future inputs, including the relevant query, key, value, and output-projection parameters.

For MLP channels, attribution is measured after the channel contributes through the output or down projection back into the residual stream. If a channel is retained, the shadow retains the associated input, gate, and output projection slices needed to compute that channel on future inputs.

Under this definition, component removal means forcing that component's residual-stream write to zero in the candidate shadow. This makes attention-head removal, MLP-channel removal, and residual-coordinate compaction part of the same principle: select components according to their actual post-output effect on the residual stream, then validate the resulting candidate shadow against the full model.


### 4.3 Activation Recording

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



### 4.4 Zero-Imputed Normalization for Residual Compaction

When residual coordinates are removed, the shadow model should not necessarily normalize over the reduced width as if it were a newly trained smaller transformer. Instead, the proposed compact shadow treats omitted residual coordinates as **implicit zeros**. The model only materializes the retained coordinates, but normalization is computed with respect to the original hidden width.

For RMSNorm-style architectures, this is direct. If the original hidden width is `d` and the retained coordinate set is `S`, the shadow computes the RMS using the retained coordinates while keeping `d` as the normalization reference:

```
r_shadow = sqrt((Σ_{i ∈ S} x_i^2) / d + ε)
y_i      = γ_i x_i / r_shadow, for i ∈ S
```

The omitted coordinates are not stored or computed; they are equivalent to zero-valued coordinates in the original full-width residual stream. This is different from standard reduced-width normalization, which would divide by `|S|` and would therefore define a different scale.

For LayerNorm-style architectures, the same zero-imputation principle can be applied by including omitted coordinates as zeros when computing the mean and variance:

```
μ_shadow  = (Σ_{i ∈ S} x_i) / d
σ_shadow² = (Σ_{i ∈ S} (x_i - μ_shadow)^2 + (d - |S|) μ_shadow^2) / d
```

LayerNorm with bias terms may require additional care, because omitted coordinates can become non-zero after centering and bias application unless the residual mask is reapplied or constant contributions are folded into downstream parameters. For this reason, RMSNorm-based models are the cleaner initial target, while LayerNorm variants should be validated separately.

Under this interpretation, residual-width compaction is not a newly trained smaller geometry. It is a compact implementation of a masked full-width model in which removed residual coordinates are zero by construction.


### 4.5 Adaptive Threshold Learning

Rather than choosing thresholds manually, the system treats threshold selection as an optimization problem. The objective is to minimize the shadow model's size while keeping its behavior close to the full model on the user's distribution.

During calibration, both the full model and the candidate shadow model are evaluated on the same prompts. Their next-token distributions are compared using KL divergence or a related distributional distance:

```
D_KL(P_full(. | x) || P_shadow(. | x)) < ε
```

If divergence remains below the target error budget, the threshold may be increased to remove more components. If divergence exceeds the error budget, the threshold is lowered or additional components are restored.

The practical optimization target is:

```
minimize     size(S_user)
subject to   E_x[D(P_full(. | x), P_shadow(. | x))] <= ε
             fallback_error_rate <= δ
             task_quality_drop <= γ
```

This framing makes the shadow model an empirically validated approximation rather than an assumed exact copy of the full model.


### 4.6 Closed-Loop Shadow Calibration

After a candidate shadow is constructed, it is not assumed to be valid solely because its components were selected by activation magnitude. During calibration, the same prompts are evaluated by both the full model and the candidate shadow. Their output distributions are compared, and intermediate hidden states may also be compared when measuring residual-stream drift.

If the candidate shadow diverges beyond the allowed tolerance, the selection threshold is lowered or additional components are restored. If the candidate remains within tolerance, the threshold can be raised to search for a smaller shadow. The accepted shadow is therefore the smallest candidate, within the tested threshold family, that satisfies the divergence constraint on the activity-specific validation set.

This calibration step is primarily part of the experimental and growth phase. Once a shadow has been accepted, it does not need to be recalibrated on every request unless the user's activity profile changes, the shadow begins to fail validation checks, or the base model is updated.


### 4.7 Convergence and Staleness Management

The shadow model should be considered converged only when measurable indicators stabilize. Possible convergence criteria include:

- the number of newly selected components per request approaches zero;
- KL divergence against the full model remains below the target threshold on a validation buffer;
- the fallback rate falls below a target value;
- task-level quality metrics remain within an acceptable margin;
- the selected component set remains stable over a moving window of interactions.

A shadow model derived from one version of a base model may become stale when the base model changes. The system should store metadata linking each shadow model to the exact base model artifact used to construct it.

Relevant metadata may include:

```
Shadow metadata:
  base_model_id: llama-3.2-3b-instruct
  base_model_version: vX.Y.Z
  tokenizer_hash: ...
  architecture_hash: ...
  layer_hashes: ...
  shadow_profile_timestamp: ...
```

When the base model changes, the shadow model should not be assumed valid. Possible strategies include:

1. invalidate the shadow and rebuild from the user's activation history;
2. refresh only affected layers if the architecture and changed weights allow it;
3. temporarily force full-model routing until the shadow passes validation again.

Layer-level hashes can help determine which parts of a shadow model might remain reusable, but revalidation is still required after any base-model update.

---


## 5. Expected Properties and Limitations

### Expected Properties

**User-specific compression.** The selected components are optimized for a particular user's observed distribution rather than for a generic benchmark.

**Multiple shadows per user.** A user does not need to be represented by a single monolithic shadow model. If their workload contains distinct recurring activities, such as programming, writing, research, or administrative tasks, the system can maintain separate activity-specific shadows and route each request to the most appropriate one.

**Dynamic adaptation.** The user profile can grow and change over time as new requests are observed.

**No full offline distillation required.** The base proposal relies on activation profiling and structured component selection, not on training a separate student model from scratch.

**Fallback-based safety mechanism.** The full model remains available for uncertain, unfamiliar, or out-of-profile requests.

**Potential local deployment.** If a shadow model becomes sufficiently compact and can be materialized as a dense model, it may be suitable for lower-latency or on-device inference.

### Limitations

**No exact correctness guarantee.** Copying or retaining activated components does not guarantee equivalence to the full model. The system must rely on empirical validation and conservative routing.

**Transformer interactions are complex.** Attention heads, MLP channels, residual streams, normalization layers, and output projections interact in ways that may make naive component removal unstable.

**Dense compaction is nontrivial.** A structured mask is easier to build than a standalone dense model. Converting a validated mask into a compact model requires additional engineering and validation.

**Confidence estimation is imperfect.** Low entropy does not necessarily imply correctness. Routing must combine multiple signals and should be evaluated for false accepts.

**Activity discovery and shadow selection.** Broad or highly variable users may not converge to a single compact activation profile, but this does not rule out personalization. Their behavior may instead decompose into multiple recurring activity profiles, each with its own shadow model. The challenge is to discover these activity clusters, maintain separate shadows, and route requests to the correct shadow without increasing complexity or false accepts.

---


## 6. Differences from Prior Work

| Property | Static pruning | Distillation | MoE | Adapters / fine-tuning | This proposal |
|---|---:|---:|---:|---:|---:|
| Personalized per user | Usually no | Usually no | No | Yes | Yes |
| Dynamic from real user behavior | Usually no | Usually no | No | Sometimes | Yes |
| Uses internal activations as primary signal | Sometimes | No | Router-dependent | No | Yes |
| Requires gradient updates | No / sometimes | Yes | During training | Yes | Not in base version |
| Has full-model fallback | No | No | Not usually | No | Yes |
| Can adapt as usage changes | Limited | Requires retraining | Limited | Yes | Yes |
| Goal is smaller standalone inference model | Sometimes | Yes | No | No | Eventually |

The closest related areas are activation-aware pruning, structured pruning, dynamic sparse inference, model routing, semantic caching, and personalized adaptation. The proposed contribution is the combination of user-specific activation profiling, dynamic component selection, empirical divergence validation, and fallback-based deployment.

---


## 7. Experimental Plan

A credible evaluation should begin with a modest open-source model and a controlled set of user profiles.


### 7.1 Base Models

Possible starting points:

- TinyLlama or another small transformer for rapid iteration;
- LLaMA 3.2 1B or 3B for more realistic behavior;
- a larger model only after the mechanism is validated at smaller scale.


### 7.2 User Profiles

Define several synthetic or collected user profiles, such as:

1. Python developer;
2. frontend developer;
3. creative writer;
4. math tutor;
5. general assistant;
6. mixed-domain user.

Each profile should include a growth set, an in-domain validation set, and an out-of-domain test set.


### 7.3 Shadow Construction Variants

Evaluate multiple variants:

1. MLP-channel selection only;
2. attention-head selection only;
3. combined MLP and attention selection;
4. global threshold selection;
5. per-layer threshold selection;
6. closed-loop threshold calibration;
7. zero-imputed normalization for residual-width compaction;
8. structured mask implementation;
9. compact dense implementation, if feasible.


### 7.4 Metrics

The evaluation should report:

- KL divergence between full and shadow next-token distributions;
- perplexity or loss on in-domain validation prompts;
- task-level quality where applicable;
- shadow size as a percentage of the full model;
- memory reduction;
- latency and tokens per second;
- fallback rate;
- false accept rate, where the shadow answers but diverges significantly from the full model;
- residual-stream or hidden-state drift between the full model and the shadow;
- convergence speed measured in number of user interactions;
- stability of the selected component set over time.


### 7.5 Baselines

The shadow model should be compared against:

- random structured pruning at the same size;
- magnitude-based pruning;
- activation-aware pruning on a general calibration set;
- activation-aware pruning on the user's calibration set;
- a small off-the-shelf model;
- quantized full model;
- semantic cache plus full-model fallback;
- optional distilled student model, if resources allow.


### 7.6 Proposed Validation Procedure

```
1. Select a base transformer model.
2. Define several user profiles.
3. Run N profile-specific prompts through the full model.
4. Record structured activation statistics.
5. Construct candidate shadow models at multiple compression ratios or threshold values.
6. Calibrate candidates in closed loop by comparing them against the full model on the activity-specific validation set.
7. Accept the smallest candidate whose output divergence and optional hidden-state drift remain within tolerance.
8. Evaluate each accepted shadow on:
   a. seen prompts;
   b. held-out in-domain prompts;
   c. related but shifted prompts;
   d. out-of-domain prompts.
9. Measure quality, divergence, latency, memory, fallback rate, and false accepts.
10. Compare against pruning, quantization, cache-based, and small-model baselines.
11. Analyze whether selected components are stable and user-specific.
```

The key empirical question is not whether the shadow can exactly reproduce the full model. The key question is whether user-specific activation profiles produce better compression-quality tradeoffs than non-personalized baselines.

---


## 8. Open Challenges

### 8.1 Residual Stream Stability

Removing components may change the residual-stream representations seen by downstream layers. The proposed closed-loop calibration step is intended to measure this effect directly rather than assume it away. Candidate shadows should be accepted only when their hidden-state drift and output divergence remain below predefined tolerances on the activity-specific validation set.

This reframes residual-stream shift as a measurable stability condition: if a candidate shadow changes the residual stream too much, the threshold is lowered or additional components are restored until the candidate falls back within the accepted divergence budget.

### 8.2 Normalization Validation

The proposal does not rely on standard reduced-width normalization when residual coordinates are removed. Instead, Section 4.4 defines zero-imputed normalization: removed coordinates are treated as zeros for normalization purposes while not being materialized in memory or computation. This preserves the interpretation of the shadow as a compact implementation of a masked full-width model.

The remaining question is empirical and architecture-dependent. RMSNorm makes zero-imputed normalization relatively direct, while LayerNorm with bias terms may require post-normalization masking or bias folding. Candidate shadows should therefore validate the chosen normalization rule during closed-loop calibration rather than assuming that reduced-width normalization is equivalent to the original model.

### 8.3 Reliable Coverage Detection

Routing errors are a central risk. The most important failure mode is a false accept: the shadow model answers directly when it should have fallen back to the full model. Evaluation should explicitly measure and minimize this rate.

### 8.4 From Masked Model to Dense Model

The practical efficiency gain depends on whether the selected structure can be compiled or materialized into a smaller dense model. If the system only applies masks to the original model, memory and latency benefits may be limited. Dense compaction is therefore a key engineering challenge.

### 8.5 Privacy of User-Specific Profiles

The shadow model does not necessarily store raw user prompts or train new weights on private data. However, the personalized component mask, activation statistics, thresholds, activity clusters, routing metadata, and validation buffers are derived from user behavior. These artifacts may reveal sensitive information about the user's domains of interest, workflows, or recurring activities even if they do not contain the original prompts.

A deployable system should therefore treat user-specific shadow metadata as private user data. Raw prompts and validation buffers should be minimized or stored locally when possible, profiles should support deletion, and shadows should not be exposed to unrelated users or services.

---


## 9. Conclusion


This draft proposes dynamic personalized shadow models as a research direction for user-specific LLM inference efficiency. The central idea is to use a user's own interaction history to identify which structured components of a full model are repeatedly important, then build and validate a compact shadow model specialized for that user's workload.

The proposal is intentionally framed as a hypothesis rather than a proven guarantee. Transformer internals are highly coupled, and activation-guided component selection may or may not produce a sufficiently accurate standalone model without additional calibration. The value of the approach must therefore be established empirically through controlled experiments against strong compression and routing baselines.

If validated, personalized activation-guided shadow models could provide a useful path toward cheaper, lower-latency, and more private LLM deployment for long-term users with stable workloads.

---

*Research draft. Not peer reviewed.*
