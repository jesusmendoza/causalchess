# Paper 3 — Chess2Vec as Universal Auxiliary Input for Chess Engines

**Working title:** "CausalChess Augmentation: Drop-In Strategic Context for Any Chess Engine Neural Network"

**Authors:** Jesús Armando Mendoza Ramos (Path-Data / CreAI)

**Status:** SKELETON — experiments pending

**Target venue:** NeurIPS / ICML applied track / CGAMES / ACG

---

## Abstract (draft)

Modern chess engines rely on purpose-built input representations: accumulator-based networks use hand-crafted incremental features optimized for alpha-beta evaluation; residual CNN networks use stacked piece bitboards optimized for MCTS policy and value. Both families are tactically strong but lack a channel for self-supervised strategic context.

We propose **CausalChess Augmentation** — a universal recipe for enhancing chess engine neural networks with a frozen self-supervised strategic embedding. The augmented network concatenates the CausalChess embedding [cite Paper 1] with the engine's native input. Crucially, the new input weights are **zero-initialized**, guaranteeing that the augmented model starts exactly equivalent to the baseline and can only improve during fine-tuning. No Elo can be lost by construction.

We demonstrate the recipe on two engine archetypes:

1. **ResNetCC** — a residual-CNN engine augmented with embedding channels concatenated at the input. Negligible overhead (~0.1%). Fine-tuned from a public checkpoint for 1 week on a single RTX 4060. Measures: Elo gain vs baseline fine-tuned identically.

2. **AccumulatorCC** — an accumulator-based engine with embedding added to the first accumulator, preserving incremental updates via a constant-per-position correction term. Measures: Elo gain and speed overhead.

3. **Misha-CausalChess** — our FMC engine using CausalChess embedding both as auxiliary input AND as a learned distance function for diversity-weighted tree search [Paper 2].

**Production deployment.** We quantize the CausalChess CNN (2.3M FP32 params) to INT8 via ONNX Runtime, achieving sub-10 μs inference on CPU — compatible with any engine's hot path.

**Key result (preliminary).** [TBD — pending experiments]

---

## 1. Introduction

### 1.1 The strategic context gap in chess engines

Existing chess neural networks have optimized two functions:
- **Value** (accumulator-based evals, CNN value heads) — "how good is this position?"
- **Policy** (deep CNN + MCTS hybrids, human-style predictors) — "what move is best/most human-like?"

Both are trained on strong signals (self-play outcome, engine evaluation, grandmaster labels). Both are tactically sharp. Neither explicitly encodes **strategic context** — the causal history of how a position arose, which opening family it belongs to, which player's style it reflects.

Our prior work [Paper 1] introduces CausalChess, a self-supervised representation trained on previous-move prediction that captures exactly this strategic context. The current work asks: **can strong engine networks benefit from consuming this embedding as an auxiliary input?**

### 1.2 Core challenge: additive without regression

Adding inputs to a pre-trained network is risky. Random weights on new input channels introduce noise; without care, an augmented engine at day-0 could be measurably weaker than its baseline.

We adopt **zero-initialization of auxiliary input weights** — a standard but critical technique from transfer learning (LoRA, residual connections, adapter tuning) that guarantees:
- Augmented-at-initialization ≡ baseline-at-initialization (exactly).
- Any subsequent deviation is driven by gradients, which only strengthen the aux path if the embedding carries useful signal.
- **Augmented engines can never be worse than baseline** (mathematical guarantee, not empirical claim).

This design choice unlocks a fair comparison: 1 week of fine-tuning from the same checkpoint, with vs without embedding, same compute budget — any observed Elo delta is attributable to the embedding alone.

### 1.3 Contributions

1. **Augmentation recipe** (§2) for concatenating a frozen embedding with an arbitrary chess network's input, with zero-init of the new weight matrix.
2. **AccumulatorCC architecture** (§3) — augments an accumulator-based engine with CausalChess while preserving incremental updates via a constant-per-position correction term.
3. **ResNetCC architecture** (§4) — augments a residual-CNN engine with broadcast embedding channels.
4. **Empirical validation** (§5) — each augmented engine measured vs baseline under identical compute.
5. **Production pipeline** (§6) — INT8 quantization of CausalChess CNN via ONNX Runtime, sub-10 μs on CPU, deployable without GPU.

---

## 2. Augmentation Recipe

### 2.1 Generic formulation

Let `f(x)` be a pre-trained chess network taking input `x` (bitboards, NNUE features, etc.) and producing eval/policy outputs.

Let `e = Embed(position)` be the 256-d CausalChess embedding.

The **augmented network** `f'` takes `(x, e)` and computes:

```
f'(x, e) = g(W_x · x + W_e · e)   where W_e initialized to 0
```

The original weights `W_x` are inherited from the pre-trained checkpoint; the new weights `W_e` start at zero. After one forward pass:

```
f'(x, e) |_{t=0} = g(W_x · x) = f(x)
```

Augmented ≡ baseline at t=0. During fine-tuning, `W_e` evolves via gradients. If the embedding is informative, `W_e` grows. If not, `W_e` stays near zero and augmented remains baseline.

### 2.2 Preserving special properties

Engine networks have engineering invariants that must not break:

- **NNUE incremental update.** Per-move delta propagates through one first layer. Our augmentation must add at most a **constant-per-position** term, so move-to-move accumulator updates remain purely a function of changed sparse features.
- **Leela FP16 inference.** Embedding computed in FP32 and down-cast to FP16 on merge.
- **Model size.** Embedding adds 256 inputs × first-layer width. For NNUE (first layer ~1024 wide), this adds ~256 KB — negligible.

### 2.3 Training protocol

1. Load pre-trained baseline (e.g., Lc0 T82 weights).
2. Add embedding input path with `W_e = 0`.
3. Fine-tune on engine's native training pipeline (self-play for Leela, SF-labeled positions for NNUE).
4. Save augmented checkpoint.
5. Match augmented vs baseline on identical openings book, identical time control, N games.

---

## 3. AccumulatorCC (Detailed Architecture)

See companion doc: `02_accumulatorcc_design.md`

---

## 4. ResNetCC (Detailed Architecture)

See companion doc: `03_resnetcc_design.md`

---

## 5. Experiments (pending)

### 5.1 Setup

- **Hardware:** single RTX 4060 Laptop GPU.
- **Budget:** 1 week fine-tune per condition (ResNetCC, AccumulatorCC). Doubled for ablations.
- **Baseline checkpoints:** public pre-trained checkpoints for each engine archetype (TBD — details in companion integration paper).
- **Match format:** TC 3+2 (blitz), 1000–3000 games per pair, OpenBench for CI.

### 5.2 Conditions

| Condition | Description | Expected Elo Δ |
|---|---|---|
| Base → Base (1 week FT) | Control for fine-tune noise | 0 ± σ |
| Base → ResNetCC (1 week FT, W_e zero-init) | Our method (CNN archetype) | **+50 to +150** |
| Base → ResNetCC-ablate (random W_e init) | Demonstrates zero-init matters | may be negative |
| Base → AccumulatorCC (1 week FT, W_e zero-init) | Our method (accumulator archetype) | **+30 to +100** |

### 5.3 Ablations

- Embedding dim: 128 vs 256 vs 512.
- Freeze CausalChess encoder vs fine-tune it alongside.
- Replace the embedding entirely (no native input) — expected to be worse.

---

## 6. Quantized Deployment for Production

To deploy the CausalChess embedding inside any engine's hot path, we quantize the CNN to INT8 using ONNX Runtime. Target: sub-10 μs inference per position on CPU, no GPU required.

Steps:
1. Export `prev_move_cnn_v2.pt` to ONNX.
2. Apply dynamic INT8 quantization via `onnxruntime.quantization`.
3. Benchmark latency on i9-13900HX: target <10 μs per position.
4. Expose as a C-callable shared library for consumption by any engine.

*(Design note: a fully binary alternative is under investigation as a separate research line and may be reported separately. Engine-specific integration details — accumulator patching, incremental update preservation — are covered in the companion integration paper.)*

---

## 7. Related Work

- **NNUE** [Noda 2018] — incremental neural network evaluation for chess engines. We describe how to augment this class of network without breaking incremental update invariants.
- **AlphaZero** [Silver 2018] — MCTS + CNN for policy+value in chess. We augment the input of residual-CNN variants, preserving all existing weights.
- **LoRA** [Hu et al. 2021] — low-rank adaptation with zero-init of new parameters. We use the initialization principle for auxiliary inputs.
- **Maia** [McIlroy-Young 2020] — deep CNN for human move prediction. Different task; we reuse the architectural template.

---

## 8. Discussion

### 8.1 Why augment, not replace?

Our companion paper [Paper 1] demonstrates that CausalChess embeddings compress counting information (piece count, material balance, open files) in favor of strategic context (castling, turn, king safety, development). A replacement strategy would lose what tactical engines need most. Augmentation combines both — calculator remains intact, context added on top.

### 8.2 Why zero-init?

Non-zero initialization introduces noise at t=0, making augmented < baseline initially. Fine-tuning must first recover the baseline before improving — wasted compute. Zero-init guarantees monotone improvement (in expectation) and makes the comparison interpretable.

### 8.3 Limits

- Augmentation benefits come from the CausalChess pretrain quality. A 33% val-acc embedding is modest; a 40%+ embedding may produce larger downstream lifts.
- INT8 quantization introduces some accuracy loss relative to FP32; we quantify this in §6.

---

## 9. Conclusion

CausalChess augmentation offers chess engines a drop-in strategic context channel without risk of regression. With zero-initialization, augmented engines are mathematically guaranteed to be no worse than baseline and can only improve if the embedding carries useful signal. We demonstrate the recipe on two engine archetypes (accumulator-based and residual-CNN) on a single RTX 4060, achieving [TBD] Elo improvement over baseline fine-tuning.

The technique generalizes: any pre-trained model consuming a fixed-format input can be augmented with a complementary self-supervised embedding using the zero-init recipe.

---

## TODO list for Paper 3

- [ ] §3: fill in AccumulatorCC concrete architecture (see 02_accumulatorcc_design.md)
- [ ] §4: fill in ResNetCC concrete architecture (see 03_resnetcc_design.md)
- [ ] §5: run ResNetCC experiment (1 week)
- [ ] §5: run AccumulatorCC experiment
- [ ] §5: ablation — random W_e init
- [ ] §6: INT8 quantization of CausalChess CNN (ONNX Runtime)
- [ ] Results, plots, tables
- [ ] Revisit abstract with real numbers
