# LCC — Leela + CausalChess Design Doc

## Goal

Augment Leela's CNN (policy+value heads) with the CausalChess embedding as auxiliary input, starting from an existing Lc0 checkpoint, with zero-init of new weights and 1 week of fine-tuning on a single RTX 4060.

## Leela recap (relevant parts)

Lc0 uses AlphaZero-style architecture:
- Input: 112-plane tensor (8×8×112) encoding board + recent history + rules state
- Deep residual CNN (e.g., T82: 15 blocks × 768 filters, or T80: 10 blocks × 512 filters)
- Two heads:
  - **Policy head:** 1858-dim output (all legal move candidates)
  - **Value head:** 3-dim output (win/draw/loss probabilities)
- Training: self-play with MCTS exploration, ~1B positions for top networks

Inference cost per position: 10-50 ms on laptop GPU. Dominant cost in MCTS.

## Augmentation

### Math

Leela input is a 8×8×112 tensor. We add 256 channels by broadcasting the CausalChess embedding:

```
original_input: 8×8×112 (piece planes + history)
embedding_input: 256-d vector → broadcast to 8×8×256 (tiled across spatial dims)
augmented_input: 8×8×(112+256) = 8×8×368
```

The first conv layer of Leela expands to accept 368 channels:
```
conv1_weights: (filters × 368 × 3 × 3)
  = concat along input-channel axis:
    - conv1_original: (filters × 112 × 3 × 3) ← INHERITED from baseline
    - conv1_aux:      (filters × 256 × 3 × 3) ← NEW, zero-initialized
```

### Property

At t=0, conv1_aux = 0. The embedding contribution `conv1_aux × embedding_broadcast = 0`. The augmented network produces identical output to baseline Leela.

During fine-tuning, gradients flow into conv1_aux. If the embedding provides useful spatial context, conv1_aux grows. Otherwise it stays at zero.

### Alternative: broadcast into deeper layer

Instead of expanding conv1, inject the embedding into a later residual block via a residual-add path. Equivalent conceptually but avoids enlarging conv1 parameters. Either works — conv1 expansion is simpler to implement.

## Integration

### Codebase targets

- **Leela training:** `LeelaChessZero/lczero-training` (Python + TF)
- **Leela inference:** `LeelaChessZero/lc0` (C++ + CUDA/OpenCL)

### Training pipeline

```python
# Pseudocode
from causal_chess import CausalChessEncoder

# Training data: position + move_probs + outcome (from Leela self-play)
# Add: embedding for each position

for batch in dataloader:
    position, target_policy, target_value = batch
    embedding = causal_chess_encoder(position).detach()  # frozen
    aug_input = augment_input(position.board_planes, embedding)
    policy, value = lcc_model(aug_input)
    loss = policy_loss(policy, target_policy) + value_loss(value, target_value)
    loss.backward()
    optimizer.step()
```

### Inference pipeline

In lc0:
- Keep existing eval pipeline.
- Add one hook: compute CausalChess embedding per position (~5 μs with INT8-quantized CNN, ~50 ms with FP32 CNN).
- Concatenate into conv1 input.

Overhead per eval: +5 μs (INT8 CNN) on top of 10-50 ms (Leela eval) = **0.01-0.05%**. Negligible.

## Zero-init protocol

```
conv1_aux = zeros (filters × 256 × 3 × 3)
conv1_original = inherited from Lc0 T82 checkpoint
all other weights = inherited from Lc0 T82
embedding network = frozen, from CausalChess Paper 1 checkpoint
```

## Training

**Data:** self-play games from Lc0 T82 training pipeline. Re-run one week of training but with augmented input.

**Baseline condition.** Also fine-tune baseline Lc0 for one week (no embedding) — control for fine-tune noise.

**Hyperparameters:** inherit from Lc0 training config. Only change: input dim.

**Compute estimate:** 
- Lc0 T82 full training: ~3 months on a GPU cluster
- Our fine-tune budget: 1 week on RTX 4060
- What we can achieve: maybe 1% of the original training budget, enough to see if augmentation provides signal.

## Evaluation

**Matches.** 
- LCC (fine-tuned 1 week) vs baseline Lc0 T82 (no fine-tune)
- LCC (fine-tuned 1 week) vs baseline Lc0 T82 (fine-tuned 1 week, same compute, no embedding)

Second match is critical — shows LCC benefits specifically from the embedding, not just from extra fine-tuning time.

- Time control: 3+2 or 5+3
- Games: 2000-5000 for tight CI
- Tool: OpenBench or cutechess-cli

**Expected.** +50 to +150 Elo over matched-compute baseline. Leela benefits more than NNUE because MCTS exploration can leverage strategic context for diversity.

## Risks

- **Fine-tune doesn't move baseline much in 1 week.** Possible 4060 produces no change — baseline = baseline + 1 week. If so, we can't distinguish fine-tune-nothing from augmentation-nothing. Mitigation: very tight CI on both conditions.
- **Embedding not helpful for Leela.** Leela's CNN already sees the whole board; maybe the embedding adds nothing. Mitigation: paper reports honestly, with probe evidence that strategic info is there (Paper 1).
- **Compute ceiling.** 1 week of 4060 may be too little to see a signal. Mitigation: find collaborator with bigger GPU for final numbers.

## Status

- Design complete (this document)
- Code fork: NOT STARTED (target: `lczero-training` fork)
- Training: NOT STARTED
- Match: NOT STARTED

## Dependencies

- Lc0 source (public, GPL)
- lczero-training source (public)
- Lc0 T82 checkpoint (public download)
- CausalChess CNN checkpoint (from Paper 1, v2)
- INT8-quantized CNN (for production-speed inference)
- ~1 week continuous GPU time on 4060

## Risk assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Augmented model matches baseline (no improvement) | medium | medium | Still valid negative result; gradients showed no useful signal |
| Augmented model is worse due to impl bug | medium | high | Unit tests on zero-init → output parity with baseline |
| Compute insufficient | high | low | Set expectations; 1 week is a starting point not a final answer |
| Baseline fine-tune decays | low | medium | Snapshot frequently; compare to original T82 |
| Embedding inference too slow | medium | low | INT8 quantization (ONNX Runtime) |
