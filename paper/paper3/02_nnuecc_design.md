# NNUECC — NNUE + CausalChess Design Doc

## Goal

Augment Stockfish's NNUE evaluation network with the CausalChess embedding as auxiliary input, **preserving incremental updates** and minimal overhead in alpha-beta search.

## NNUE recap (relevant parts)

Stockfish 16+ uses HalfKAv2-hm features:
- Input: ~45K sparse binary features (king-piece-square encoding)
- Only a few features change per move → incremental accumulator update
- First layer: `accumulator = W_nnue · x_sparse` (only changed features update accumulator)
- Rest of network: clipped-ReLU, integer quantized, small (~4 hidden layers, ~512 wide)

Key speed property: **move-to-move eval is O(changed features)**, not O(all features).

## Augmentation

We want to add a 256-d CausalChess embedding `e` to the accumulator **without breaking incrementality.**

### Math

Original:
```
accumulator = W_nnue · x_sparse           ← incremental over moves
eval = NNUE_head(accumulator)
```

Augmented:
```
accumulator = W_nnue · x_sparse + W_e · e
             ↑ still incremental           ↑ CONSTANT for a given position
```

Key observation: `W_e · e` depends only on the **current position**, not on the move. So within a subtree rooted at position P:
- Compute `c_P := W_e · Embed(P)` once when entering P.
- Every descendant eval uses `accumulator + c_P`.
- As moves are made, `accumulator` updates incrementally (as before). `c_P` is a constant added at the final layer.

### Architecture

```
Input stream:
  x_sparse (45K sparse binary features, incremental)
  embedding (256-d FP32, computed once per position by CausalChess CNN)
        ↓
First accumulator (1024-d):
  acc = W_nnue × x_sparse  +  W_e × embedding
       ↑ inherited from Stockfish       ↑ new, zero-initialized
        ↓
Clipped-ReLU, subsequent NNUE hidden layers (unchanged)
        ↓
Eval (centipawn score)
```

### Integration in search

For each node visited in alpha-beta search:

```c
// Pseudocode
void on_enter_node(Node* n) {
    if (embedding_cached(zobrist(n))) {
        c_n = cached_correction(zobrist(n));
    } else {
        e = CausalChess_embed(n->position);    // ~3-10 μs with INT8-quantized CNN
        c_n = W_e * e;                         // 256 → 1024 matmul, ~1 μs
        cache_correction(zobrist(n), c_n);     // store in TT
    }
    current_correction = c_n;
}

int32_t eval(Node* n) {
    return nnue_head(accumulator + current_correction);
}
```

**Overhead per new position:** ~5-15 μs (embedding + correction compute).
**Amortized over transposition table hits:** effectively 1-2 μs per eval (assuming 80% hit rate in search).

## Zero-init protocol

```
W_e initialized to zeros (256 × 1024 matrix of zeros)
All other NNUE weights: inherit from baseline Stockfish 16+ NNUE
```

Day 0: `W_e · e = 0` → augmented NNUE produces identical eval to baseline.

During fine-tuning: `W_e` learns from gradient signal. Grows if embedding provides useful information for eval refinement.

## Training

**Data.** Stockfish depth-20 labeled positions (same as base NNUE training). 100M+ positions available publicly.

**Objective.** Mean squared error between NNUECC eval and Stockfish depth-20 eval (same as baseline NNUE).

**Hyperparameters.** Inherit from Stockfish nnue-pytorch training repo. Only change: include the embedding input.

**Compute estimate.** 1 day on RTX 4060 for a ~100M position fine-tune (NNUE is tiny — training is data-bound, not compute-bound).

## Evaluation

Tournament: NNUECC Stockfish vs baseline Stockfish 16.1.
- Same time control (5+3 typical SF testing).
- Same opening book (UHO 2.0 or similar).
- 5K+ games → CI ±5 Elo.

**Expected result.** +20 to +100 Elo. NNUE is already near-optimal for SF's alpha-beta, so gains are smaller than for Leela (where exploration can benefit more from strategic context).

## Risks

- **Incremental performance hit.** Computing CausalChess embedding on every new position in search adds overhead. Mitigation: INT8-quantized CNN, TT caching.
- **No signal found.** Possible the gradient finds nothing useful in embedding for NNUE → W_e stays zero → NNUECC = baseline. Not a failure mode, just uninteresting result.
- **NNUE quantization.** NNUE is integer-quantized for speed. W_e will need quantization too. Standard procedure.

## Status

- Design complete (this document)
- Code fork: NOT STARTED
- Training: NOT STARTED
- Match: NOT STARTED

## Dependencies

- Stockfish source (public, GPL)
- nnue-pytorch trainer (public)
- CausalChess CNN checkpoint (from Paper 1)
- INT8-quantized CNN (for production, not training)
