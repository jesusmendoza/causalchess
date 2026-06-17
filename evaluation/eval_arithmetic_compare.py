#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
eval_arithmetic_compare.py — Run arithmetic tests on CNN (trained) vs CNN (untrained/random).

Baseline (untrained): should produce noise — no meaningful chess structure.
Trained CNN (33.15% val acc): should produce strategic midpoints, material symmetry, etc.

Saves both outputs to files for side-by-side comparison.
"""
import sys
import os
import torch
import numpy as np
from prev_move_models import create_model
from eval_arithmetic import (
    load_corpus, compute_corpus_embeddings,
    run_arithmetic_tests, run_sum_tests
)


def make_untrained(arch, n_moves, embed_dim):
    """Create CNN architecture with random initialization (no checkpoint load)."""
    model = create_model(arch, n_moves, embed_dim=embed_dim)
    model.eval()
    # Use fixed seed for reproducibility
    return model


def load_trained(arch, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('val_acc', 0), ckpt['n_moves'], ckpt['embed_dim']


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    tsv = "lichess_data/prev_move_2400.tsv"
    print("Loading corpus (50k positions)...", flush=True)
    corpus_fens = load_corpus(tsv, max_n=50000)
    print(f"  Loaded {len(corpus_fens)} positions\n", flush=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    args = ap.parse_args()

    # ── Trained CNN ──────────────────────────────────
    path = args.ckpt
    _peek = torch.load(path, map_location="cpu", weights_only=False)
    arch = _peek.get('arch', 'cnn')  # use the arch stored in the checkpoint

    print("=" * 70)
    print("  PART 1: TRAINED CNN (33.15% val acc)")
    print("=" * 70)

    trained, val_acc, n_moves, embed_dim = load_trained(arch, path)
    print(f"  Loaded {arch}, val_acc={val_acc:.2f}%, n_moves={n_moves}, embed={embed_dim}")
    print(f"  Computing corpus embeddings...", flush=True)
    corpus_embs_trained = compute_corpus_embeddings(trained, corpus_fens)

    run_arithmetic_tests(trained, f"CNN trained (val={val_acc:.1f}%)",
                         corpus_fens, corpus_embs_trained)
    run_sum_tests(trained, f"CNN trained (val={val_acc:.1f}%)",
                  corpus_fens, corpus_embs_trained)

    # ── Untrained CNN (random baseline) ──────────────
    print("\n\n")
    print("=" * 70)
    print("  PART 2: UNTRAINED CNN (random weights, should be NOISE)")
    print("=" * 70)

    untrained = make_untrained(arch, n_moves, embed_dim)
    print(f"  Created random-init {arch}, n_moves={n_moves}, embed={embed_dim}")
    print(f"  Computing corpus embeddings...", flush=True)
    corpus_embs_untrained = compute_corpus_embeddings(untrained, corpus_fens)

    run_arithmetic_tests(untrained, "CNN UNTRAINED (random)",
                         corpus_fens, corpus_embs_untrained)
    run_sum_tests(untrained, "CNN UNTRAINED (random)",
                  corpus_fens, corpus_embs_untrained)


if __name__ == "__main__":
    main()
