#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_strategic_distance.py — Correlate embedding distance with Stockfish
evaluation difference ("strategic distance").

For two positions A, B:
  d_emb(A, B)  = ||embedding(A) - embedding(B)||₂
  d_eval(A, B) = |eval_SF(A) - eval_SF(B)|   (centipawns)

A strategically meaningful embedding should satisfy:
  d_emb and d_eval are positively correlated (positions with very different
  evals are far in embedding space; positions with similar evals are close).

We compute Pearson and Spearman correlation over N² pairs (or a random
sub-sample for speed). Compare against:
  (a) untrained-CNN baseline
  (b) raw-bitboard Hamming distance (piece-placement sanity check)

Input: (fen, eval_cp) TSV produced by compute_stockfish_evals.py.
"""
import argparse
import os
import sys
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import compute_corpus_embeddings


def load_eval_tsv(path, max_n=0):
    fens, evals = [], []
    with open(path) as f:
        header = f.readline().strip().split('\t')
        i_fen = header.index('fen')
        i_eval = header.index('eval_cp')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < max(i_fen, i_eval) + 1:
                continue
            try:
                e = float(parts[i_eval])
            except ValueError:
                continue
            fens.append(parts[i_fen])
            evals.append(e)
            if max_n > 0 and len(fens) >= max_n:
                break
    return fens, np.array(evals)


def pairwise_sample(n, n_pairs, rng):
    """Random (i, j) pairs with i != j."""
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    return i[keep], j[keep]


def compute_correlations(d_emb, d_eval, label):
    r, _ = pearsonr(d_emb, d_eval)
    rho, _ = spearmanr(d_emb, d_eval)
    print(f"  {label:<30}  Pearson r = {r:+.4f}   Spearman ρ = {rho:+.4f}")
    return r, rho


def hamming_distance_bitboards(fens_a, fens_b):
    """Raw Hamming distance between two sets of 768-bit position vectors."""
    bits_a = np.stack([fen_to_bitboards(f) for f in fens_a])
    bits_b = np.stack([fen_to_bitboards(f) for f in fens_b])
    return (bits_a != bits_b).sum(axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--evals", default="lichess_data/stockfish_evals.tsv",
                    help="TSV produced by compute_stockfish_evals.py")
    ap.add_argument("--n", type=int, default=5000,
                    help="Use first N positions from the eval TSV.")
    ap.add_argument("--n-pairs", type=int, default=100000,
                    help="Number of random (A,B) pairs to correlate.")
    ap.add_argument("--clip-eval", type=float, default=2000.0,
                    help="Clip |eval_cp| to this (large values dominate corr).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.evals):
        print(f"ERROR: eval TSV not found: {args.evals}")
        print(f"Run compute_stockfish_evals.py first to generate it.")
        sys.exit(1)

    print(f"Loading Stockfish evals from {args.evals}...")
    fens, evals = load_eval_tsv(args.evals, max_n=args.n)
    # Clip extreme evals (mate scores etc.)
    evals_clip = np.clip(evals, -args.clip_eval, args.clip_eval)
    print(f"  {len(fens)} positions, eval range [{evals.min():.0f}, {evals.max():.0f}] cp, "
          f"clipped to ±{args.clip_eval:.0f}")

    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get('arch', 'cnn')
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  arch={arch}, val_acc={ckpt.get('val_acc', 0):.2f}%")

    print(f"\nComputing {len(fens)} embeddings (trained)...")
    emb_t = compute_corpus_embeddings(model, fens)

    print("Computing embeddings (untrained baseline)...")
    torch.manual_seed(args.seed)
    untrained = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    untrained.eval()
    emb_u = compute_corpus_embeddings(untrained, fens)

    rng = np.random.default_rng(args.seed)
    i, j = pairwise_sample(len(fens), args.n_pairs, rng)
    print(f"\nSampled {len(i):,} (A, B) pairs.")

    d_eval = np.abs(evals_clip[i] - evals_clip[j])
    d_emb_t = np.linalg.norm(emb_t[i] - emb_t[j], axis=1)
    d_emb_u = np.linalg.norm(emb_u[i] - emb_u[j], axis=1)

    print("\n" + "=" * 70)
    print("  Correlation: embedding distance vs |eval(A) - eval(B)|")
    print("=" * 70)
    r_t, rho_t = compute_correlations(d_emb_t, d_eval, "Trained embedding")
    r_u, rho_u = compute_correlations(d_emb_u, d_eval, "Untrained (control)")

    # Raw-bitboard Hamming as sanity check — does piece-count alone correlate?
    print(f"\nComputing Hamming on {min(10000, len(i)):,} pairs (raw bitboard sanity)...")
    max_hamming = min(10000, len(i))
    d_hamming = hamming_distance_bitboards([fens[k] for k in i[:max_hamming]],
                                           [fens[k] for k in j[:max_hamming]])
    r_h, rho_h = compute_correlations(d_hamming, d_eval[:max_hamming],
                                       "Hamming-bitboard (sanity)")

    print("=" * 70)
    print(f"\n  LIFT trained - untrained:")
    print(f"    Pearson  : {r_t - r_u:+.4f}")
    print(f"    Spearman : {rho_t - rho_u:+.4f}")


if __name__ == "__main__":
    import sys
    main()
