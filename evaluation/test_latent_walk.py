#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_latent_walk.py — random walk in embedding space.

Given an anchor position P, we compute emb(P), then perturb it with a random
vector of norm epsilon, and look up the nearest position in the corpus to the
perturbed point. Sweep epsilon to see how the concept evolves as we walk away
from the anchor.

What we expect:
  - Well-structured embedding: small epsilon lands on semantically similar
    positions (same opening, similar material, similar king safety).
  - As epsilon grows, the neighbor drifts to related concepts (Najdorf ->
    another Sicilian -> another e4 opening -> another opening in general).
  - Chaotic/memorized embedding: small epsilon already lands on random stuff.

This is a qualitative figure for the paper: "the radius of Najdorf" in the
embedding space. Comparison trained vs untrained makes the effect visible.
"""
import argparse
import numpy as np
import torch

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus, compute_corpus_embeddings


ANCHORS = [
    ("Sicilian Najdorf",
     "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("King+Pawn endgame",
     "8/5k2/8/8/8/8/4PK2/8 w - - 0 1"),
    ("Italian Giuoco Piano",
     "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("Back-rank mate threat",
     "6k1/5ppp/8/8/8/8/r7/6K1 b - - 0 1"),
    ("Queen+Rook endgame",
     "8/5k2/8/8/8/8/4QK2/5r2 w - - 0 1"),
]

EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_DIRECTIONS = 3           # random directions per epsilon
TOP_K = 3                  # nearest neighbors to report
N_CORPUS = 50000


def get_emb(model, fen):
    device = next(model.parameters()).device
    x = torch.from_numpy(fen_to_bitboards(fen)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()[0]


def nearest(target_emb, corpus_embs, corpus_fens, top_k):
    d = np.linalg.norm(corpus_embs - target_emb, axis=1)
    idxs = np.argsort(d)[:top_k]
    return [(corpus_fens[i], d[i]) for i in idxs]


def walk(model, name, corpus_fens, corpus_embs, rng):
    print(f"\n  ── Anchor: {name} ──")
    e0 = get_emb(model, ANCHORS_DICT[name])
    print(f"    ANCHOR emb norm = {np.linalg.norm(e0):.2f}")

    # Self-match distance (to anchor in its own emb space)
    d_self = np.linalg.norm(corpus_embs - e0, axis=1)
    top_self = np.argsort(d_self)[:TOP_K]
    print(f"    ε=0.0  (pure anchor, top {TOP_K} real neighbors):")
    for i in top_self:
        print(f"        d={d_self[i]:>6.2f}  {corpus_fens[i]}")

    for eps in EPSILONS:
        print(f"\n    ε={eps:<4}")
        for k in range(N_DIRECTIONS):
            direction = rng.normal(size=e0.shape)
            direction /= np.linalg.norm(direction)
            target = e0 + eps * direction
            nbrs = nearest(target, corpus_embs, corpus_fens, TOP_K)
            print(f"      dir {k+1}:")
            for fen, dist in nbrs:
                print(f"        d={dist:>6.2f}  {fen}")


ANCHORS_DICT = dict(ANCHORS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--untrained", action="store_true",
                    help="Also run the untrained (random init) baseline.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  val_acc = {ckpt.get('val_acc', 0):.2f}%")

    print(f"Loading corpus ({N_CORPUS} positions)...")
    corpus_fens = load_corpus(args.corpus, max_n=N_CORPUS)
    print(f"Computing corpus embeddings (trained)...")
    corpus_embs = compute_corpus_embeddings(model, corpus_fens)

    print(f"\n{'='*70}\n  TRAINED MODEL LATENT WALK\n{'='*70}")
    for name, _ in ANCHORS:
        walk(model, name, corpus_fens, corpus_embs, rng)

    if args.untrained:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        untrained = create_model(ckpt['arch'], ckpt['n_moves'],
                                 embed_dim=ckpt['embed_dim'])
        untrained.eval()
        print(f"\nComputing corpus embeddings (untrained)...")
        corpus_embs_u = compute_corpus_embeddings(untrained, corpus_fens)
        print(f"\n{'='*70}\n  UNTRAINED MODEL LATENT WALK (control)\n{'='*70}")
        for name, _ in ANCHORS:
            walk(untrained, name, corpus_fens, corpus_embs_u, rng)


if __name__ == "__main__":
    main()
