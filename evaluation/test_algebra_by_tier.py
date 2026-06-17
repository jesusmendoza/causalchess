#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_algebra_by_tier.py — chess arithmetic broken down by position uniqueness.

Key concern: arithmetic midpoints (Sicilian + Queen's Gambit = English) may
work ONLY on positions heavily seen in training (memorization). If midpoint
accuracy collapses on positions seen only a few times, the "algebra" is just
memorized table lookup.

For each position in the corpus, we count how often its bitboard pattern
appears in the training set. Positions are binned into tiers by frequency:

    Tier 1: appears 100+ times in training (openings, mainstream theory)
    Tier 2: 10-99 times
    Tier 3: 2-9 times
    Tier 4: 1 time (unique middlegame/endgame)

Algebra test (midpoint): for triplet (A, B, C) such that A+C=B (e.g., the
midpoint of two related positions approximates a third), we check if
emb(A+C)/2 is close to emb(B) in cosine similarity.

We report midpoint accuracy per tier. If Tier 4 works -> learned real
structure. If only Tier 1 works -> memorized opening tree.
"""
import argparse
from collections import Counter
import numpy as np
import torch

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus, compute_corpus_embeddings


N_CORPUS = 50000
TIERS = [
    ("Tier 1 (100+)", 100, 10**9),
    ("Tier 2 (10-99)", 10, 100),
    ("Tier 3 (2-9)", 2, 10),
    ("Tier 4 (unique)", 1, 2),
]


def bitboards_key(fen):
    """Position identity by placement (12 bitboards), same granularity as the
    CNN input. Ignores side-to-move / castling / en-passant (those aren't in x).
    """
    bits = fen_to_bitboards(fen)
    return bytes(bits.astype(np.uint8))


def build_train_frequency_map(train_x_path, n_train):
    """Count how many times each 12-bitboard key appears in training set."""
    print(f"Loading train bitboards from {train_x_path}...")
    x = np.memmap(train_x_path, dtype=np.uint64, mode='r', shape=(n_train, 12))
    counter = Counter()
    chunk = 500_000
    for start in range(0, n_train, chunk):
        end = min(start + chunk, n_train)
        block = np.asarray(x[start:end])
        for row in block:
            counter[row.tobytes()] += 1
        print(f"  {end}/{n_train}", flush=True)
    print(f"  {len(counter):,} unique positions in train")
    return counter


def tier_of(count):
    for name, lo, hi in TIERS:
        if lo <= count < hi:
            return name
    return None


def fen_key_from_bitboards_key(fen):
    """Converts fen -> 12 uint64 bitboards (same packing as train chess_x.bin),
    returns bytes key.
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    boards = np.zeros(12, dtype=np.uint64)
    sq = 56
    for ch in fen.split()[0]:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            idx = piece_map.get(ch)
            if idx is not None:
                boards[idx] |= np.uint64(1) << np.uint64(sq)
            sq += 1
    return boards.tobytes()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest_cos(target, corpus_embs, corpus_fens, top_k=5):
    """Return top_k (fen, cosine_sim) closest to target by cosine."""
    t_norm = target / (np.linalg.norm(target) + 1e-12)
    c_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-12)
    sims = c_norm @ t_norm
    idxs = np.argsort(-sims)[:top_k]
    return [(corpus_fens[i], float(sims[i])) for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--train-x", default="data/chess_x.bin")
    ap.add_argument("--train-meta", default="data/chess_meta.txt")
    ap.add_argument("--n-pairs", type=int, default=500,
                    help="Number of (A,B) random pairs to test midpoint on.")
    args = ap.parse_args()

    # Load train meta
    with open(args.train_meta) as f:
        meta = dict(line.strip().split('=') for line in f if '=' in line)
    n_train = int(meta['n_samples'])

    # Build frequency map
    freq = build_train_frequency_map(args.train_x, n_train)

    # Load model
    print(f"\nLoading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model']); model.eval()

    # Load corpus and classify each position into a tier
    print(f"Loading {N_CORPUS} corpus positions...")
    corpus_fens = load_corpus(args.corpus, max_n=N_CORPUS)
    print("Classifying corpus positions into tiers...")
    tier_idx = {name: [] for name, _, _ in TIERS}
    for i, fen in enumerate(corpus_fens):
        key = fen_key_from_bitboards_key(fen)
        t = tier_of(freq.get(key, 1))
        if t is not None:
            tier_idx[t].append(i)

    print("\nCorpus distribution by tier:")
    for name, _, _ in TIERS:
        print(f"  {name:<18}: {len(tier_idx[name]):>6,} positions")

    print(f"\nComputing embeddings...")
    corpus_embs = compute_corpus_embeddings(model, corpus_fens)

    # Midpoint test: for each tier, sample random pairs (A, B). Compute
    # midpoint embedding (emb(A)+emb(B))/2. Find its nearest neighbor C in
    # the corpus. Report: cosine(mid, emb(C)). If C == A or B, mark trivial.
    # Average cosine per tier, and 'non-trivial rate' (fraction where C is
    # different from A and B).
    rng = np.random.default_rng(42)
    print(f"\nMidpoint test — {args.n_pairs} random pairs per tier:")
    print(f"  {'Tier':<18}  {'mean cos':>9}  {'nontriv rate':>13}  {'self match':>11}")
    for name, _, _ in TIERS:
        idxs = tier_idx[name]
        if len(idxs) < 2:
            print(f"  {name:<18}  {'n/a':>9}  {'n/a':>13}  {'n/a':>11}")
            continue
        sample = rng.choice(idxs, size=(min(args.n_pairs, len(idxs)//2), 2),
                            replace=False)
        sims = []
        nontriv = 0
        selfmatch = 0
        for a, b in sample:
            if a == b: continue
            mid = (corpus_embs[a] + corpus_embs[b]) / 2
            # Find nearest, excluding A and B themselves
            target = mid / (np.linalg.norm(mid) + 1e-12)
            c_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-12)
            s = c_norm @ target
            s[a] = -np.inf; s[b] = -np.inf
            c = int(np.argmax(s))
            sims.append(float(s[c]))
            if c != a and c != b:
                nontriv += 1
            # Also compute cos against original midpoint endpoints (sanity)
            if cosine(mid, corpus_embs[a]) > s[c] or cosine(mid, corpus_embs[b]) > s[c]:
                selfmatch += 1
        sims = np.array(sims)
        print(f"  {name:<18}  {sims.mean():>9.3f}  "
              f"{100*nontriv/len(sims):>12.1f}%  {100*selfmatch/len(sims):>10.1f}%")


if __name__ == "__main__":
    main()
