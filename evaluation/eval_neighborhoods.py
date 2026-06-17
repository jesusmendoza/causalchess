#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
eval_neighborhoods.py — Find nearest neighbors in embedding space.

Given anchor positions (1.e4, rook endgame, mate in 3, etc.), find the 10
nearest positions in the corpus. Run on trained CNN and untrained CNN.

Tests if the embedding captures semantic similarity — positions near
the anchor should be structurally/strategically similar.
"""
import sys
import os
import torch
import numpy as np
from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus, compute_corpus_embeddings


def get_emb(model, fen):
    device = next(model.parameters()).device
    x = torch.from_numpy(fen_to_bitboards(fen)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()[0]


def nearest(target_emb, corpus_embs, corpus_fens, top_k=10):
    dists = np.linalg.norm(corpus_embs - target_emb, axis=1)
    idxs = np.argsort(dists)[:top_k]
    return [(corpus_fens[i], dists[i]) for i in idxs]


ANCHORS = [
    # ─ Openings ─────────────────────────────────────────
    ("1.e4 (King's pawn opening)",
     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ("1.d4 (Queen's pawn opening)",
     "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"),
    ("1.Nf3 (Reti opening)",
     "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
    ("1.c4 (English opening)",
     "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1"),

    # ─ Middlegame structures ────────────────────────────
    ("Caro-Kann main line (Qxc2 typical structure)",
     "r1bqkb1r/pp1npppp/2p2n2/8/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 2 6"),
    ("Sicilian Najdorf",
     "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("Italian Giuoco Piano (classical middlegame)",
     "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),

    # ─ Endgames ─────────────────────────────────────────
    ("King and Pawn endgame (K+P vs K)",
     "8/5k2/8/8/8/8/4PK2/8 w - - 0 1"),
    ("Rook endgame (KR vs K)",
     "8/5k2/8/8/8/8/4RK2/8 w - - 0 1"),
    ("Rook + Pawn endgame (KRP vs K)",
     "8/5k2/8/8/3P4/8/4RK2/8 w - - 0 1"),
    ("Queen endgame (KQ vs K)",
     "8/5k2/8/8/8/8/4QK2/8 w - - 0 1"),
    ("Bishop pair endgame",
     "8/5k2/8/8/8/8/2B1BK2/8 w - - 0 1"),

    # ─ Tactical positions ───────────────────────────────
    ("Typical mate-in-3 setup (rook on 7th)",
     "6k1/R7/6K1/8/8/8/8/8 w - - 0 1"),
    ("Back-rank mate threat",
     "6k1/5ppp/8/8/8/8/r7/6K1 b - - 0 1"),

    # ─ Asymmetric material ──────────────────────────────
    ("Queen vs Rook",
     "8/5k2/8/8/8/8/4QK2/5r2 w - - 0 1"),
    ("Down a piece (tactical compensation)",
     "r1bqkb1r/pppp1ppp/2n5/4n3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
]


def run_neighborhoods(model, arch_name, corpus_fens, corpus_embs):
    print(f"\n{'='*70}")
    print(f"  NEAREST NEIGHBORS: {arch_name}")
    print(f"{'='*70}")

    for desc, fen in ANCHORS:
        emb = get_emb(model, fen)
        neighbors = nearest(emb, corpus_embs, corpus_fens, top_k=5)
        print(f"\n  ── {desc} ──")
        print(f"    ANCHOR: {fen}")
        for n_fen, dist in neighbors:
            if n_fen == fen:
                marker = "   *"  # self-match
            else:
                marker = "    "
            print(f"    {dist:>5.1f}{marker} {n_fen}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    tsv = "lichess_data/prev_move_2400.tsv"
    print("Loading corpus (50k positions)...", flush=True)
    corpus_fens = load_corpus(tsv, max_n=50000)
    print(f"  Loaded {len(corpus_fens)} positions", flush=True)

    # ── Trained CNN ───────────────────────────────────
    path = args.ckpt
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get('arch', 'cnn')
    n_moves = ckpt['n_moves']
    embed_dim = ckpt['embed_dim']

    print("\n=== PART 1: TRAINED CNN ===", flush=True)
    trained = create_model(arch, n_moves, embed_dim=embed_dim)
    trained.load_state_dict(ckpt['model'])
    trained.eval()
    print("Computing corpus embeddings (trained)...", flush=True)
    corpus_embs_trained = compute_corpus_embeddings(trained, corpus_fens)
    run_neighborhoods(trained, f"CNN trained (val={ckpt['val_acc']:.1f}%)",
                      corpus_fens, corpus_embs_trained)

    # ── Untrained CNN ─────────────────────────────────
    print("\n\n=== PART 2: UNTRAINED CNN (control) ===", flush=True)
    untrained = create_model(arch, n_moves, embed_dim=embed_dim)
    untrained.eval()
    print("Computing corpus embeddings (untrained)...", flush=True)
    corpus_embs_untrained = compute_corpus_embeddings(untrained, corpus_fens)
    run_neighborhoods(untrained, "CNN UNTRAINED (random)",
                      corpus_fens, corpus_embs_untrained)


if __name__ == "__main__":
    main()
