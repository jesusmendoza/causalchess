#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_strategic_distance_controlled.py — Material-controlled eval prediction
from the frozen embedding.

Background: test_strategic_distance.py shows near-zero correlation between
embedding distance and eval difference for trained models, while an untrained
CNN scores +0.32 via piece-count summation. That test is confounded by
material (Stockfish eval is largely material + a bit of strategy).

This script removes the material confound and asks a SHARPER question:
"Given the position embedding alone, can a small head predict position
evaluation?" — i.e., does the embedding carry STRATEGIC information that
survives once material is factored out?

Pipeline:
  1. Load (FEN, eval_cp) pairs from Stockfish.
  2. Extract embeddings for each position.
  3. Compute per-position piece count + material balance (from FEN).
  4. Train TWO models on 80% of the data:
       baseline:       piece-count + material  →  eval_cp
       embedding-only: embedding (256)         →  eval_cp
       combined:       concat(baseline_feats, embedding) → eval_cp
  5. Report R² on held-out 20%.

If 'embedding-only' >> 'baseline', the embedding has extractable strategic
info on top of material. If 'combined' > 'embedding-only' + 'baseline',
the embedding and material are complementary — neither replaces the other.
"""
import argparse
import os
import numpy as np
import torch
import chess
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import compute_corpus_embeddings


PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9}


def material_features(fen: str):
    """6 floats: piece_count, material_total, material_white, material_black,
    material_balance (W-B), phase (3 tiers via total mat)."""
    board = chess.Board(fen)
    mat_w, mat_b = 0, 0
    n_pieces = 0
    for pt, v in PIECE_VALUE.items():
        w = len(board.pieces(pt, chess.WHITE))
        b = len(board.pieces(pt, chess.BLACK))
        mat_w += w * v
        mat_b += b * v
        n_pieces += w + b
    # king is always both sides; include for completeness
    n_pieces += 2
    mat_total = mat_w + mat_b
    mat_bal = mat_w - mat_b
    phase = 0.0 if mat_total >= 70 else (1.0 if mat_total >= 40 else 2.0)
    return np.array([n_pieces, mat_total, mat_w, mat_b, mat_bal, phase],
                    dtype=np.float32)


def load_eval_tsv(path, max_n=0):
    fens, evals = [], []
    with open(path) as f:
        header = f.readline().strip().split('\t')
        i_fen = header.index('fen')
        i_eval = header.index('eval_cp')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            try:
                e = float(parts[i_eval])
            except ValueError:
                continue
            fens.append(parts[i_fen])
            evals.append(e)
            if max_n > 0 and len(fens) >= max_n:
                break
    return fens, np.array(evals)


def train_and_r2(X_tr, y_tr, X_val, y_val, label):
    clf = Ridge(alpha=1.0)
    clf.fit(X_tr, y_tr)
    r2_tr = r2_score(y_tr, clf.predict(X_tr))
    r2_val = r2_score(y_val, clf.predict(X_val))
    mae = np.mean(np.abs(y_val - clf.predict(X_val)))
    print(f"  {label:<35} train R²={r2_tr:+.4f}  val R²={r2_val:+.4f}  val MAE={mae:.1f} cp")
    return r2_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--evals", default="lichess_data/stockfish_evals.tsv")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--clip-eval", type=float, default=2000.0)
    args = ap.parse_args()

    if not os.path.exists(args.evals):
        print(f"ERROR: {args.evals} not found — run compute_stockfish_evals.py first")
        return

    print(f"Loading {args.n} (FEN, eval) from {args.evals}...")
    fens, evals = load_eval_tsv(args.evals, max_n=args.n)
    evals_clip = np.clip(evals, -args.clip_eval, args.clip_eval)
    print(f"  {len(fens)} positions, eval range "
          f"[{evals.min():.0f}, {evals.max():.0f}] clipped to ±{args.clip_eval:.0f}")

    print("\nComputing material features...")
    mat_feats = np.stack([material_features(f) for f in fens])
    print(f"  material features shape: {mat_feats.shape}")

    print(f"\nLoading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get('arch', 'cnn')
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    emb = compute_corpus_embeddings(model, fens)
    print(f"  embeddings shape: {emb.shape}")

    print("\nComputing embeddings (untrained control)...")
    torch.manual_seed(42)
    untrained = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    untrained.eval()
    emb_u = compute_corpus_embeddings(untrained, fens)

    # Train/val split
    idx_tr, idx_val = train_test_split(np.arange(len(fens)), test_size=0.2, random_state=42)
    y_tr, y_val = evals_clip[idx_tr], evals_clip[idx_val]

    print("\n" + "=" * 80)
    print("  Ridge regression: predict Stockfish eval (cp) from frozen features")
    print("=" * 80)

    # Baselines
    train_and_r2(mat_feats[idx_tr], y_tr, mat_feats[idx_val], y_val,
                 "Material (6-d) only")

    # Trained embedding variants
    train_and_r2(emb[idx_tr], y_tr, emb[idx_val], y_val,
                 f"Trained embedding ({arch}, 256-d)")
    train_and_r2(np.concatenate([mat_feats, emb], axis=1)[idx_tr], y_tr,
                 np.concatenate([mat_feats, emb], axis=1)[idx_val], y_val,
                 f"Material + trained embedding (262-d)")

    # Untrained control
    train_and_r2(emb_u[idx_tr], y_tr, emb_u[idx_val], y_val,
                 "Untrained embedding (256-d)")
    train_and_r2(np.concatenate([mat_feats, emb_u], axis=1)[idx_tr], y_tr,
                 np.concatenate([mat_feats, emb_u], axis=1)[idx_val], y_val,
                 "Material + untrained embedding (262-d)")

    print("=" * 80)
    print("\nInterpretation:")
    print("  - Material-only R²: how much of eval is explained by piece counts.")
    print("  - Trained emb R² > material R²: embedding has STRATEGIC info that")
    print("    survives material removal (the question we care about).")
    print("  - Trained emb R² < material R²: material dominates, embedding")
    print("    mostly captures stuff orthogonal to eval.")
    print("  - Combined > both individually: complementary signals.")


if __name__ == "__main__":
    main()
