#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_nextmove.py — Test 3: next-move prediction from frozen embedding.

Does the prev-move-trained embedding generalize to predict the NEXT move?

Setup:
  1. Extract (position_BEFORE, next_move) pairs from PGN (small sample)
  2. Compute embeddings with frozen trained / untrained CNN
  3. Train linear classifier: embedding → next_move class
  4. Compare accuracies

If trained >> untrained, the embedding captures strategic knowledge
TRANSFERABLE beyond its specific objective — a core claim for Chess2Vec.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import chess
import chess.pgn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


PGN_PATH = "lichess_data/lichess_2013_01.pgn"
N_GAMES = 3000   # 3000 games × ~40 positions = ~120k pairs
MIN_PLY = 4
MAX_PLY = 200
MIN_ELO = 0   # no filter — process faster, amateur data is fine for transfer test


def extract_nextmove_pairs(pgn_path, n_games, min_ply, min_elo):
    """Returns list of (fen_before, next_move_uci)."""
    pairs = []
    games = 0
    with open(pgn_path) as f:
        while games < n_games:
            game = chess.pgn.read_game(f)
            if game is None: break
            try:
                w = int(game.headers.get("WhiteElo", "0"))
                b = int(game.headers.get("BlackElo", "0"))
                if w < min_elo or b < min_elo: continue
            except (ValueError, TypeError): pass
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < min_ply: continue
            for i, move in enumerate(moves):
                if i >= min_ply - 1 and i < MAX_PLY:
                    fen_before = board.fen()
                    pairs.append((fen_before, move.uci()))
                board.push(move)
            games += 1
            if games % 100 == 0:
                print(f"  {games} games, {len(pairs)} pairs", flush=True)
    return pairs


def load_trained(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def load_untrained(ckpt):
    torch.manual_seed(42)
    m = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    m.eval()
    return m


def embed_batch(model, fens, batch=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(device)
        with torch.no_grad():
            out.append(model.get_embedding(x).cpu().numpy())
    return np.vstack(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--pgn", default=PGN_PATH,
                    help="PGN file for next-move extraction (default: "
                         "lichess 2013_01). Override to point at the "
                         "SEALED validation set.")
    args = ap.parse_args()
    ckpt_path = args.ckpt
    trained, ckpt = load_trained(ckpt_path)
    untrained = load_untrained(ckpt)
    print(f"Loaded CNN val_acc={ckpt.get('val_acc', 0):.2f}%")

    print(f"Extracting (position, next_move) from {N_GAMES} games in {args.pgn}...")
    pairs = extract_nextmove_pairs(args.pgn, N_GAMES, MIN_PLY, MIN_ELO)
    print(f"  Total: {len(pairs)} pairs")

    # Build vocab from top-K most frequent moves (reduce class imbalance)
    from collections import Counter
    move_counts = Counter(m for _, m in pairs)
    top_k = 500  # top 500 most frequent moves
    top_moves = [m for m, _ in move_counts.most_common(top_k)]
    vocab = {m: i for i, m in enumerate(top_moves)}
    print(f"  Using top-{top_k} moves as classes (covers "
          f"{sum(c for _, c in move_counts.most_common(top_k)) / len(pairs) * 100:.1f}% of data)")

    filtered = [(f, vocab[m]) for f, m in pairs if m in vocab]
    print(f"  After top-k filter: {len(filtered)} pairs")

    fens = [f for f, _ in filtered]
    labels = np.array([y for _, y in filtered])

    # Cap to reasonable size
    if len(fens) > 50000:
        np.random.seed(42)
        idx = np.random.choice(len(fens), 50000, replace=False)
        fens = [fens[i] for i in idx]
        labels = labels[idx]
        print(f"  Subsampled to {len(fens)} pairs for speed")

    print("Computing embeddings...")
    emb_t = embed_batch(trained, fens)
    emb_u = embed_batch(untrained, fens)

    # Train-val split
    X_tr_t, X_val_t, y_tr, y_val = train_test_split(
        emb_t, labels, test_size=0.2, random_state=42, stratify=None)
    X_tr_u, X_val_u, _, _ = train_test_split(
        emb_u, labels, test_size=0.2, random_state=42, stratify=None)

    print("Training linear heads...")
    clf_t = LogisticRegression(max_iter=1000, verbose=0)
    clf_t.fit(X_tr_t, y_tr)
    acc_t = accuracy_score(y_val, clf_t.predict(X_val_t))
    print(f"  Trained CNN + linear head:   val_acc = {acc_t*100:.2f}%")

    clf_u = LogisticRegression(max_iter=1000, verbose=0)
    clf_u.fit(X_tr_u, y_tr)
    acc_u = accuracy_score(y_val, clf_u.predict(X_val_u))
    print(f"  Untrained CNN + linear head: val_acc = {acc_u*100:.2f}%")

    majority = max(np.mean(y_val == c) for c in np.unique(y_tr))
    print(f"  Majority baseline:           {majority*100:.2f}%")
    print(f"  Random (1/{top_k}):            {100/top_k:.3f}%")

    # Top-5 accuracy
    probs_t = clf_t.predict_proba(X_val_t)
    top5_t = np.mean([y_val[i] in np.argsort(probs_t[i])[-5:] for i in range(len(y_val))])
    probs_u = clf_u.predict_proba(X_val_u)
    top5_u = np.mean([y_val[i] in np.argsort(probs_u[i])[-5:] for i in range(len(y_val))])
    print(f"\n  Top-5 accuracy:")
    print(f"    Trained:   {top5_t*100:.2f}%")
    print(f"    Untrained: {top5_u*100:.2f}%")

    print(f"\n### Summary: {'✓ TRANSFERABLE' if (acc_t - acc_u) > 0.02 else '✗ no transfer'} ###")
    print(f"  Lift (trained − untrained): {(acc_t - acc_u)*100:+.2f}%")


if __name__ == "__main__":
    main()
