#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_style_endgame.py — Hardest style detection: endgame-only, material-balanced.

Ruling OUT the "cheating with opening" hypothesis.

Filters:
  - MIN_PLY >= 50 (well past opening)
  - Piece count <= 20 (simpler position, less material signal)
  - |material balance| <= 2 pawns (no blowout games)

If the embedding STILL identifies player at >random in these positions,
the style signal is REAL (not opening memorization).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import chess
import chess.pgn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


PGN_PATH = "data_style/gm_games.pgn"
PLAYERS = ["penguingim1", "msb2", "NihalSarin", "DrNykterstein"]
MIN_PLY = 50        # well past opening
MAX_PIECES = 20     # endgame-ish
MAX_MAT_DIFF = 2    # material roughly balanced
MAX_POS_PER_GAME = 30


def material_diff(board):
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    d = 0
    for pt, v in pmap.items():
        d += len(board.pieces(pt, chess.WHITE)) * v
        d -= len(board.pieces(pt, chess.BLACK)) * v
    return abs(d)


def piece_count(board):
    return len(board.piece_map())


def extract_data(pgn_path, players):
    ppos = {p: [] for p in players}
    games_seen = 0
    games_with_player = 0
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            games_seen += 1
            w = game.headers.get("White", "?")
            b = game.headers.get("Black", "?")
            tw = w in players; tb = b in players
            if not (tw or tb): continue
            games_with_player += 1

            board = game.board()
            moves = list(game.mainline_moves())
            n_pos = 0
            for i, move in enumerate(moves):
                mover_white = (i % 2 == 0)
                if i >= MIN_PLY:
                    # Filters
                    if piece_count(board) <= MAX_PIECES and material_diff(board) <= MAX_MAT_DIFF:
                        if mover_white and tw:
                            ppos[w].append(board.fen()); n_pos += 1
                        elif (not mover_white) and tb:
                            ppos[b].append(board.fen()); n_pos += 1
                        if n_pos >= MAX_POS_PER_GAME: break
                board.push(move)
    print(f"  Games scanned: {games_seen}, with target player: {games_with_player}")
    return ppos


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
    args = ap.parse_args()

    print(f"ENDGAME-ONLY STYLE TEST")
    print(f"Filters: MIN_PLY>={MIN_PLY}, pieces<={MAX_PIECES}, "
          f"|mat_diff|<={MAX_MAT_DIFF}")
    print(f"\nExtracting positions for {PLAYERS}...")
    data = extract_data(PGN_PATH, PLAYERS)
    for p, positions in data.items():
        print(f"  {p}: {len(positions)} positions")

    # Balance
    min_n = min(len(v) for v in data.values())
    if min_n < 100:
        print(f"\nWARNING: only {min_n} positions for smallest class. Results may be noisy.")
    print(f"\nBalancing to {min_n} per player")

    fens = []
    labels = []
    for i, p in enumerate(PLAYERS):
        sel = data[p][:min_n]
        fens.extend(sel)
        labels.extend([i] * len(sel))
    labels = np.array(labels)
    n = len(fens)
    print(f"Total: {n} positions, {len(PLAYERS)} classes")
    print(f"Random baseline: {100/len(PLAYERS):.1f}%")

    if n < 200:
        print("\nToo few samples — aborting.")
        return

    ckpt_path = args.ckpt
    trained, ckpt = load_trained(ckpt_path)
    untrained = load_untrained(ckpt)
    print(f"Loaded CNN val_acc={ckpt.get('val_acc',0):.2f}%\n")

    print("Computing embeddings...")
    emb_t = embed_batch(trained, fens)
    emb_u = embed_batch(untrained, fens)

    X_tr_t, X_val_t, y_tr, y_val = train_test_split(
        emb_t, labels, test_size=0.2, random_state=42, stratify=labels)
    X_tr_u, X_val_u, _, _ = train_test_split(
        emb_u, labels, test_size=0.2, random_state=42, stratify=labels)

    clf_t = LogisticRegression(max_iter=2000)
    clf_t.fit(X_tr_t, y_tr)
    acc_t = accuracy_score(y_val, clf_t.predict(X_val_t))

    clf_u = LogisticRegression(max_iter=2000)
    clf_u.fit(X_tr_u, y_tr)
    acc_u = accuracy_score(y_val, clf_u.predict(X_val_u))

    print(f"\n  Trained:   {acc_t*100:.2f}%")
    print(f"  Untrained: {acc_u*100:.2f}%")
    print(f"  Random:    {100/len(PLAYERS):.2f}%")
    print(f"  LIFT:      {(acc_t-acc_u)*100:+.2f}%\n")

    cm = confusion_matrix(y_val, clf_t.predict(X_val_t), normalize='true')
    print("Confusion matrix (trained):")
    print(f"            {' '.join(f'{p[:8]:>8}' for p in PLAYERS)}")
    for i, p in enumerate(PLAYERS):
        row = ' '.join(f'{cm[i,j]*100:>7.1f}%' for j in range(len(PLAYERS)))
        print(f"  {p[:10]:<10} {row}")

    print(f"\n{'✓ STYLE IS REAL (not opening)' if (acc_t - acc_u) > 0.03 else '✗ style collapses in endgame'}")

    # Comparison with all-phase (from prev test)
    print("\nReference: all-phase trained = 47.4% (paper baseline)")
    print(f"           endgame-only trained = {acc_t*100:.1f}%")
    print(f"           degradation = {47.4 - acc_t*100:+.1f}pp")


if __name__ == "__main__":
    main()
