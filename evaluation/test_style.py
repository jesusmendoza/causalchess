#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_style.py — player-identification position-split diagnostic.

Given a position from a GM game, can a linear probe on the CNN embedding
identify WHICH GM played that position?

This diagnostic uses a position-level split and cannot establish player
style generalization to unseen games. See test_style_game_split.py.

Data: data_style/gm_games.pgn (2700+ Elo games)
Top players: penguingim1 (Andrew Tang), msb2, NihalSarin, DrNykterstein (Carlsen)

Task: 4-way classification given position. Use positions AFTER the player
moved (so model sees the player's decisions).
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
MIN_PLY = 6       # skip opening (too generic)
MAX_POS_PER_GAME = 30


def extract_style_data(pgn_path, players):
    """For each target player, extract positions AFTER they moved."""
    player_positions = {p: [] for p in players}
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            w = game.headers.get("White", "?")
            b = game.headers.get("Black", "?")
            target_white = w in players
            target_black = b in players
            if not (target_white or target_black):
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            n_pos = 0
            for i, move in enumerate(moves):
                mover_is_white = (i % 2 == 0)
                # Position BEFORE move; mover is the one to move
                if i >= MIN_PLY:
                    if mover_is_white and target_white:
                        player_positions[w].append(board.fen())
                        n_pos += 1
                    elif (not mover_is_white) and target_black:
                        player_positions[b].append(board.fen())
                        n_pos += 1
                    if n_pos >= MAX_POS_PER_GAME:
                        break
                board.push(move)
    return player_positions


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

    print(f"Extracting positions for {PLAYERS}...")
    data = extract_style_data(PGN_PATH, PLAYERS)
    for p, positions in data.items():
        print(f"  {p}: {len(positions)} positions")

    # Balance: take min N per player to avoid imbalance
    min_n = min(len(v) for v in data.values())
    print(f"Balancing to {min_n} per player")
    fens = []
    labels = []
    for i, p in enumerate(PLAYERS):
        sel = data[p][:min_n]
        fens.extend(sel)
        labels.extend([i] * len(sel))
    labels = np.array(labels)
    print(f"Total: {len(fens)} positions, {len(PLAYERS)} classes")
    print(f"Random baseline: {100/len(PLAYERS):.1f}%")

    ckpt_path = args.ckpt
    trained, ckpt = load_trained(ckpt_path)
    untrained = load_untrained(ckpt)
    print(f"Loaded CNN val_acc={ckpt.get('val_acc',0):.2f}%")

    print("Computing embeddings...")
    emb_t = embed_batch(trained, fens)
    emb_u = embed_batch(untrained, fens)

    X_tr_t, X_val_t, y_tr, y_val = train_test_split(
        emb_t, labels, test_size=0.2, random_state=42, stratify=labels)
    X_tr_u, X_val_u, _, _ = train_test_split(
        emb_u, labels, test_size=0.2, random_state=42, stratify=labels)

    print("Training linear probes...")
    clf_t = LogisticRegression(max_iter=1000)
    clf_t.fit(X_tr_t, y_tr)
    acc_t = accuracy_score(y_val, clf_t.predict(X_val_t))

    clf_u = LogisticRegression(max_iter=1000)
    clf_u.fit(X_tr_u, y_tr)
    acc_u = accuracy_score(y_val, clf_u.predict(X_val_u))

    print(f"\n  Trained:   {acc_t*100:.2f}%")
    print(f"  Untrained: {acc_u*100:.2f}%")
    print(f"  Random:    {100/len(PLAYERS):.2f}%")
    print(f"  LIFT (trained - untrained): {(acc_t-acc_u)*100:+.2f}%\n")

    # Confusion matrix for trained
    cm = confusion_matrix(y_val, clf_t.predict(X_val_t), normalize='true')
    print("Confusion matrix (trained, normalized by true class):")
    print(f"            {' '.join(f'{p[:8]:>8}' for p in PLAYERS)}")
    for i, p in enumerate(PLAYERS):
        row = ' '.join(f'{cm[i,j]*100:>7.1f}%' for j in range(len(PLAYERS)))
        print(f"  {p[:10]:<10} {row}")

    print("\nPosition-split diagnostic only; use test_style_game_split.py for unseen games.")


if __name__ == "__main__":
    main()
