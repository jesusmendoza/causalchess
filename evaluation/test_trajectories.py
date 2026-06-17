#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_trajectories.py — Test 4: game trajectories in embedding space.

Hypothesis: the embedding traces smooth curves through a game
(opening → middlegame → endgame) without discontinuities.

If embedding captures structure, consecutive positions should be close,
and the trajectory should visit distinct "regions" matching game phases.

Produces:
- PCA / t-SNE plot of 10 full games colored by move number
- Distance(pos_t, pos_{t+1}) curve per game (should be small & smooth)
- Distance(pos_t, pos_0) curve (should grow monotonically until endgame)
"""
import sys
import os
import torch
import numpy as np
import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('val_acc', 0)


def game_to_embeddings(model, moves_uci):
    """Given list of UCI moves, return embeddings for position after each move."""
    board = chess.Board()
    fens = [board.fen()]
    for m in moves_uci:
        try:
            board.push_uci(m)
        except Exception:
            break
        fens.append(board.fen())

    # Batch embed (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in fens]).to(device)
    with torch.no_grad():
        emb = model.get_embedding(x).cpu().numpy()
    return emb, fens


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    args = ap.parse_args()
    model_path = args.ckpt
    if not os.path.isfile(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    print(f"Loading {model_path}...")
    model, val_acc = load_model(model_path)
    print(f"  val_acc = {val_acc:.2f}%")

    # A handful of real games (well-known) in UCI format
    games = {
        "Kasparov-Topalov 1999 (Kasparov's Immortal)": [
            "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6",
            "c1e3", "f8g7", "d1d2", "c7c6", "f2f3", "b7b5",
            "g1e2", "b8d7", "e3h6", "g7h6", "d2h6", "c8b7",
            "a2a3", "e7e5", "e1c1", "d8e7", "c1b1", "a7a6",
            "e2c1", "e8c8", "c1b3", "e5d4", "d1d4", "c6c5",
            "d4d1", "d7b6", "g2g3", "c8b8", "c3a5", "b7a8",
            "f1h3", "d6d5", "h6f4", "b8a7", "h1e1", "d5d4",
            "a5d5", "b6d5", "e4d5", "e7d6", "d1d4", "c5d4",
            "e1e7", "a7b6", "f4d4", "b6a5", "b2b4", "a5a4",
            "d4c3", "d6d5", "e7a7", "a8b7", "a7b7", "d5c4",
            "c3f6", "a4a3", "f6a6", "a3b4", "c2c3", "b4c3",
            "a6a1", "c3d2", "a1b2",
        ],
        "Scholar's Mate (4-move mate)": [
            "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6",
            "h5f7",
        ],
        "Steinitz-Rock 1873 (quick tactical)": [
            "e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "c8g4",
            "d4e5", "g4f3", "d1f3", "d6e5", "f1c4", "g8f6",
            "f3b3", "d8e7", "b1c3", "c7c6", "c1g5", "b7b5",
            "c3b5", "c6b5", "c4b5", "b8d7", "e1c1", "a8d8",
            "d1d7", "d8d7", "h1d1", "e7e6", "b5d7", "f6d7",
            "b3b8", "d7b8", "d1d8",
        ],
    }

    fig, axes = plt.subplots(len(games), 3, figsize=(18, 5 * len(games)))
    if len(games) == 1:
        axes = axes.reshape(1, -1)

    # Collect all embeddings for shared PCA
    all_embs = []
    all_phases = []  # move index for coloring
    for game_name, moves in games.items():
        emb, fens = game_to_embeddings(model, moves)
        all_embs.append(emb)
        all_phases.append(np.arange(len(emb)))

    concat = np.vstack(all_embs)
    pca = PCA(n_components=2)
    pca.fit(concat)

    for row, (game_name, moves) in enumerate(games.items()):
        emb, fens = game_to_embeddings(model, moves)
        n = len(emb)

        # Plot 1: PCA trajectory (colored by move)
        ax = axes[row, 0]
        proj = pca.transform(emb)
        scatter = ax.scatter(proj[:, 0], proj[:, 1],
                             c=np.arange(n), cmap="viridis",
                             s=30, alpha=0.8)
        ax.plot(proj[:, 0], proj[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.annotate(f"start", (proj[0, 0], proj[0, 1]),
                    fontsize=8, color="green", fontweight="bold")
        ax.annotate(f"end", (proj[-1, 0], proj[-1, 1]),
                    fontsize=8, color="red", fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="move number")
        ax.set_title(f"{game_name}\nPCA trajectory ({n} positions)",
                     fontsize=9)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.grid(True, alpha=0.3)

        # Plot 2: consecutive distance (how "jumpy" is the trajectory?)
        ax = axes[row, 1]
        consec = np.linalg.norm(np.diff(emb, axis=0), axis=1)
        ax.plot(consec, "b-")
        ax.axhline(consec.mean(), color="r", linestyle="--",
                   label=f"mean={consec.mean():.2f}")
        ax.set_title("Distance(t, t+1) — should be smooth")
        ax.set_xlabel("move index"); ax.set_ylabel("L2 distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: distance from start
        ax = axes[row, 2]
        from_start = np.linalg.norm(emb - emb[0], axis=1)
        ax.plot(from_start, "g-")
        ax.set_title("Distance(t, start) — trajectory from opening")
        ax.set_xlabel("move index"); ax.set_ylabel("L2 distance")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "paper/figures/test_trajectories.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved {out}")

    # Numeric summary
    print("\nSummary stats per game:")
    for game_name, moves in games.items():
        emb, _ = game_to_embeddings(model, moves)
        consec = np.linalg.norm(np.diff(emb, axis=0), axis=1)
        from_start = np.linalg.norm(emb - emb[0], axis=1)
        print(f"  {game_name}")
        print(f"    {len(emb)} positions")
        print(f"    consec dist: mean={consec.mean():.2f}, "
              f"max={consec.max():.2f}, std={consec.std():.2f}")
        print(f"    from-start dist: final={from_start[-1]:.2f}, "
              f"max={from_start.max():.2f}")


if __name__ == "__main__":
    main()
