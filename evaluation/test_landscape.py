#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_landscape.py — chess embedding "universe" visualization.

Sample 5000 positions from the corpus, project to 2D (UMAP + t-SNE + PCA),
color by game phase / material. Overlay 3 real game trajectories.

The 2-D projections are descriptive views of one trained embedding space.
"""
import os
import sys
import tempfile
import torch
import numpy as np
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus
from trajectory_games import load_kasparov_topalov

# Numba's packaged UMAP decorators need a writable cache in this environment.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache_causalchess"))

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('val_acc', 0)


def fen_phase(fen):
    """Classify game phase by material count.
    Returns: 0=opening, 1=middlegame, 2=endgame.
    """
    board = chess.Board(fen)
    # Piece values (exclude kings): P=1, N=3, B=3, R=5, Q=9
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for pt, v in pmap.items():
        total += len(board.pieces(pt, chess.WHITE)) * v
        total += len(board.pieces(pt, chess.BLACK)) * v
    # Starting total (both sides): 2*(8*1 + 2*3 + 2*3 + 2*5 + 9) = 2*39 = 78
    # Opening: ~72-78, Middlegame: ~40-72, Endgame: <40
    if total >= 70:
        return 0
    elif total >= 40:
        return 1
    else:
        return 2


def fen_piece_count(fen):
    board = chess.Board(fen)
    return sum(1 for _ in board.piece_map().values())


def embed_fens(model, fens, batch=1024):
    # CPU inference keeps regenerated projections stable across GPU kernels.
    device = torch.device("cpu")
    model = model.to(device)
    all_embs = []
    for i in range(0, len(fens), batch):
        batch_fens = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in batch_fens]).to(device)
        with torch.no_grad():
            emb = model.get_embedding(x).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs)


def game_embeddings(model, moves_uci):
    board = chess.Board()
    fens = [board.fen()]
    for i, m in enumerate(moves_uci):
        try:
            board.push_uci(m)
        except Exception as exc:
            raise ValueError(
                f"illegal move {m!r} at ply {i + 1} (move {i // 2 + 1}"
                f"{'.' if i % 2 == 0 else '...'}) after {i} accepted moves: {exc}"
            ) from exc
        fens.append(board.fen())
    return embed_fens(model, fens), fens


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt")
    args = ap.parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model_path = args.ckpt
    model, val_acc = load_model(model_path)
    print(f"Loaded CNN (val_acc={val_acc:.2f}%)")

    # Sample 5000 positions from corpus
    print("Loading corpus...")
    tsv = "lichess_data/prev_move_2400.tsv"
    corpus_fens = load_corpus(tsv, max_n=50000)
    sample_idx = np.random.choice(len(corpus_fens), size=5000, replace=False)
    sample_fens = [corpus_fens[i] for i in sample_idx]

    print("Computing 5000 embeddings...")
    sample_embs = embed_fens(model, sample_fens)

    print("Computing phases and piece counts...")
    phases = np.array([fen_phase(f) for f in sample_fens])
    piece_counts = np.array([fen_piece_count(f) for f in sample_fens])

    # K–T comes from the canonical full PGN; never maintain a second move list.
    kasparov_uci, _, _ = load_kasparov_topalov()

    # Real games to overlay
    games = {
        "Kasparov–Topalov 1999 (complete, 87 plies)": (kasparov_uci, "black"),
        "Scholar's Mate": ([
            "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
        ], "#2f6fbd"),
        "Morphy's Opera Game 1858": ([
            "e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "c8g4",
            "d4e5", "g4f3", "d1f3", "d6e5", "f1c4", "g8f6",
            "f3b3", "d8e7", "b1c3", "c7c6", "c1g5", "b7b5",
            "c3b5", "c6b5", "c4b5", "b8d7", "e1c1", "a8d8",
            "d1d7", "d8d7", "h1d1", "e7e6", "b5d7", "f6d7",
            "b3b8", "d7b8", "d1d8",
        ], "orange"),
    }

    game_trajs = {}
    for name, (moves, color) in games.items():
        traj_emb, _ = game_embeddings(model, moves)
        game_trajs[name] = (traj_emb, color)

    # Compute projections: PCA, t-SNE, UMAP
    projections = {}

    print("PCA...")
    pca = PCA(n_components=2)
    pca.fit(sample_embs)
    projections["PCA"] = pca

    print("t-SNE...")
    # Fit on sample + all trajectories so they share space
    all_traj = np.vstack([e for e, _ in game_trajs.values()])
    combined = np.vstack([sample_embs, all_traj])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca",
                learning_rate="auto")
    tsne_out = tsne.fit_transform(combined)
    projections["t-SNE"] = ("precomputed", tsne_out, len(sample_embs))

    if HAS_UMAP:
        print("UMAP...")
        um = umap.UMAP(n_components=2, random_state=42)
        um.fit(sample_embs)
        projections["UMAP"] = um

    # Build plot grid: 1 row of 3 (or 2 if no umap), 2 columns (phase, pieces)
    n_methods = len(projections)
    fig, axes = plt.subplots(n_methods, 2, figsize=(13, 4.2 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    colors_phase = ['#4287f5', '#f59e42', '#f54242']  # blue, orange, red
    phase_names = ['opening', 'middlegame', 'endgame']

    for row, (method_name, proj) in enumerate(projections.items()):
        if method_name == "t-SNE":
            _, tsne_pts, n_sample = proj
            sample_2d = tsne_pts[:n_sample]
            # traj_2d is the rest; split per game
            traj_offset = n_sample
            traj_2d_per_game = {}
            for name, (emb, color) in game_trajs.items():
                n = len(emb)
                traj_2d_per_game[name] = (tsne_pts[traj_offset:traj_offset+n], color)
                traj_offset += n
        else:
            sample_2d = proj.transform(sample_embs)
            traj_2d_per_game = {}
            for name, (emb, color) in game_trajs.items():
                traj_2d_per_game[name] = (proj.transform(emb), color)

        # Subplot 1: colored by phase
        ax = axes[row, 0]
        for p in range(3):
            mask = phases == p
            ax.scatter(sample_2d[mask, 0], sample_2d[mask, 1],
                       c=colors_phase[p], s=4, alpha=0.4,
                       label=phase_names[p])
        # overlay trajectories
        for name, (pts_2d, color) in traj_2d_per_game.items():
            ax.plot(pts_2d[:, 0], pts_2d[:, 1], '-o',
                    color=color, linewidth=1.2, markersize=3,
                    alpha=0.9, label=name)
            ax.scatter(pts_2d[0, 0], pts_2d[0, 1],
                       c="green", s=60, marker="s", zorder=5,
                       edgecolors="black")  # start
            ax.scatter(pts_2d[-1, 0], pts_2d[-1, 1],
                       c="black", s=60, marker="x", zorder=5)  # end
        ax.set_title(f"{method_name}: game phase\n"
                     f"(green square = start; black X = end)",
                     fontsize=18)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)

        # Subplot 2: colored by piece count
        ax = axes[row, 1]
        sc = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                        c=piece_counts, cmap="viridis",
                        s=6, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("piece count (2-32)", fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        for name, (pts_2d, color) in traj_2d_per_game.items():
            ax.plot(pts_2d[:, 0], pts_2d[:, 1], '-',
                    color=color, linewidth=1.5, alpha=0.7)
        ax.set_title(f"{method_name}: piece count", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)

    # One shared legend avoids covering the trajectories in each phase panel.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=12, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig("paper/figures/test_landscape.png", dpi=140, bbox_inches="tight")
    print("Saved test_landscape.png")


if __name__ == "__main__":
    main()
