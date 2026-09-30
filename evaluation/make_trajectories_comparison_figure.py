#!/usr/bin/env python3
"""Render unit-L2 trajectory steps for PMP-mem and PMP-no-mem."""
import sys as _sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_PARENT))

import argparse

import chess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prev_move_models import create_model, fen_to_bitboards
from trajectory_games import load_kasparov_topalov


OTHER_GAMES = {
    "Scholar's Mate": [
        "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
    ],
    "Morphy Opera (33 plies)": [
        "e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "c8g4",
        "d4e5", "g4f3", "d1f3", "d6e5", "f1c4", "g8f6",
        "f3b3", "d8e7", "b1c3", "c7c6", "c1g5", "b7b5",
        "c3b5", "c6b5", "c4b5", "b8d7", "e1c1", "a8d8",
        "d1d7", "d8d7", "h1d1", "e7e6", "b5d7", "f6d7",
        "b3b8", "d7b8", "d1d8",
    ],
}


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["arch"], checkpoint["n_moves"],
                         embed_dim=checkpoint["embed_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model.cpu(), checkpoint.get("val_acc", 0.0)


def game_fens(uci_moves):
    board = chess.Board()
    fens = [board.fen()]
    for ply, uci in enumerate(uci_moves, 1):
        try:
            board.push_uci(uci)
        except Exception as exc:
            raise ValueError(f"illegal move {uci!r} at ply {ply}: {exc}") from exc
        fens.append(board.fen())
    return fens


def embed_all(model, fens):
    x = torch.stack([torch.from_numpy(fen_to_bitboards(fen)) for fen in fens])
    with torch.inference_mode():
        return model.get_embedding(x).cpu().numpy()


def chord_steps(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if not np.all(np.isfinite(emb)):
        raise ValueError("Embeddings must contain only finite values")
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot normalize a near-zero embedding")
    unit = emb / norms
    return np.linalg.norm(np.diff(unit, axis=0), axis=1)


def plot_panel(ax, dists, model_name, game_name, color, y_max, show_ylabel):
    xs = np.arange(1, len(dists) + 1)
    ax.plot(xs, dists, color=color, lw=1.8, marker="o", markersize=2.4)
    ax.axhline(dists.mean(), linestyle="--", color="grey", lw=1.0, alpha=0.8)
    ax.set_title(
        f"{model_name} · {game_name}\nmean={dists.mean():.3f}; SD={dists.std():.3f}",
        fontsize=11.5,
    )
    ax.set_xlabel("Ply", fontsize=15)
    ax.set_ylabel("Unit-L2 chord distance" if show_ylabel else "", fontsize=13)
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--nomem", default="models/prev_move_cnn_v2_dedup.pt")
    ap.add_argument("--output", default="paper/figures/trajectories_mem_vs_nomem.png")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(1)
    kt_uci, kt_san, kt_fens = load_kasparov_topalov()
    games = [("K–T (full, 87 plies)", kt_fens)]
    games.extend((name, game_fens(moves)) for name, moves in OTHER_GAMES.items())

    print(f"Loading PMP-mem from {args.mem}...")
    mem, _ = load_model(args.mem)
    print(f"Loading PMP-no-mem from {args.nomem}...")
    nomem, _ = load_model(args.nomem)

    fig, axes = plt.subplots(len(games), 2, figsize=(7.2, 9.3), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (game_name, fens) in enumerate(games):
        emb_mem = embed_all(mem, fens)
        emb_nomem = embed_all(nomem, fens)
        steps = {
            "mem": chord_steps(emb_mem),
            "no_mem": chord_steps(emb_nomem),
        }
        ymax = max(steps["mem"].max(), steps["no_mem"].max()) * 1.12
        plot_panel(axes[row, 0], steps["mem"], "PMP-mem", game_name,
                   "steelblue", ymax, show_ylabel=True)
        plot_panel(axes[row, 1], steps["no_mem"], "PMP-no-mem", game_name,
                   "seagreen", ymax, show_ylabel=False)

    fig.suptitle(
        "Consecutive trajectory steps after per-position L2 normalization",
        fontsize=14,
    )
    fig.text(
        0.5, -0.008,
        "Distances are unit-sphere chord lengths (0–2); curves describe individual games.",
        ha="center", fontsize=9,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Kasparov–Topalov: {len(kt_uci)} plies, {len(kt_fens)} positions; final {kt_san[-1]}")


if __name__ == "__main__":
    main()
