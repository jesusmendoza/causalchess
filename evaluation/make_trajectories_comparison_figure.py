#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
make_trajectories_comparison_figure.py — render a side-by-side figure
comparing PMP-mem and PMP-no-mem embedding trajectories
across a small set of famous games.

Produces `paper/figures/trajectories_mem_vs_nomem.png`:
one row per game; columns = mem (left), no-mem (right). Each panel shows
the consecutive-distance curve with mean and std annotated. The figure
supports the §3.3/§4.5 claim that no-mem has a more uniform phase-distance
profile (no opening/endgame funnel).
"""
import argparse
import os
import chess
import numpy as np
import torch
import matplotlib.pyplot as plt
from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


# Same games used in test_trajectories.py
GAMES = {
    "Kasparov-Topalov 1999": [
        "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7",
        "d1d2", "c7c6", "f2f3", "b7b5", "g1e2", "b8d7", "e3h6", "g7h6",
        "d2h6", "c8b7", "a2a3", "e7e5", "e1c1", "d8e7", "c1b1", "a7a6",
        "e2c1", "e8c8", "c1b3", "e5d4", "d1d4", "c6c5", "d4d1", "d7b6",
        "g2g3", "c8b8", "c3a5", "b7a8",
    ],
    "Scholar's Mate": [
        "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
    ],
    "Steinitz-Rock 1873": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
        "d2d3", "d7d6", "c2c3", "c8g4", "b1d2", "f8e7", "h2h3", "g4h5",
        "f3h4", "h5d1", "e1d1", "e8g8", "h4f5", "e7f6", "c1g5", "f6g5",
        "d2f3", "g5f6", "f3h2", "f6h4", "h2g4", "f6h4", "g4f6",
    ],
}


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "cnn")
    m = create_model(arch, ck["n_moves"], embed_dim=ck["embed_dim"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("val_acc", 0.0)


def game_fens(uci_moves):
    board = chess.Board()
    fens = [board.fen()]
    for u in uci_moves:
        try:
            board.push_uci(u)
        except Exception:
            break
        fens.append(board.fen())
    return fens


def embed_all(model, fens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in fens]).to(device)
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()


def consec_dists(emb):
    return np.linalg.norm(np.diff(emb, axis=0), axis=1)


def plot_panel(ax, dists, title, color):
    xs = np.arange(1, len(dists) + 1)
    ax.plot(xs, dists, color=color, lw=1.5, marker="o", markersize=3)
    ax.axhline(dists.mean(), linestyle="--", color="grey", lw=0.8, alpha=0.8)
    ax.set_title(f"{title}\nmean={dists.mean():.1f}  std={dists.std():.1f}",
                 fontsize=10)
    ax.set_xlabel("move index", fontsize=9)
    ax.set_ylabel("||emb(t+1) - emb(t)||", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--nomem", default="models/prev_move_cnn_v2_dedup.pt")
    ap.add_argument("--output", default="paper/figures/trajectories_mem_vs_nomem.png")
    args = ap.parse_args()

    print(f"Loading mem from {args.mem}...")
    mem, mem_acc = load_model(args.mem)
    print(f"  val_acc={mem_acc:.2f}%")
    print(f"Loading nomem from {args.nomem}...")
    nomem, nomem_acc = load_model(args.nomem)
    print(f"  val_acc={nomem_acc:.2f}%")

    fig, axes = plt.subplots(len(GAMES), 2, figsize=(12, 2.5 * len(GAMES)))
    axes = np.atleast_2d(axes)

    # For consistent y-axis across both columns of each row
    for i, (name, moves) in enumerate(GAMES.items()):
        fens = game_fens(moves)
        if len(fens) < 3:
            continue

        emb_mem = embed_all(mem, fens)
        emb_nomem = embed_all(nomem, fens)

        d_mem = consec_dists(emb_mem)
        d_nomem = consec_dists(emb_nomem)

        ymax = max(d_mem.max(), d_nomem.max()) * 1.1

        plot_panel(axes[i, 0], d_mem,
                   f"{name} — mem (val {mem_acc:.1f}%)", "steelblue")
        plot_panel(axes[i, 1], d_nomem,
                   f"{name} — no-mem (val {nomem_acc:.1f}%)", "seagreen")
        axes[i, 0].set_ylim(0, ymax)
        axes[i, 1].set_ylim(0, ymax)

    plt.suptitle("Consecutive-position embedding distances\nmem vs no-mem comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
