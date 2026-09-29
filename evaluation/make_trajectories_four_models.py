#!/usr/bin/env python3
"""Render Fig. trajectories: four models on Kasparov–Topalov 1999."""
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)

import argparse
import numpy as np
import torch
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from prev_move_models import create_model, fen_to_bitboards

KASPAROV = [
    "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7",
    "d1d2", "c7c6", "f2f3", "b7b5", "g1e2", "b8d7", "e3h6", "g7h6",
    "d2h6", "c8b7", "a2a3", "e7e5", "e1c1", "d8e7", "c1b1", "a7a6",
    "e2c1", "e8c8", "c1b3", "e5d4", "d1d4", "c6c5", "d4d1", "d7b6",
    "g2g3", "c8b8", "b3a5", "b7a8", "f1h3", "d6d5", "h6f4", "b8a7",
    "h1e1", "d5d4", "c3d5", "b6d5", "e4d5", "e7d6", "d1d4", "c5d4",
    "e1e7", "a7b6", "f4d4", "b6a5", "b2b4", "a5a4", "d4c3", "d6d5",
    "e7a7", "a8b7", "a7b7", "d5c4", "c3f6", "a4a3", "f6a6", "a3b4",
    "c2c3", "b4c3", "a6a1", "c3d2", "a1b2",
]

MODELS = [
    ("PMP-mem", "models/prev_move_cnn_v2.pt", "steelblue"),
    ("PMP-no-mem", "models/prev_move_cnn_v2_dedup.pt", "seagreen"),
    ("AE", "models/ae_cnn.pt", "darkorange"),
    ("SimCLR", "models/simclr_cnn_final.pt", "mediumpurple"),
]


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = create_model(ck["arch"], ck["n_moves"], embed_dim=ck["embed_dim"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def game_fens(moves):
    board = chess.Board()
    fens = [board.fen()]
    for i, u in enumerate(moves):
        try:
            board.push_uci(u)
        except Exception as exc:
            raise ValueError(
                f"illegal move {u!r} at ply {i + 1} (move {i // 2 + 1}"
                f"{'.' if i % 2 == 0 else '...'}) after {i} accepted moves: {exc}"
            ) from exc
        fens.append(board.fen())
    return fens


def embed_all(model, fens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in fens]).to(device)
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="paper/figures/test_trajectories.png")
    args = ap.parse_args()

    fens = game_fens(KASPAROV)
    emb_by = {}
    for name, path, color in MODELS:
        print(f"Loading {name} from {path}...")
        emb_by[name] = (embed_all(load_model(path), fens), color)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: consecutive distances overlay
    ax = axes[0, 0]
    for name, (emb, color) in emb_by.items():
        d = np.linalg.norm(np.diff(emb, axis=0), axis=1)
        ax.plot(np.arange(1, len(d) + 1), d, color=color, lw=2,
                label=f"{name} (μ={d.mean():.1f})")
    ax.set_title("Consecutive L2 distance (should be smooth)", fontsize=13)
    ax.set_xlabel("move index", fontsize=12)
    ax.set_ylabel(r"$\|emb(t{+}1)-emb(t)\|$", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Top-right: distance from start
    ax = axes[0, 1]
    for name, (emb, color) in emb_by.items():
        d0 = np.linalg.norm(emb - emb[0], axis=1)
        ax.plot(np.arange(len(d0)), d0, color=color, lw=2,
                label=f"{name} (final={d0[-1]:.1f})")
    ax.set_title("Distance from opening (final drift)", fontsize=13)
    ax.set_xlabel("move index", fontsize=12)
    ax.set_ylabel(r"$\|emb(t)-emb(0)\|$", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Bottom: PCA trajectories (shared fit on concat)
    concat = np.vstack([e for e, _ in emb_by.values()])
    pca = PCA(n_components=2).fit(concat)
    ax = axes[1, 0]
    for name in ("PMP-mem", "PMP-no-mem"):
        emb, color = emb_by[name]
        p = pca.transform(emb)
        ax.plot(p[:, 0], p[:, 1], "-o", color=color, ms=3, lw=1.5, label=name, alpha=0.85)
        ax.scatter(p[0, 0], p[0, 1], c="green", s=60, zorder=5, marker="s")
        ax.scatter(p[-1, 0], p[-1, 1], c="black", s=60, zorder=5, marker="x")
    ax.set_title("PCA: PMP variants (shared space)", fontsize=13)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for name in ("AE", "SimCLR"):
        emb, color = emb_by[name]
        p = pca.transform(emb)
        ax.plot(p[:, 0], p[:, 1], "-o", color=color, ms=3, lw=1.5, label=name, alpha=0.85)
        ax.scatter(p[0, 0], p[0, 1], c="green", s=60, zorder=5, marker="s")
        ax.scatter(p[-1, 0], p[-1, 1], c="black", s=60, zorder=5, marker="x")
    ax.set_title("PCA: AE vs SimCLR (same shared space)", fontsize=13)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Kasparov–Topalov 1999 — embedding trajectories by pretraining objective",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _os.makedirs(_os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"Saved {args.output}")
    print(f"positions={len(fens)} (start + {len(KASPAROV)} UCI)")
    print("Table-ready stats (mean consec, std consec, final drift):")
    for name, (emb, _) in emb_by.items():
        consec = np.linalg.norm(np.diff(emb, axis=0), axis=1)
        drift = np.linalg.norm(emb - emb[0], axis=1)
        print(f"  {name}: {consec.mean():.2f} / {consec.std():.2f} / {drift[-1]:.2f}")


if __name__ == "__main__":
    main()
