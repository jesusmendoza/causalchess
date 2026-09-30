#!/usr/bin/env python3
"""Render scale-controlled trajectory metrics/figure for the full K–T game."""
import sys as _sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_PARENT))

import json
import platform

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from prev_move_models import create_model, fen_to_bitboards
from trajectory_games import (
    EXPECTED_FINAL_FEN,
    EXPECTED_PLIES,
    EXPECTED_RESULT,
    KASPAROV_TOPALOV_PGN,
    load_kasparov_topalov,
)
from trajectory_metrics import sha256_file, summarize_trajectory


MODELS = [
    ("PMP-mem", "models/prev_move_cnn_v2.pt", "steelblue"),
    ("PMP-no-mem", "models/prev_move_cnn_v2_dedup.pt", "seagreen"),
    ("AE", "models/ae_cnn.pt", "darkorange"),
    ("SimCLR", "models/simclr_cnn_final.pt", "mediumpurple"),
]
SEED = 42


def load_model(path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(
        checkpoint["arch"], checkpoint["n_moves"],
        embed_dim=checkpoint["embed_dim"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model.cpu()


def embed_all(model: torch.nn.Module, fens: list[str]) -> np.ndarray:
    inputs = torch.stack([torch.from_numpy(fen_to_bitboards(fen)) for fen in fens])
    with torch.inference_mode():
        return model.get_embedding(inputs).cpu().numpy().astype(np.float64)


def write_metrics_json(path: Path, san: list[str], fens: list[str], metrics: dict) -> None:
    payload = {
        "game": {
            "white": "Garry Kasparov",
            "black": "Veselin Topalov",
            "event": "Hoogovens Group A",
            "date": "1999-01-20",
            "plies": len(san),
            "positions_including_initial": len(fens),
            "last_san": san[-1],
            "result": EXPECTED_RESULT,
            "final_fen": fens[-1],
            "expected_final_fen": EXPECTED_FINAL_FEN,
            "pgn_path": str(KASPAROV_TOPALOV_PGN.relative_to(_PARENT)),
            "pgn_sha256": sha256_file(KASPAROV_TOPALOV_PGN),
        },
        "protocol": {
            "primary_distance": "per-position unit-L2 chord distance",
            "normalization": "each embedding row divided by its own L2 norm",
            "raw_distance_units": "native units of each model's embedding",
            "pca": "independent fit for each PMP model; coordinates are not cross-model comparable",
            "seed": SEED,
            "landscape_seed": SEED,
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "checkpoint_sha256": {
                name: sha256_file(path) for name, path, _ in MODELS
            },
        },
        "metrics": metrics,
        "interpretation": (
            "Descriptive trajectories from one game and one checkpoint per objective; "
            "no population-level or statistical comparison is implied."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_results_markdown(path: Path, san: list[str], fens: list[str], metrics: dict) -> None:
    model_names = [name for name, _, _ in MODELS]
    lines = [
        "# Trajectory stats — Kasparov–Topalov 1999, complete PGN",
        "",
        f"**Game:** 87 plies; {len(fens)} positions including the initial board; "
        f"last move {len(san) // 2 + 1}.{san[-1]}; result {EXPECTED_RESULT}.",
        f"**Final FEN:** `{fens[-1]}`",
        f"**PGN SHA-256:** `{sha256_file(KASPAROV_TOPALOV_PGN)}`",
        "**Checkpoints:** SHA-256 hashes and protocol metadata are recorded in `results_traj_fixed.json`.",
        "",
        "## Primary scale-controlled metrics (Table IX candidate)",
        "",
        "Each position embedding is L2-normalized separately. Distances are chord lengths on the unit sphere (range 0–2).",
        "",
        "| Metric | " + " | ".join(model_names) + " |",
        "|---|" + "---:|" * len(model_names),
    ]
    keys = [
        ("Consecutive distance, mean", "consecutive_mean"),
        ("Consecutive distance, population SD", "consecutive_std_population"),
        ("Final drift from opening", "final_drift"),
    ]
    for label, key in keys:
        values = [metrics[name]["per_position_unit_l2_chord"][key] for name in model_names]
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "## Raw embedding-unit diagnostics",
        "",
        "Raw Euclidean distances retain the original metric, but scales are native to each model and are not directly comparable across objectives.",
        "",
        "| Metric | " + " | ".join(model_names) + " |",
        "|---|" + "---:|" * len(model_names),
    ]
    for label, key in keys:
        values = [metrics[name]["raw_embedding_units"][key] for name in model_names]
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "The population SD is across consecutive plies within this single game, not across games or model training seeds. These are descriptive measurements of one trajectory and one checkpoint per objective; they do not establish a general smoothness ranking.",
        "",
        "**Reproduction:** run `python evaluation/make_trajectories_four_models.py`; it validates the canonical PGN and writes the figure, this file, and the JSON metadata/hashes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    kasparov_uci, san, fens = load_kasparov_topalov()
    if len(kasparov_uci) != EXPECTED_PLIES or len(fens) != EXPECTED_PLIES + 1:
        raise AssertionError("Canonical game validation returned an unexpected length")

    output = _PARENT / "paper/figures/test_trajectories.png"
    json_path = _PARENT / "paper/revision_tog_2026/results_traj_fixed.json"
    md_path = _PARENT / "paper/revision_tog_2026/results_traj_fixed.md"

    embeddings: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    colors: dict[str, str] = {}
    for name, ckpt_path, color in MODELS:
        print(f"Loading {name}: {ckpt_path}")
        embeddings[name] = embed_all(load_model(ckpt_path), fens)
        metrics[name] = summarize_trajectory(embeddings[name])
        colors[name] = color

    # Figure panels: normalized distances are primary; PCA panels each use a
    # separate fit to avoid pretending the learned spaces share coordinates.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)
    plies = np.arange(1, len(fens))
    for name, _, _ in MODELS:
        emb = embeddings[name]
        unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        step = np.linalg.norm(np.diff(unit, axis=0), axis=1)
        drift = np.linalg.norm(unit - unit[0], axis=1)
        axes[0, 0].plot(
            plies, step, color=colors[name], lw=1.6,
            label=f"{name} (mean {step.mean():.3f})",
        )
        axes[0, 1].plot(
            np.arange(len(fens)), drift, color=colors[name], lw=1.6,
            label=f"{name} (final {drift[-1]:.3f})",
        )

    axes[0, 0].set_title("Per-position unit-L2 chord steps")
    axes[0, 0].set_xlabel("Ply")
    axes[0, 0].set_ylabel("Chord distance (0–2)")
    axes[0, 1].set_title("Unit-L2 drift from opening")
    axes[0, 1].set_xlabel("Position after ply")
    axes[0, 1].set_ylabel("Chord distance (0–2)")
    for ax in axes[0]:
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.25)

    for ax, name in zip(axes[1], ("PMP-mem", "PMP-no-mem")):
        projected = PCA(n_components=2, svd_solver="full").fit_transform(embeddings[name])
        ax.plot(projected[:, 0], projected[:, 1], "-o", color=colors[name], ms=2.1, lw=1.0)
        ax.scatter(projected[0, 0], projected[0, 1], c="green", s=32, marker="s", zorder=5)
        ax.scatter(projected[-1, 0], projected[-1, 1], c="black", s=38, marker="x", zorder=5)
        ax.set_title(f"{name}: independent PCA")
        ax.set_xlabel("PC 1 (own fit)")
        ax.set_ylabel("PC 2 (own fit)")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Kasparov–Topalov 1999: complete game (87 plies, 88 positions)",
        fontsize=12,
    )
    fig.text(
        0.5, -0.01,
        "Distances use unit-L2 embeddings. PCA fits are independent; coordinates are not comparable across panels.",
        ha="center", fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)

    write_metrics_json(json_path, san, fens, metrics)
    write_results_markdown(md_path, san, fens, metrics)
    print(f"Saved {output}")
    print(f"Saved {md_path}")
    print(f"Saved {json_path}")
    print(f"positions={len(fens)} (start + {len(kasparov_uci)} UCI), final={san[-1]} {EXPECTED_RESULT}")
    print("Scale-controlled metrics: mean step / population SD / final drift")
    for name, _, _ in MODELS:
        norm = metrics[name]["per_position_unit_l2_chord"]
        raw = metrics[name]["raw_embedding_units"]
        print(
            f"  {name}: chord {norm['consecutive_mean']:.4f} / "
            f"{norm['consecutive_std_population']:.4f} / {norm['final_drift']:.4f}; "
            f"raw {raw['consecutive_mean']:.2f} / "
            f"{raw['consecutive_std_population']:.2f} / {raw['final_drift']:.2f}"
        )


if __name__ == "__main__":
    main()
