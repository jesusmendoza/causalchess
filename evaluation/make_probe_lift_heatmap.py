#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
make_probe_lift_heatmap.py — render a heatmap visualizing probe lifts
(trained − untrained, in pp) across models × concepts. Supports §4.1/§4.7.

Reads per-model metrics from results_*/ directories, computes lift over
the untrained control, and emits `paper/figures/probe_lift_heatmap.png`.
"""
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


MODELS = [
    ("AE",              "results_ae"),
    ("SimCLR-flip",     "results_simclr_final"),
    ("Mem",             "results_mem_at40"),
    ("No-mem (e1)",     "results_nomem_e1"),
]

ROOT = "evaluation"


def slurp(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def extract_probes(text):
    out = {}
    cls = [
        ("phase",      r"phase \(0/1/2\)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"),
        ("castled_W",  r"white castled\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"),
        ("castled_B",  r"black castled\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"),
        ("turn",       r"turn \(white/black\)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"),
        ("in_check",   r"in check\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"),
    ]
    for name, pat in cls:
        m = re.search(pat, text)
        if m:
            trained = float(m.group(1))
            untrained = float(m.group(2))
            out[name] = trained - untrained
        else:
            out[name] = np.nan
    reg = [
        ("material_bal", r"material balance \(W-B\)\s+R²=([+\-\d.]+)\s+R²=([+\-\d.]+)"),
        ("piece_count",  r"piece count \(2-32\)\s+R²=([+\-\d.]+)\s+R²=([+\-\d.]+)"),
        ("isolated_W",   r"white isolated pawns\s+R²=([+\-\d.]+)\s+R²=([+\-\d.]+)"),
        ("open_files",   r"open files \(0-8\)\s+R²=([+\-\d.]+)\s+R²=([+\-\d.]+)"),
    ]
    for name, pat in reg:
        m = re.search(pat, text)
        if m:
            out[name] = (float(m.group(1)) - float(m.group(2))) * 100  # scale R² lift ×100 to look like pp
        else:
            out[name] = np.nan
    return out


CONCEPTS = ["phase", "castled_W", "castled_B", "turn", "in_check",
            "material_bal", "piece_count", "isolated_W", "open_files"]
CONCEPT_LABELS = {
    "phase": "Phase",
    "castled_W": "Castled W",
    "castled_B": "Castled B",
    "turn": "Turn",
    "in_check": "In-check",
    "material_bal": "Material bal (R²×100)",
    "piece_count": "Piece count (R²×100)",
    "isolated_W": "Isolated W (R²×100)",
    "open_files": "Open files (R²×100)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="paper/figures/probe_lift_heatmap.png")
    args = ap.parse_args()

    M = np.zeros((len(CONCEPTS), len(MODELS)))
    for j, (lbl, rdir) in enumerate(MODELS):
        data = extract_probes(slurp(os.path.join(ROOT, rdir, "probes.log")))
        for i, c in enumerate(CONCEPTS):
            M[i, j] = data.get(c, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    # diverging scale centered at 0
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Text annotations
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        color=color, fontsize=9)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[0] for m in MODELS])
    ax.set_yticks(range(len(CONCEPTS)))
    ax.set_yticklabels([CONCEPT_LABELS[c] for c in CONCEPTS])
    ax.set_title("Probe lift over untrained control (pp for cls, R²×100 for reg)\n"
                 "Red = trained model encodes concept; blue = information discarded",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="lift over untrained")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
