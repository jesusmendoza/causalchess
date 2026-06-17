#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_ensemble_ablation.py — leave-one-out ablation on the 3-model ensemble.

Given the 768-d concatenation (AE + SimCLR + Mem), we measure probe/style/
next-move performance with each component REMOVED (512-d embedding). The
drop quantifies how much that model contributed to the ensemble's score on
each test.

Output: for each concept, a triple (score_remove_AE, score_remove_SimCLR,
score_remove_Mem) plus the full-ensemble score. Interpretation:

- Concept where "remove AE" drops most → AE owned it.
- Concept where all three ablations ~equal → shared contribution.
- Concept where "remove Mem" drops most → Mem's unique niche (e.g., turn, next-move).

Usage:
    python3 evaluation/test_ensemble_ablation.py \\
        --ckpts models/ae_cnn.pt models/simclr_cnn_final.pt models/prev_move_cnn_v2.pt \\
        --labels AE SimCLR Mem --n 10000
"""
import argparse
import os
import numpy as np
import torch
import chess
import chess.pgn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus
from test_probes_strategic import (
    label_phase, label_castled_white, label_castled_black,
    label_turn, label_in_check, label_material_balance,
    label_piece_count, label_white_isolated_pawns, label_open_files,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "cnn")
    m = create_model(arch, ck["n_moves"], embed_dim=ck["embed_dim"])
    m.load_state_dict(ck["model"])
    m.eval().to(DEVICE)
    return m


def embed(model, fens, batch=1024):
    out = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(DEVICE)
        with torch.no_grad():
            out.append(model.get_embedding(x).cpu().numpy())
    return np.vstack(out)


PROBES = [
    ("phase", label_phase, "cls"),
    ("castled_W", label_castled_white, "cls"),
    ("castled_B", label_castled_black, "cls"),
    ("turn", label_turn, "cls"),
    ("in_check", label_in_check, "cls"),
    ("material_bal", label_material_balance, "reg"),
    ("piece_count", label_piece_count, "reg"),
    ("isolated_W", label_white_isolated_pawns, "reg"),
    ("open_files", label_open_files, "reg"),
]


def fit_probe(X_tr, y_tr, X_val, y_val, kind):
    if kind == "cls":
        if len(np.unique(y_tr)) < 2:
            return float("nan")
        clf = LogisticRegression(max_iter=800)
        clf.fit(X_tr, y_tr)
        return accuracy_score(y_val, clf.predict(X_val)) * 100
    else:
        clf = Ridge(alpha=1.0)
        clf.fit(X_tr, y_tr)
        return r2_score(y_val, clf.predict(X_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="list of 3 checkpoint paths")
    ap.add_argument("--labels", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    if len(args.ckpts) != 3:
        raise ValueError("This ablation expects exactly 3 checkpoints")

    print("Loading 3 models...")
    models = {lbl: load_model(p) for lbl, p in zip(args.labels, args.ckpts)}

    print(f"Loading {args.n} corpus positions...")
    all_fens = load_corpus(args.corpus, max_n=50000)
    rng = np.random.default_rng(42)
    fens = [all_fens[i] for i in rng.choice(len(all_fens), args.n, replace=False)]

    print("Computing embeddings for each model...")
    embs = {lbl: embed(m, fens) for lbl, m in models.items()}

    # Build full concatenated ensemble + all 3 leave-one-out versions
    full_emb = np.concatenate([embs[l] for l in args.labels], axis=1)
    loo_embs = {}
    for i, lbl in enumerate(args.labels):
        keep = [l for l in args.labels if l != lbl]
        loo_embs[f"−{lbl}"] = np.concatenate([embs[k] for k in keep], axis=1)
    # Also include each individual
    individual_embs = {f"only {l}": embs[l] for l in args.labels}

    # Labels
    print("Computing labels...")
    labels = {name: np.array([fn(f) for f in fens]) for name, fn, _ in PROBES}

    # Evaluate each variant on each probe
    variants = {"FULL": full_emb, **loo_embs, **individual_embs}

    print()
    header = f"  {'Concept':<14} {'kind':<4}" + "".join(f" {n:>11}" for n in variants)
    print(header)
    print("-" * len(header))
    for concept, _, kind in PROBES:
        y = labels[concept]
        row = f"  {concept:<14} {kind:<4}"
        for vname, vemb in variants.items():
            X_tr, X_val, y_tr, y_val = train_test_split(
                vemb, y, test_size=0.2, random_state=42)
            s = fit_probe(X_tr, y_tr, X_val, y_val, kind)
            if kind == "cls":
                row += f" {s:>10.1f}%"
            else:
                row += f"  R² {s:>+6.3f}"
        print(row)

    # Delta table: how much does FULL drop when removing each model?
    print("\n### Contribution of each model to the FULL ensemble")
    print("### (drop when that model is removed, i.e., FULL − FULL\\{model})")
    print()
    print(f"  {'Concept':<14}" + "".join(f" {'−'+l:>12}" for l in args.labels))
    print("-" * 60)
    for concept, _, kind in PROBES:
        y = labels[concept]
        X_tr, X_val, y_tr, y_val = train_test_split(
            full_emb, y, test_size=0.2, random_state=42)
        full_score = fit_probe(X_tr, y_tr, X_val, y_val, kind)
        row = f"  {concept:<14}"
        for lbl in args.labels:
            loo = loo_embs[f"−{lbl}"]
            X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
                loo, y, test_size=0.2, random_state=42)
            loo_score = fit_probe(X_tr2, y_tr2, X_val2, y_val2, kind)
            delta = full_score - loo_score
            if kind == "cls":
                row += f" {delta:>+11.1f}pp"
            else:
                row += f" ΔR² {delta:>+7.3f}"
        print(row)


if __name__ == "__main__":
    main()
