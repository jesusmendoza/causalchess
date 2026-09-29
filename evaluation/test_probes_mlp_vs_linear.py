#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_probes_mlp_vs_linear.py — Tier-2 comparison, same positions and same
checkpoint for both probe families.

The Tier-2 table in the notes compared MLP-trained against the MLP *untrained*
column while labelling it "Linear probe". This script reports the true Ridge
(Tier-1) score beside the MLP score on identical embeddings, so the non-linear
lift is computed against the right baseline.

Usage:
    python3 evaluation/test_probes_mlp_vs_linear.py \
        --ckpt models/prev_move_cnn_v2.pt
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus
from test_probes_strategic import PROBES
from test_probes_mlp import run_mlp_probe, N_POSITIONS

SEED = 42


def seed_probe():
    """Make each MLP probe reproducible regardless of probe order."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def embed_batch(model, fens, batch=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i + batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(device)
        with torch.no_grad():
            out.append(model.get_embedding(x).cpu().numpy())
    return np.vstack(out)


def run_ridge(emb, labels):
    X_tr, X_val, y_tr, y_val = train_test_split(
        emb, labels, test_size=0.2, random_state=42)
    clf = Ridge()
    clf.fit(X_tr, y_tr)
    return r2_score(y_val, clf.predict(X_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--n", type=int, default=N_POSITIONS)
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    args = ap.parse_args()
    seed_probe()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = create_model(ckpt["arch"], ckpt["n_moves"], embed_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}  val_acc={ckpt.get('val_acc', 0):.3f}%")

    all_fens = load_corpus(args.corpus, max_n=50000)
    np.random.seed(SEED)
    idx = np.random.choice(len(all_fens), args.n, replace=False)
    fens = [all_fens[i] for i in idx]
    print(f"positions={len(fens)}  (seed 42, corpus={args.corpus})")

    emb = embed_batch(model, fens)

    reg_probes = [(name, fn) for name, fn, kind in PROBES if kind == "reg"]
    print(f"\n{'Concept':<22}{'Ridge (linear)':>16}{'MLP (trained)':>16}"
          f"{'MLP lift':>12}")
    print("-" * 66)
    for name, fn in reg_probes:
        labels = np.array([fn(f) for f in fens])
        lin = run_ridge(emb, labels)
        seed_probe()
        mlp = run_mlp_probe(emb, labels, "reg")
        print(f"  {name:<20}{lin:>+16.3f}{mlp:>+16.3f}{mlp - lin:>+12.3f}")


if __name__ == "__main__":
    main()
