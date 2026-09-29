#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
build_strategic_table.py — rebuild Table `tab:strategic` from checkpoints.

Reports Ridge R^2 on frozen embeddings for the four pretraining objectives,
using one shared 20K position sample. Values are the mean of the white-side
and black-side probes for the side-symmetric concepts.

The previous table mixed runs and labelled trained-minus-untrained lifts as
R^2; this regenerates it from the checkpoints that the rest of the paper uses.

Usage:
    python3 evaluation/build_strategic_table.py
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

MODELS = [
    ("PMP-mem",     "models/prev_move_cnn_v2.pt"),
    ("PMP-no-mem",  "models/prev_move_cnn_v2_dedup.pt"),
    ("AE",          "models/ae_cnn.pt"),
    ("SimCLR",      "models/simclr_cnn_final.pt"),
]
ROWS = [
    ("Development",     "development W",     "development B"),
    ("Center Control",  "center control W",  "center control B"),
    ("Space",           "space W",           "space B"),
    ("King Safety",     "king shield W",     "king shield B"),
]
N_POSITIONS = 20000


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


def ridge_r2(emb, labels):
    X_tr, X_val, y_tr, y_val = train_test_split(
        emb, labels, test_size=0.2, random_state=42)
    clf = Ridge()
    clf.fit(X_tr, y_tr)
    return r2_score(y_val, clf.predict(X_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_POSITIONS)
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    args = ap.parse_args()

    fns = {name: fn for name, fn, _ in PROBES}
    all_fens = load_corpus(args.corpus, max_n=50000)
    np.random.seed(42)
    idx = np.random.choice(len(all_fens), args.n, replace=False)
    fens = [all_fens[i] for i in idx]
    labels = {n: np.array([fns[n](f) for f in fens]) for _, w, b in ROWS for n in (w, b)}
    print(f"positions={len(fens)}  corpus={args.corpus}  (seed 42)")

    scores = {}
    for mname, path in MODELS:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = create_model(ck["arch"], ck["n_moves"], embed_dim=ck["embed_dim"])
        model.load_state_dict(ck["model"])
        model.eval()
        emb = embed_batch(model, fens)
        print(f"  {mname:<12} {_os.path.basename(path):<32} "
              f"val_acc={ck.get('val_acc', 0):.3f}%")
        scores[mname] = {
            disp: 0.5 * (ridge_r2(emb, labels[w]) + ridge_r2(emb, labels[b]))
            for disp, w, b in ROWS
        }

    print(f"\nRidge R^2 on frozen embeddings (mean of white/black probes)\n")
    print(f"{'Concept':<18}" + "".join(f"{m[0]:>13}" for m in MODELS))
    print("-" * (18 + 13 * len(MODELS)))
    for disp, _, _ in ROWS:
        vals = [scores[m[0]][disp] for m in MODELS]
        cells = [f"{v:>13.3f}" for v in vals]
        print(f"{disp:<18}" + "".join(cells))

    print("\nLaTeX body (bold = max, ties bolded):")
    for disp, _, _ in ROWS:
        vals = [scores[m[0]][disp] for m in MODELS]
        top = max(vals)
        cells = []
        for v in vals:
            cells.append(f"\\textbf{{{v:.2f}}}" if abs(v - top) < 5e-3 else f"{v:.2f}")
        print(f"{disp} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
