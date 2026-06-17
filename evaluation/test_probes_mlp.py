#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_probes_mlp.py — Tier 2 probes with a small MLP (non-linear) head.

Linear probes in test_probes_strategic.py measure *linear* separability of
concepts in the embedding. An MLP probe measures whether the information is
*present* in the embedding at all — even if tangled in a non-linear way.

If MLP_score >> linear_score, the embedding has the info but it's encoded
non-linearly. If MLP_score ~ linear_score, there's no extra hidden info.

Each probe trains a 2-layer MLP on the frozen embeddings: emb_dim -> 128 -> out.
Compared against untrained-CNN (control) and linear logreg (from previous test).
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus
from test_probes_strategic import PROBES, embed_batch


N_POSITIONS = 20000
HIDDEN = 128
EPOCHS = 30
BATCH = 512
LR = 1e-3


class MLPClf(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


class MLPReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, kind,
              device=("cuda" if torch.cuda.is_available() else "cpu")):
    in_dim = X_tr.shape[1]
    X_tr = torch.from_numpy(X_tr).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    if kind == 'cls':
        n_cls = int(max(np.max(y_tr), np.max(y_val))) + 1
        model = MLPClf(in_dim, n_cls).to(device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)
        y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = MLPReg(in_dim).to(device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
        loss_fn = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    n = len(X_tr)
    best = -1e9
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X_tr[idx], y_tr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            if kind == 'cls':
                acc = (pred.argmax(1) == y_val_t).float().mean().item()
                score = acc
            else:
                ss_res = ((pred - y_val_t) ** 2).sum().item()
                ss_tot = ((y_val_t - y_val_t.mean()) ** 2).sum().item()
                score = 1 - ss_res / max(ss_tot, 1e-12)
            if score > best:
                best = score
    return best


def run_mlp_probe(emb, labels, kind):
    X_tr, X_val, y_tr, y_val = train_test_split(
        emb, labels, test_size=0.2, random_state=42)
    if kind == 'cls' and len(np.unique(y_tr)) < 2:
        return float('nan')
    score = train_mlp(X_tr, y_tr, X_val, y_val, kind)
    return score


def fmt(score, kind):
    if np.isnan(score): return "   -  "
    if kind == 'cls': return f"{score*100:5.1f}%"
    return f"R²={score:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--n", type=int, default=N_POSITIONS)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"Loaded {args.ckpt}  val_acc={ckpt.get('val_acc',0):.2f}%")

    trained = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    trained.load_state_dict(ckpt['model']); trained.eval()

    torch.manual_seed(42)
    untrained = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    untrained.eval()

    print(f"Loading {args.n} positions...")
    all_fens = load_corpus(args.corpus, max_n=50000)
    np.random.seed(42)
    fens = [all_fens[i] for i in np.random.choice(len(all_fens), args.n, replace=False)]

    print("Computing labels + embeddings...")
    labels = {name: np.array([fn(f) for f in fens]) for name, fn, _ in PROBES}
    emb_t = embed_batch(trained, fens)
    emb_u = embed_batch(untrained, fens)

    hdr = f"  {'Concept':<22}  {'kind':<4}  {'MLP trained':>12}  {'MLP untrained':>14}  {'MLP lift':>10}"
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))
    for name, _, kind in PROBES:
        y = labels[name]
        t_score = run_mlp_probe(emb_t, y, kind)
        u_score = run_mlp_probe(emb_u, y, kind)
        lift = (t_score - u_score) * (100 if kind == 'cls' else 1)
        print(f"  {name:<22}  {kind:<4}  {fmt(t_score,kind):>12}  {fmt(u_score,kind):>14}  {lift:>+10.2f}")
    print("="*len(hdr))


if __name__ == "__main__":
    main()
