#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_finetune.py — Tier 3: end-to-end fine-tune on next-move prediction.

Tier 1 (linear probe) and Tier 2 (MLP probe) freeze the embedding. Tier 3
unfreezes the full encoder and fine-tunes it on a downstream task. The
question is: does the prev-move pretraining produce better INIT weights
than random for next-move prediction?

We compare three regimes on the same next-move data (from lichess_2013_01.pgn):
  - Baseline: random-init CNN, fine-tuned end-to-end
  - PMP-mem: pretrained PMP-mem CNN, fine-tuned end-to-end
  - PMP-no-mem: pretrained PMP-no-mem CNN, fine-tuned end-to-end

Metric: val accuracy on top-500-classes next-move prediction.

If fine-tuning from PMP beats fine-tuning from scratch, the pretraining
is a USEFUL init — transfer learning works.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


PGN_PATH = "lichess_data/lichess_2013_01.pgn"
N_GAMES = 3000
TOP_K = 500
EPOCHS = 8
BATCH = 256
LR = 1e-4


def extract_pairs(pgn_path, n_games):
    pairs = []
    games = 0
    with open(pgn_path) as f:
        while games < n_games:
            game = chess.pgn.read_game(f)
            if game is None: break
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < 4: continue
            for i, m in enumerate(moves):
                if 3 <= i < 200:
                    pairs.append((board.fen(), m.uci()))
                board.push(m)
            games += 1
            if games % 200 == 0:
                print(f"  {games} games, {len(pairs)} pairs", flush=True)
    return pairs


class FinetuneHead(nn.Module):
    def __init__(self, encoder, embed_dim, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)
    def forward(self, x):
        emb = self.encoder.get_embedding(x)
        return self.head(emb)


def prepare_tensors(pairs, vocab):
    fens, labels = [], []
    for f, m in pairs:
        if m in vocab:
            fens.append(f); labels.append(vocab[m])
    X = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in fens])
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def fit(model, X_tr, y_tr, X_val, y_val, device, seed):
    model.to(device)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(X_tr)
    best = 0.0
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()
        print(f"    epoch {ep+1}/{EPOCHS}  val_acc={acc*100:.2f}%")
        if acc > best: best = acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt",
                    help="Pretrained PMP checkpoint. Pass one per run; "
                         "to compare mem/no-mem, invoke the script twice.")
    ap.add_argument("--label", default="pmp",
                    help="Friendly label for output (mem/no-mem/scratch).")
    ap.add_argument("--n-games", type=int, default=N_GAMES)
    ap.add_argument("--n-cap", type=int, default=40000,
                    help="Max pairs after top-K filter (for wall-time control).")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Ignore --ckpt, initialize encoder randomly "
                         "(scratch baseline).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for split / init / batch shuffle.")
    ap.add_argument("--k", type=int, default=1,
                    help="If >1, run seeds 0..k-1 and report mean±std.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print(f"Extracting next-move pairs from {args.n_games} games...")
    pairs = extract_pairs(PGN_PATH, args.n_games)
    print(f"  {len(pairs)} pairs")

    from collections import Counter
    counts = Counter(m for _, m in pairs)
    top_moves = [m for m, _ in counts.most_common(TOP_K)]
    vocab = {m: i for i, m in enumerate(top_moves)}
    filtered = [(f, m) for f, m in pairs if m in vocab]
    print(f"  {len(filtered)} pairs after top-{TOP_K} filter")

    # Fixed corpus subsample; seed only affects split + training dynamics
    if len(filtered) > args.n_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(filtered), args.n_cap, replace=False)
        filtered = [filtered[i] for i in idx]

    X, y = prepare_tensors(filtered, vocab)
    print(f"  corpus={len(X)} after cap")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    seeds = list(range(args.k)) if args.k > 1 else [args.seed]
    bests = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        n_val = len(X) // 5
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        print(f"\n=== Fine-tune ({args.label}) seed={seed} "
              f"train={len(X_tr)} val={len(X_val)} ===")

        encoder = create_model(ckpt['arch'], ckpt['n_moves'],
                               embed_dim=ckpt['embed_dim'])
        if args.from_scratch:
            torch.manual_seed(seed)
            for p in encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            print(f"  SCRATCH baseline: random-init encoder (seed={seed})")
        else:
            encoder.load_state_dict(ckpt['model'])
            print(f"  Pretrained from {args.ckpt}  "
                  f"val={ckpt.get('val_acc',0):.2f}%")

        # Fresh head each seed
        torch.manual_seed(seed + 17)
        model = FinetuneHead(encoder, ckpt['embed_dim'], len(vocab))
        best = fit(model, X_tr, y_tr, X_val, y_val, device, seed)
        bests.append(best)
        print(f"### {args.label} seed={seed}: best val_acc = {best*100:.2f}%")

    arr = np.array(bests, dtype=np.float64)
    if len(arr) == 1:
        print(f"\n### {args.label}: best val_acc = {arr[0]*100:.2f}%")
    else:
        print(f"\n### {args.label}: best val_acc = "
              f"{arr.mean()*100:.2f}±{arr.std()*100:.2f}%  (K={len(arr)})")


if __name__ == "__main__":
    main()
