#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_ensemble.py — Run key evaluations on the CONCATENATION of multiple
pretrained embeddings, to test whether different pretraining objectives are
complementary or redundant.

Given N checkpoints (e.g., AE + SimCLR-flip + SimCLR-temporal + NextMove +
PMP-mem + PMP-no-mem), build the 256×N-dimensional
concatenated embedding per position and run:

  (a) Linear probes (9 basic chess concepts)
  (b) Player style detection (4 GMs)
  (c) Next-move transfer (linear head, top-500 moves)
  (d) Strategic distance (material-controlled Ridge on Stockfish evals)

Compares the ensemble's score against each individual component.

If the ensemble consistently outperforms each component, pretrainings are
complementary. If it matches the best individual, they are redundant.

Usage:
    python3 evaluation/test_ensemble.py \\
        --ckpts models/ae_cnn.pt models/simclr_cnn.pt models/prev_move_cnn_v2.pt \\
        --labels AE SimCLR Mem
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
from test_probes_strategic import (PROBES, label_phase, label_castled_white,
                                    label_castled_black, label_turn,
                                    label_in_check, label_material_balance,
                                    label_piece_count, label_white_isolated_pawns,
                                    label_open_files)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get('arch', 'cnn')
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval().to(DEVICE)
    return model, ckpt


def embed_corpus(model, fens, batch=1024):
    """Compute embeddings on GPU if available."""
    out = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(DEVICE)
        with torch.no_grad():
            out.append(model.get_embedding(x).cpu().numpy())
    return np.vstack(out)


# ── Test A: basic probes ───────────────────────────────────────────

PROBE_FNS = [
    ("phase", label_phase, 'cls'),
    ("castled_W", label_castled_white, 'cls'),
    ("castled_B", label_castled_black, 'cls'),
    ("turn", label_turn, 'cls'),
    ("in_check", label_in_check, 'cls'),
    ("material_balance", label_material_balance, 'reg'),
    ("piece_count", label_piece_count, 'reg'),
    ("isolated_pawns_W", label_white_isolated_pawns, 'reg'),
    ("open_files", label_open_files, 'reg'),
]


def run_probes(embeddings_by_name, fens, labels_dict):
    """Train linear probes on each embedding, compare."""
    print("\n" + "=" * 90)
    print("  A. Linear probes (9 concepts, lift over untrained)")
    print("=" * 90)
    header = f"  {'Concept':<18} {'kind':<4}" + "".join(f" {n:>10}" for n in embeddings_by_name)
    print(header)
    print("-" * len(header))
    for name, _, kind in PROBE_FNS:
        y = labels_dict[name]
        row = f"  {name:<18} {kind:<4}"
        for emb_name, emb in embeddings_by_name.items():
            X_tr, X_val, y_tr, y_val = train_test_split(
                emb, y, test_size=0.2, random_state=42)
            if kind == 'cls':
                if len(np.unique(y_tr)) < 2:
                    row += f" {'-':>10}"
                    continue
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_tr, y_tr)
                s = accuracy_score(y_val, clf.predict(X_val)) * 100
                row += f" {s:>9.1f}%"
            else:
                clf = Ridge(alpha=1.0)
                clf.fit(X_tr, y_tr)
                s = r2_score(y_val, clf.predict(X_val))
                row += f" R²{s:>+7.3f}"
        print(row)


# ── Test B: style detection ─────────────────────────────────────────

STYLE_PLAYERS = ['penguingim1', 'msb2', 'NihalSarin', 'DrNykterstein']
STYLE_PGN = "data_style/gm_games.pgn"


def extract_style_data(pgn_path, players, min_ply=10):
    from collections import defaultdict
    data = defaultdict(list)
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            w = game.headers.get("White", "")
            b = game.headers.get("Black", "")
            target = None
            if w in players: target = w
            elif b in players: target = b
            if target is None: continue
            board = game.board()
            for i, m in enumerate(game.mainline_moves()):
                board.push(m)
                if i >= min_ply:
                    data[target].append(board.fen())
    return data


def run_style(embeddings_by_name, models_by_name, pgn_path):
    if not os.path.exists(pgn_path):
        print("\n[B. Style detection skipped — PGN not found]")
        return
    print("\n" + "=" * 90)
    print("  B. Style detection (4 GMs)")
    print("=" * 90)
    data = extract_style_data(pgn_path, STYLE_PLAYERS)
    min_n = min(len(v) for v in data.values())
    fens, labels = [], []
    for i, p in enumerate(STYLE_PLAYERS):
        for f in data[p][:min_n]:
            fens.append(f); labels.append(i)
    labels = np.array(labels)
    print(f"  {len(fens)} positions, {len(STYLE_PLAYERS)} classes, random = 25%")
    header = f"  {'Source':<30}"
    for emb_name in embeddings_by_name: header += f" {emb_name:>10}"
    # Re-embed with fresh style FENs
    embs_style = {n: embed_corpus(m, fens) for n, m in models_by_name.items()}
    if 'ENSEMBLE' in embeddings_by_name:
        embs_style['ENSEMBLE'] = np.concatenate(
            [embs_style[n] for n in embeddings_by_name if n != 'ENSEMBLE'], axis=1)
    row = f"  {'Style accuracy':<30}"
    for emb_name in embeddings_by_name:
        X_tr, X_val, y_tr, y_val = train_test_split(
            embs_style[emb_name], labels, test_size=0.2, random_state=42, stratify=labels)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        s = accuracy_score(y_val, clf.predict(X_val)) * 100
        row += f" {s:>9.1f}%"
    print(header)
    print(row)


# ── Test C: next-move transfer ──────────────────────────────────────

def extract_nextmove_pairs(pgn_path, n_games=1500):
    pairs = []
    n = 0
    with open(pgn_path) as f:
        while n < n_games:
            g = chess.pgn.read_game(f)
            if g is None: break
            board = g.board()
            moves = list(g.mainline_moves())
            if len(moves) < 4: continue
            for i, m in enumerate(moves):
                if 3 <= i < 150:
                    pairs.append((board.fen(), m.uci()))
                board.push(m)
            n += 1
    return pairs


def run_nextmove(embeddings_by_name, models_by_name, pgn_path):
    if not os.path.exists(pgn_path):
        print("\n[C. Next-move skipped — PGN not found]")
        return
    print("\n" + "=" * 90)
    print("  C. Next-move transfer (linear head, top-500 classes)")
    print("=" * 90)
    pairs = extract_nextmove_pairs(pgn_path, n_games=1500)
    from collections import Counter
    cnt = Counter(m for _, m in pairs)
    top = [m for m, _ in cnt.most_common(500)]
    vocab = {m: i for i, m in enumerate(top)}
    filtered = [(f, vocab[m]) for f, m in pairs if m in vocab]
    if len(filtered) > 30000:
        idx = np.random.default_rng(42).choice(len(filtered), 30000, replace=False)
        filtered = [filtered[i] for i in idx]
    fens = [f for f, _ in filtered]
    labels = np.array([y for _, y in filtered])
    print(f"  {len(fens)} (position, next_move) pairs, 500 classes, random = {100/500:.2f}%")
    embs = {n: embed_corpus(m, fens) for n, m in models_by_name.items()}
    if 'ENSEMBLE' in embeddings_by_name:
        embs['ENSEMBLE'] = np.concatenate(
            [embs[n] for n in embeddings_by_name if n != 'ENSEMBLE'], axis=1)
    header = f"  {'Metric':<20}"
    for n in embeddings_by_name: header += f" {n:>10}"
    print(header)
    row = f"  {'Top-1 acc':<20}"
    for emb_name in embeddings_by_name:
        X_tr, X_val, y_tr, y_val = train_test_split(
            embs[emb_name], labels, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_tr, y_tr)
        s = accuracy_score(y_val, clf.predict(X_val)) * 100
        row += f" {s:>9.2f}%"
    print(row)


# ── Test D: strategic distance controlled ──────────────────────────

def run_strategic(embeddings_by_name, models_by_name, evals_path):
    if not os.path.exists(evals_path):
        print("\n[D. Strategic distance skipped — evals not found]")
        return
    print("\n" + "=" * 90)
    print("  D. Strategic distance (Ridge predict Stockfish eval)")
    print("=" * 90)
    fens, evals = [], []
    with open(evals_path) as f:
        hdr = f.readline().strip().split('\t')
        i_fen = hdr.index('fen'); i_ev = hdr.index('eval_cp')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            try:
                e = float(parts[i_ev])
            except ValueError: continue
            fens.append(parts[i_fen]); evals.append(e)
    evals = np.clip(np.array(evals), -2000, 2000)
    print(f"  {len(fens)} positions, eval clipped to ±2000 cp")
    embs = {n: embed_corpus(m, fens) for n, m in models_by_name.items()}
    if 'ENSEMBLE' in embeddings_by_name:
        embs['ENSEMBLE'] = np.concatenate(
            [embs[n] for n in embeddings_by_name if n != 'ENSEMBLE'], axis=1)
    idx_tr, idx_val = train_test_split(np.arange(len(fens)), test_size=0.2, random_state=42)
    header = f"  {'Metric':<20}"
    for n in embeddings_by_name: header += f" {n:>10}"
    print(header)
    row_r2 = f"  {'R² eval':<20}"
    row_mae = f"  {'MAE cp':<20}"
    for emb_name in embeddings_by_name:
        clf = Ridge(alpha=1.0)
        clf.fit(embs[emb_name][idx_tr], evals[idx_tr])
        pred = clf.predict(embs[emb_name][idx_val])
        row_r2 += f" {r2_score(evals[idx_val], pred):>+10.3f}"
        row_mae += f" {mean_absolute_error(evals[idx_val], pred):>9.0f} "
    print(row_r2)
    print(row_mae)


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs='+', required=True,
                    help="List of checkpoint paths to ensemble.")
    ap.add_argument("--labels", nargs='+', default=None,
                    help="Names for each checkpoint (default = file stems).")
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--n", type=int, default=20000,
                    help="Positions from corpus for probe tests.")
    ap.add_argument("--pgn", default="lichess_data/lichess_2013_01.pgn")
    ap.add_argument("--style-pgn", default=STYLE_PGN)
    ap.add_argument("--evals", default="lichess_data/stockfish_evals.tsv")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.ckpts):
        raise ValueError("--labels must have same length as --ckpts")
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.ckpts]

    print(f"Loading {len(args.ckpts)} checkpoints...")
    models = {}
    for lbl, path in zip(labels, args.ckpts):
        print(f"  {lbl}: {path}")
        m, _ = load_model_from_ckpt(path)
        models[lbl] = m

    # Probe corpus + labels
    print(f"\nLoading {args.n} corpus positions...")
    all_fens = load_corpus(args.corpus, max_n=50000)
    rng = np.random.default_rng(42)
    fens = [all_fens[i] for i in rng.choice(len(all_fens), args.n, replace=False)]

    print("Computing labels...")
    labels_dict = {name: np.array([fn(f) for f in fens]) for name, fn, _ in PROBE_FNS}

    print("Embedding corpus through each model...")
    embs = {lbl: embed_corpus(m, fens) for lbl, m in models.items()}
    # Ensemble = concatenation
    embs['ENSEMBLE'] = np.concatenate([embs[lbl] for lbl in labels], axis=1)
    print(f"  Ensemble dim = {embs['ENSEMBLE'].shape[1]}")

    # Run tests
    run_probes(embs, fens, labels_dict)
    run_style(embs, models, args.style_pgn)
    run_nextmove(embs, models, args.pgn)
    run_strategic(embs, models, args.evals)

    print("\n" + "=" * 90)
    print("Done. Ensemble vs individual: is concatenation complementary or redundant?")


if __name__ == "__main__":
    main()
