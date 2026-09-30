#!/usr/bin/env python3
"""
test_probes_raw_baseline.py — Sprint 1A/1B (R&R ToG).

Compare linear concept probes on:
  - raw 768-d bitboards (initial state; R2/R3 baseline)
  - frozen PMP / next-move / AE / SimCLR / untrained CNN embeddings (256-d)

Reports mean ± std over K probe seeds (default 3).
Uses torch-only probes (no sklearn) for environment portability.

Usage:
  python3 evaluation/test_probes_raw_baseline.py --k 3 --n 20000
  python3 evaluation/test_probes_raw_baseline.py --ckpt-next models/next_move_cnn.pt
"""
import sys as _sys
import os as _os

_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from prev_move_models import create_model, fen_to_bitboards


# ── Labels (same definitions as test_probes.py) ─────────────────────

def label_phase(fen):
    board = chess.Board(fen)
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for pt, v in pmap.items():
        total += (len(board.pieces(pt, chess.WHITE))
                  + len(board.pieces(pt, chess.BLACK))) * v
    if total >= 70:
        return 0
    if total >= 40:
        return 1
    return 2


def label_castled_white(fen):
    """King on c1/g1 with no castling rights; does not record a castling move."""
    board = chess.Board(fen)
    wk_sq = board.king(chess.WHITE)
    can_castle = (board.has_kingside_castling_rights(chess.WHITE)
                  or board.has_queenside_castling_rights(chess.WHITE))
    return 1 if wk_sq in (chess.G1, chess.C1) and not can_castle else 0


def label_castled_black(fen):
    """King on c8/g8 with no castling rights; does not record a castling move."""
    board = chess.Board(fen)
    bk_sq = board.king(chess.BLACK)
    can_castle = (board.has_kingside_castling_rights(chess.BLACK)
                  or board.has_queenside_castling_rights(chess.BLACK))
    return 1 if bk_sq in (chess.G8, chess.C8) and not can_castle else 0


def label_material(fen):
    board = chess.Board(fen)
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    val = 0
    for pt, v in pmap.items():
        val += len(board.pieces(pt, chess.WHITE)) * v
        val -= len(board.pieces(pt, chess.BLACK)) * v
    return float(val)


def label_piece_count(fen):
    return float(len(chess.Board(fen).piece_map()))


def label_white_isolated_pawns(fen):
    board = chess.Board(fen)
    wp_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)}
    isolated = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        f = chess.square_file(sq)
        if (f - 1 not in wp_files) and (f + 1 not in wp_files):
            isolated += 1
    return isolated


def label_open_files(fen):
    board = chess.Board(fen)
    files_with_pawn = set()
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        files_with_pawn.add(chess.square_file(sq))
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        files_with_pawn.add(chess.square_file(sq))
    return 8 - len(files_with_pawn)


def label_turn(fen):
    return 0 if chess.Board(fen).turn == chess.WHITE else 1


def label_in_check(fen):
    return 1 if chess.Board(fen).is_check() else 0


PROBES = [
    ("phase", label_phase, "cls"),
    ("castled_W", label_castled_white, "cls"),
    ("castled_B", label_castled_black, "cls"),
    ("turn", label_turn, "cls"),
    ("in_check", label_in_check, "cls"),
    ("material", label_material, "reg"),
    ("piece_count", label_piece_count, "reg"),
    ("isolated_W", label_white_isolated_pawns, "reg"),
    ("open_files", label_open_files, "reg"),
]


def load_corpus(tsv_path, max_n=50000):
    fens = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fens.append(row["fen"])
            if len(fens) >= max_n:
                break
    return fens


def load_trained(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt["arch"], ckpt["n_moves"], embed_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_untrained(ckpt):
    torch.manual_seed(42)
    model = create_model(ckpt["arch"], ckpt["n_moves"], embed_dim=ckpt["embed_dim"])
    model.eval()
    return model


def embed_batch(model, fens, batch=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_embs = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i + batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(device)
        with torch.no_grad():
            emb = model.get_embedding(x).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs)


def raw_features(fens):
    return np.stack([fen_to_bitboards(f) for f in fens], axis=0)


def train_test_split_idx(n, test_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * test_size))
    return idx[n_val:], idx[:n_val]


def probe_cls(X, y, seed, max_iter=200):
    tr, va = train_test_split_idx(len(y), 0.2, seed)
    Xtr = torch.from_numpy(X[tr]).float()
    ytr = torch.from_numpy(y[tr]).long()
    Xva = torch.from_numpy(X[va]).float()
    yva = torch.from_numpy(y[va]).long()
    mu, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - mu) / std
    Xva = (Xva - mu) / std
    n_classes = int(y.max()) + 1
    W = nn.Parameter(torch.zeros(X.shape[1], n_classes))
    b = nn.Parameter(torch.zeros(n_classes))
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(((Xva @ W + b).argmax(1) == yva).float().mean())


def probe_reg(X, y, seed, alpha=1.0):
    tr, va = train_test_split_idx(len(y), 0.2, seed)
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr].astype(np.float64), y[va].astype(np.float64)
    mu, std = Xtr.mean(0), Xtr.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    Xtr = (Xtr - mu) / std
    Xva = (Xva - mu) / std
    Xtr_b = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xva_b = np.hstack([Xva, np.ones((len(Xva), 1))])
    d = Xtr_b.shape[1]
    A = Xtr_b.T @ Xtr_b + alpha * np.eye(d)
    A[-1, -1] -= alpha
    w = np.linalg.solve(A, Xtr_b.T @ ytr)
    pred = Xva_b @ w
    ss_res = np.sum((yva - pred) ** 2)
    ss_tot = np.sum((yva - yva.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def probe_once(X, y, kind, seed):
    return probe_cls(X, y, seed) if kind == "cls" else probe_reg(X, y, seed)


def probe_k(X, y, kind, k):
    scores = [probe_once(X, y, kind, seed=s) for s in range(k)]
    return float(np.mean(scores)), float(np.std(scores))


def fmt(mean, std, kind):
    if kind == "cls":
        return f"{100 * mean:5.1f}±{100 * std:4.1f}%"
    return f"{mean:+.3f}±{std:.3f}"


def load_optional(path, label):
    if not path or not _os.path.isfile(path):
        print(f"  skip {label}: missing {path}")
        return None
    model, ckpt = load_trained(path)
    print(f"  loaded {label}: {path} (arch={ckpt.get('arch')})")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-mem", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--ckpt-next", default="models/next_move_cnn.pt",
                    help="Same-arch next-move CNN baseline (H1.1/H1.3)")
    ap.add_argument("--ckpt-nomem", default="models/prev_move_cnn_v2_dedup.pt")
    ap.add_argument("--ckpt-ae", default="models/ae_cnn.pt")
    ap.add_argument("--ckpt-simclr", default="models/simclr_cnn_final.pt")
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=50000)
    ap.add_argument("--out-md", default="",
                    help="Markdown results path (default: results_1b if next-move loaded)")
    args = ap.parse_args()

    print(f"Loading up to {args.pool} FENs from {args.corpus}...")
    fens = load_corpus(args.corpus, max_n=args.pool)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(fens), size=min(args.n, len(fens)), replace=False)
    fens = [fens[i] for i in idx]
    print(f"  using {len(fens)} positions, K={args.k} probe seeds")

    print("Computing labels...")
    labels = {name: np.array([fn(f) for f in fens]) for name, fn, _ in PROBES}

    features = {}
    print("Building raw 768-d bitboards...")
    features["raw"] = raw_features(fens)

    print("Loading models / embeddings...")
    # Prefer column order: raw | PMP | next-move | controls
    ordered = ["raw"]

    mem = load_optional(args.ckpt_mem, "PMP-mem")
    if mem is not None:
        model, ckpt = mem
        features["PMP-mem"] = embed_batch(model, fens)
        ordered.append("PMP-mem")
        features["untrained"] = embed_batch(load_untrained(ckpt), fens)

    nxt = load_optional(args.ckpt_next, "next-move")
    if nxt is not None:
        features["next-move"] = embed_batch(nxt[0], fens)
        ordered.append("next-move")

    nomem = load_optional(args.ckpt_nomem, "PMP-no-mem")
    if nomem is not None:
        features["PMP-no-mem"] = embed_batch(nomem[0], fens)
        ordered.append("PMP-no-mem")

    ae = load_optional(args.ckpt_ae, "AE")
    if ae is not None:
        features["AE"] = embed_batch(ae[0], fens)
        ordered.append("AE")

    sim = load_optional(args.ckpt_simclr, "SimCLR")
    if sim is not None:
        features["SimCLR"] = embed_batch(sim[0], fens)
        ordered.append("SimCLR")

    if "untrained" in features:
        ordered.append("untrained")

    cols = [c for c in ordered if c in features]
    print("\n" + "=" * 100)
    print(f"{'Concept':<12}" + "".join(f"{c:>16}" for c in cols))
    print("=" * 100)

    rows = []
    for name, _, kind in PROBES:
        y = labels[name]
        line = f"{name:<12}"
        row = {"concept": name, "kind": kind}
        for c in cols:
            mean, std = probe_k(features[c], y, kind, args.k)
            row[c] = (mean, std)
            line += f"{fmt(mean, std, kind):>16}"
            print(f"  ... {name}/{c} done", flush=True)
        print(line)
        rows.append(row)
    print("=" * 100)
    print("cls = accuracy (%) | reg = R² | mean±std over probe seeds")
    print("raw = linear probe on 768-d input bitboards (no neural embedding)")
    if "next-move" in features:
        print("next-move = same CNN trunk trained on next-move prediction (H1.1/H1.3)")

    if args.out_md:
        out_md = args.out_md
    elif "next-move" in features:
        out_md = "paper/revision_tog_2026/results_1b_next_move_probes.md"
    else:
        out_md = "paper/revision_tog_2026/results_1a_raw_probes.md"
    _os.makedirs(_os.path.dirname(out_md), exist_ok=True)
    title = ("Sprint 1B — Raw vs PMP vs next-move probes"
             if "next-move" in features else
             "Sprint 1A — Raw-input linear probes")
    with open(out_md, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"N={len(fens)}, K={args.k} probe seeds, corpus=`{args.corpus}`\n\n")
        if "next-move" in features:
            f.write(f"- PMP ckpt: `{args.ckpt_mem}`\n")
            f.write(f"- next-move ckpt: `{args.ckpt_next}`\n\n")
        f.write("| Concept | " + " | ".join(cols) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in rows:
            cells = [row["concept"]]
            for c in cols:
                mean, std = row[c]
                cells.append(fmt(mean, std, row["kind"]))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n_Generated by `evaluation/test_probes_raw_baseline.py`._\n")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
