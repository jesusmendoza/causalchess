#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_probes.py — Linear probes for chess concepts on frozen embedding.

For each concept (phase, castled, material, isolated pawns, ...):
- Extract label from FEN using python-chess
- Train linear classifier on top of frozen embedding
- Report val accuracy / R²

Compare trained CNN vs untrained CNN (baseline).
If trained >> untrained, embedding has learned that concept.

Standard rep learning evaluation (BERT probes, CLIP probes).
"""
import numpy as np
import torch
import chess
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus


N_POSITIONS = 20000
LOGREG_MAX_ITER = 20000
RIDGE_SOLVER = "svd"
PROBE_DTYPE = np.float64


# ── Concept label extractors ─────────────────────────────

def label_phase(fen):
    """0=opening (>=70 material), 1=mid (40-70), 2=end (<40)."""
    board = chess.Board(fen)
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for pt, v in pmap.items():
        total += (len(board.pieces(pt, chess.WHITE))
                  + len(board.pieces(pt, chess.BLACK))) * v
    if total >= 70: return 0
    elif total >= 40: return 1
    else: return 2


def label_castled_white(fen):
    """King on c1/g1 with no castling rights: a proxy, not castling history."""
    board = chess.Board(fen)
    # This configuration can also arise through ordinary king moves.
    wk_sq = board.king(chess.WHITE)
    can_castle = board.has_kingside_castling_rights(chess.WHITE) or \
                 board.has_queenside_castling_rights(chess.WHITE)
    if wk_sq in (chess.G1, chess.C1) and not can_castle:
        return 1
    return 0


def label_castled_black(fen):
    """King on c8/g8 with no castling rights: a proxy, not castling history."""
    board = chess.Board(fen)
    bk_sq = board.king(chess.BLACK)
    can_castle = board.has_kingside_castling_rights(chess.BLACK) or \
                 board.has_queenside_castling_rights(chess.BLACK)
    if bk_sq in (chess.G8, chess.C8) and not can_castle:
        return 1
    return 0


def label_material(fen):
    """White - Black material balance (scalar)."""
    board = chess.Board(fen)
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    val = 0
    for pt, v in pmap.items():
        val += len(board.pieces(pt, chess.WHITE)) * v
        val -= len(board.pieces(pt, chess.BLACK)) * v
    return float(val)


def label_piece_count(fen):
    board = chess.Board(fen)
    return float(len(board.piece_map()))


def label_white_isolated_pawns(fen):
    """Count white pawns with no friendly pawn on adjacent files."""
    board = chess.Board(fen)
    wp_files = set()
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        wp_files.add(chess.square_file(sq))
    isolated = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        f = chess.square_file(sq)
        if (f-1 not in wp_files) and (f+1 not in wp_files):
            isolated += 1
    return isolated


def label_open_files(fen):
    """Number of files with no pawns of either color."""
    board = chess.Board(fen)
    files_with_pawn = set()
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        files_with_pawn.add(chess.square_file(sq))
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        files_with_pawn.add(chess.square_file(sq))
    return 8 - len(files_with_pawn)


def label_turn(fen):
    """0 = white to move, 1 = black to move."""
    return 0 if chess.Board(fen).turn == chess.WHITE else 1


def label_in_check(fen):
    return 1 if chess.Board(fen).is_check() else 0


# ── Model loading ────────────────────────────────────────

def load_trained(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def load_untrained(ckpt):
    torch.manual_seed(42)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.eval()
    return model


def embed_batch(model, fens, batch=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_embs = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(device)
        with torch.no_grad():
            emb = model.get_embedding(x).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs)


# ── Probe runner ─────────────────────────────────────────

def run_probe(name, emb, labels, kind, trained_flag):
    """kind: 'cls' (classification) or 'reg' (regression)"""
    # Use the same double-precision design matrix for every checkpoint; this
    # avoids model-dependent low-precision conditioning in the linear probes.
    emb = np.asarray(emb, dtype=PROBE_DTYPE)
    X_train, X_val, y_train, y_val = train_test_split(
        emb, labels, test_size=0.2, random_state=42)
    if kind == 'cls':
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, n_jobs=-1)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        # Majority baseline
        baseline = max(np.mean(y_val == c) for c in np.unique(y_train))
        return acc, baseline
    else:
        clf = Ridge(solver=RIDGE_SOLVER)
        clf.fit(X_train, y_train)
        r2 = r2_score(y_val, clf.predict(X_val))
        baseline = 0.0  # R² of predicting mean is 0
        return r2, baseline


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    ap.add_argument("--json-output", default=None,
                    help="Optional machine-readable copy of unrounded probe scores")
    args = ap.parse_args()
    ckpt_path = args.ckpt
    trained, ckpt = load_trained(ckpt_path)
    untrained = load_untrained(ckpt)
    print(f"Loaded CNN: val_acc={ckpt.get('val_acc', 0):.2f}%")

    # Sample positions
    print(f"Loading {N_POSITIONS} positions...")
    fens = load_corpus("lichess_data/prev_move_2400.tsv", max_n=50000)
    np.random.seed(42)
    fens = [fens[i] for i in np.random.choice(len(fens), N_POSITIONS, replace=False)]

    print("Computing labels...")
    probes = [
        ("phase (0/1/2)",            label_phase,               'cls'),
        ("white castled",            label_castled_white,       'cls'),
        ("black castled",            label_castled_black,       'cls'),
        ("turn (white/black)",       label_turn,                'cls'),
        ("in check",                 label_in_check,            'cls'),
        ("material balance (W-B)",   label_material,            'reg'),
        ("piece count (2-32)",       label_piece_count,         'reg'),
        ("white isolated pawns",     label_white_isolated_pawns,'reg'),
        ("open files (0-8)",         label_open_files,          'reg'),
    ]
    labels_dict = {name: np.array([fn(f) for f in fens]) for name, fn, _ in probes}

    print("Computing embeddings (trained)...")
    emb_t = embed_batch(trained, fens)
    print("Computing embeddings (untrained)...")
    emb_u = embed_batch(untrained, fens)

    print("\n" + "="*78)
    print(f"  {'Concept':<30} {'Trained':<18} {'Untrained':<18} {'Baseline':<12}")
    print("="*78)

    metrics = []
    for name, _, kind in probes:
        y = labels_dict[name]
        t_score, base = run_probe(name, emb_t, y, kind, trained_flag=True)
        u_score, _ = run_probe(name, emb_u, y, kind, trained_flag=False)
        metrics.append({
            "concept": name,
            "kind": kind,
            "metric": "accuracy_fraction" if kind == "cls" else "r2",
            "trained": float(t_score),
            "untrained": float(u_score),
            "baseline": float(base),
            "lift": float(t_score - u_score),
            "validation_n": int(len(y) * 0.2),
        })

        if kind == 'cls':
            t_str = f"{t_score*100:5.1f}%"
            u_str = f"{u_score*100:5.1f}%"
            b_str = f"{base*100:5.1f}%"
        else:
            t_str = f"R²={t_score:+.3f}"
            u_str = f"R²={u_score:+.3f}"
            b_str = "R²=0.0"

        winner = " ✓" if (t_score - u_score) > 0.02 else ""
        print(f"  {name:<30} {t_str:<18} {u_str:<18} {b_str:<12}{winner}")

    print("="*78)
    print("✓ = trained > untrained by >2% (embedding learned something)")
    print("cls = classification accuracy | reg = R² score (1.0 = perfect)")
    if args.json_output:
        payload = {
            "checkpoint": str(Path(ckpt_path)),
            "checkpoint_val_acc": float(ckpt.get("val_acc", 0)),
            "n_positions": len(fens),
            "sample_seed": 42,
            "split_test_size": 0.2,
            "split_random_state": 42,
            "probe_protocol": {
                "classifier": "sklearn LogisticRegression",
                "classifier_max_iter": LOGREG_MAX_ITER,
                "classifier_n_jobs": -1,
                "regressor": "sklearn Ridge",
                "regressor_solver": RIDGE_SOLVER,
                "embedding_and_design_matrix_dtype": str(np.dtype(PROBE_DTYPE)),
            },
            "metrics": metrics,
        }
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved exact probe metrics to {output_path}")


if __name__ == "__main__":
    main()
