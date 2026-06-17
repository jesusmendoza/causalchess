#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_probes_strategic.py — extended linear probes with STRATEGIC concepts
+ per-phase breakdown.

Strategic concepts tested:
  - Castling (W, B)
  - Turn
  - Development (count of minor pieces off starting squares)
  - Center control (attacks on d4/d5/e4/e5)
  - Space (advanced pawns)
  - King safety (pawn shield)
  - Bishop pair
  - Doubled pawns
  - Passed pawns
  - Queen present
  - Mobility (legal moves count)
  - Material imbalance
  - Pawn structure entropy

Breakdown: all probes run on (all), opening-only, middlegame-only, endgame-only.
Shows WHICH concepts the embedding learns AT WHICH stage.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import chess
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from prev_move_models import create_model
from prev_move_models import fen_to_bitboards
from eval_arithmetic import load_corpus


N_POSITIONS = 20000


# ── Helpers ───────────────────────────────────────────────

def material_total(board):
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for pt, v in pmap.items():
        total += (len(board.pieces(pt, chess.WHITE))
                  + len(board.pieces(pt, chess.BLACK))) * v
    return total


def phase_idx(fen):
    tot = material_total(chess.Board(fen))
    if tot >= 70: return 0   # opening
    elif tot >= 40: return 1 # middlegame
    else: return 2           # endgame


# ── Concept label extractors ─────────────────────────────

STARTING_MINORS_W = {chess.B1, chess.G1, chess.C1, chess.F1}  # Nb1,Ng1,Bc1,Bf1
STARTING_MINORS_B = {chess.B8, chess.G8, chess.C8, chess.F8}


def label_phase(fen):
    return phase_idx(fen)


def label_castled_white(fen):
    board = chess.Board(fen)
    wk = board.king(chess.WHITE)
    can = (board.has_kingside_castling_rights(chess.WHITE)
           or board.has_queenside_castling_rights(chess.WHITE))
    return 1 if (wk in (chess.G1, chess.C1) and not can) else 0


def label_castled_black(fen):
    board = chess.Board(fen)
    bk = board.king(chess.BLACK)
    can = (board.has_kingside_castling_rights(chess.BLACK)
           or board.has_queenside_castling_rights(chess.BLACK))
    return 1 if (bk in (chess.G8, chess.C8) and not can) else 0


def label_turn(fen):
    return 0 if chess.Board(fen).turn == chess.WHITE else 1


def label_in_check(fen):
    return 1 if chess.Board(fen).is_check() else 0


def label_material_balance(fen):
    board = chess.Board(fen)
    pmap = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    v = 0
    for pt, val in pmap.items():
        v += len(board.pieces(pt, chess.WHITE)) * val
        v -= len(board.pieces(pt, chess.BLACK)) * val
    return float(v)


def label_piece_count(fen):
    return float(len(chess.Board(fen).piece_map()))


def label_open_files(fen):
    board = chess.Board(fen)
    files_with_pawn = set()
    for sq in list(board.pieces(chess.PAWN, chess.WHITE)) + \
              list(board.pieces(chess.PAWN, chess.BLACK)):
        files_with_pawn.add(chess.square_file(sq))
    return 8 - len(files_with_pawn)


def label_white_isolated_pawns(fen):
    board = chess.Board(fen)
    wp_files = set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE))
    iso = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        f = chess.square_file(sq)
        if (f-1 not in wp_files) and (f+1 not in wp_files):
            iso += 1
    return iso


# ── NEW: strategic concepts ──────────────────────────────

def label_white_developed(fen):
    """Count of white minor pieces OFF starting squares."""
    board = chess.Board(fen)
    dev = 0
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        if sq not in STARTING_MINORS_W: dev += 1
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        if sq not in STARTING_MINORS_W: dev += 1
    return dev


def label_black_developed(fen):
    board = chess.Board(fen)
    dev = 0
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        if sq not in STARTING_MINORS_B: dev += 1
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        if sq not in STARTING_MINORS_B: dev += 1
    return dev


def label_white_center_control(fen):
    """Count of white pieces attacking d4, d5, e4, e5."""
    board = chess.Board(fen)
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    count = 0
    for sq in center:
        count += len(board.attackers(chess.WHITE, sq))
    return count


def label_black_center_control(fen):
    board = chess.Board(fen)
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    count = 0
    for sq in center:
        count += len(board.attackers(chess.BLACK, sq))
    return count


def label_white_space(fen):
    """White pawns on ranks 4-7 (advanced)."""
    board = chess.Board(fen)
    count = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_rank(sq) >= 3:  # rank 4 = index 3
            count += 1
    return count


def label_black_space(fen):
    board = chess.Board(fen)
    count = 0
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if chess.square_rank(sq) <= 4:  # rank 5 = index 4 (black pawns start rank 7)
            count += 1
    return count


def label_white_king_shield(fen):
    """Count of white pawns adjacent to white king (within 1 square)."""
    board = chess.Board(fen)
    wk = board.king(chess.WHITE)
    if wk is None: return 0
    kf = chess.square_file(wk); kr = chess.square_rank(wk)
    shield = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if abs(chess.square_file(sq) - kf) <= 1 and abs(chess.square_rank(sq) - kr) <= 1:
            shield += 1
    return shield


def label_black_king_shield(fen):
    board = chess.Board(fen)
    bk = board.king(chess.BLACK)
    if bk is None: return 0
    kf = chess.square_file(bk); kr = chess.square_rank(bk)
    shield = 0
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if abs(chess.square_file(sq) - kf) <= 1 and abs(chess.square_rank(sq) - kr) <= 1:
            shield += 1
    return shield


def label_white_bishop_pair(fen):
    board = chess.Board(fen)
    return 1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0


def label_black_bishop_pair(fen):
    board = chess.Board(fen)
    return 1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0


def label_white_doubled_pawns(fen):
    """Count of white pawn 'doublings' (2+ on same file → N-1 extras)."""
    board = chess.Board(fen)
    files = [0]*8
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        files[chess.square_file(sq)] += 1
    return sum(max(f-1, 0) for f in files)


def label_black_doubled_pawns(fen):
    board = chess.Board(fen)
    files = [0]*8
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        files[chess.square_file(sq)] += 1
    return sum(max(f-1, 0) for f in files)


def label_queens_on_board(fen):
    """0=no Q, 1=one Q, 2=both Qs."""
    board = chess.Board(fen)
    return (len(board.pieces(chess.QUEEN, chess.WHITE))
            + len(board.pieces(chess.QUEEN, chess.BLACK)))


def label_white_passed_pawns(fen):
    """White pawn with no enemy pawns ahead on same or adjacent files."""
    board = chess.Board(fen)
    bp_by_file = {}
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        f = chess.square_file(sq); r = chess.square_rank(sq)
        bp_by_file.setdefault(f, []).append(r)

    passed = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        f = chess.square_file(sq); r = chess.square_rank(sq)
        is_passed = True
        for nf in (f-1, f, f+1):
            if nf < 0 or nf > 7: continue
            for br in bp_by_file.get(nf, []):
                if br > r:  # black pawn ahead
                    is_passed = False; break
            if not is_passed: break
        if is_passed: passed += 1
    return passed


def label_mobility(fen):
    """Number of legal moves (proxy for mobility/activity)."""
    return chess.Board(fen).legal_moves.count()


def label_white_king_file(fen):
    """King file (0=a, 7=h). Useful for detecting sides of the board."""
    board = chess.Board(fen)
    wk = board.king(chess.WHITE)
    return chess.square_file(wk) if wk is not None else 4


def label_black_king_file(fen):
    board = chess.Board(fen)
    bk = board.king(chess.BLACK)
    return chess.square_file(bk) if bk is not None else 4


# ── Model loading ────────────────────────────────────────

def load_trained(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def load_untrained(ckpt):
    torch.manual_seed(42)
    m = create_model(ckpt['arch'], ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    m.eval()
    return m


def embed_batch(model, fens, batch=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    for i in range(0, len(fens), batch):
        chunk = fens[i:i+batch]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in chunk]).to(device)
        with torch.no_grad():
            out.append(model.get_embedding(x).cpu().numpy())
    return np.vstack(out)


def run_probe(emb, labels, kind):
    X_tr, X_val, y_tr, y_val = train_test_split(
        emb, labels, test_size=0.2, random_state=42)
    if kind == 'cls':
        if len(np.unique(y_tr)) < 2:
            return float('nan'), float('nan')
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr)
        return accuracy_score(y_val, clf.predict(X_val)), max(np.mean(y_val==c) for c in np.unique(y_tr))
    else:
        clf = Ridge()
        clf.fit(X_tr, y_tr)
        return r2_score(y_val, clf.predict(X_val)), 0.0


PROBES = [
    # name, fn, kind
    ("phase",                   label_phase,                  'cls'),
    ("turn",                    label_turn,                   'cls'),
    ("in check",                label_in_check,               'cls'),
    ("castled W",               label_castled_white,          'cls'),
    ("castled B",               label_castled_black,          'cls'),
    ("queens on board",         label_queens_on_board,        'reg'),
    ("bishop pair W",           label_white_bishop_pair,      'cls'),
    ("bishop pair B",           label_black_bishop_pair,      'cls'),
    ("king file W",             label_white_king_file,        'reg'),
    ("king file B",             label_black_king_file,        'reg'),
    ("material balance",        label_material_balance,       'reg'),
    ("piece count",             label_piece_count,            'reg'),
    ("open files",              label_open_files,             'reg'),
    ("isolated pawns W",        label_white_isolated_pawns,   'reg'),
    ("doubled pawns W",         label_white_doubled_pawns,    'reg'),
    ("doubled pawns B",         label_black_doubled_pawns,    'reg'),
    ("passed pawns W",          label_white_passed_pawns,     'reg'),
    ("development W",           label_white_developed,        'reg'),
    ("development B",           label_black_developed,        'reg'),
    ("center control W",        label_white_center_control,   'reg'),
    ("center control B",        label_black_center_control,   'reg'),
    ("space W",                 label_white_space,            'reg'),
    ("space B",                 label_black_space,            'reg'),
    ("king shield W",           label_white_king_shield,      'reg'),
    ("king shield B",           label_black_king_shield,      'reg'),
    ("mobility",                label_mobility,               'reg'),
]


def run_all(emb, labels_dict, subset=None):
    results = {}
    for name, _, kind in PROBES:
        y = labels_dict[name]
        if subset is not None:
            y = y[subset]
            emb_use = emb[subset]
        else:
            emb_use = emb
        t_score, base = run_probe(emb_use, y, kind)
        results[name] = (t_score, base, kind)
    return results


def fmt(score, kind):
    if np.isnan(score): return "   -  "
    if kind == 'cls': return f"{score*100:5.1f}%"
    else:             return f"R²={score:+.2f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn.pt")
    args = ap.parse_args()
    ckpt_path = args.ckpt
    trained, ckpt = load_trained(ckpt_path)
    untrained = load_untrained(ckpt)
    print(f"Loaded CNN val_acc={ckpt.get('val_acc',0):.2f}%")

    # Sample positions
    print(f"Loading {N_POSITIONS} positions...")
    all_fens = load_corpus("lichess_data/prev_move_2400.tsv", max_n=50000)
    np.random.seed(42)
    fens = [all_fens[i] for i in np.random.choice(len(all_fens), N_POSITIONS, replace=False)]

    print("Computing labels...")
    labels_dict = {name: np.array([fn(f) for f in fens]) for name, fn, _ in PROBES}
    phases = labels_dict["phase"]

    print("Computing embeddings...")
    emb_t = embed_batch(trained, fens)
    emb_u = embed_batch(untrained, fens)

    # Phase masks
    masks = {
        "ALL":        np.arange(len(fens)),
        "opening":    np.where(phases == 0)[0],
        "middlegame": np.where(phases == 1)[0],
        "endgame":    np.where(phases == 2)[0],
    }
    for k, m in masks.items():
        print(f"  {k}: {len(m)} positions")

    # Run probes per phase
    print("\nRunning probes...")
    res_t = {k: run_all(emb_t, labels_dict, subset=m) for k, m in masks.items()}
    res_u = {k: run_all(emb_u, labels_dict, subset=m) for k, m in masks.items()}

    # Build table: rows=probe, cols = ALL_T, ALL_U, OP_T, OP_U, MID_T, MID_U, END_T, END_U
    hdr = f"  {'Concept':<22}"
    phase_keys = ["ALL", "opening", "middlegame", "endgame"]
    for k in phase_keys:
        hdr += f"  {k+'_T':>8}  {k+'_U':>8}"
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))

    for name, _, kind in PROBES:
        row = f"  {name:<22}"
        for k in phase_keys:
            t_score, _, _ = res_t[k][name]
            u_score, _, _ = res_u[k][name]
            row += f"  {fmt(t_score, kind):>8}  {fmt(u_score, kind):>8}"
        # Mark winners (trained > untrained by >2% cls, >0.05 R²)
        print(row)

    print("="*len(hdr))

    # Summary: where does trained >> untrained?
    print("\n### LIFT (trained - untrained) across phases ###\n")
    print(f"  {'Concept':<22}  {'kind':<4}", end="")
    for k in phase_keys:
        print(f"  {k:>10}", end="")
    print()
    print("-"*90)
    for name, _, kind in PROBES:
        row = f"  {name:<22}  {kind:<4}"
        for k in phase_keys:
            t, _, _ = res_t[k][name]
            u, _, _ = res_u[k][name]
            if np.isnan(t) or np.isnan(u):
                lift = float('nan')
            else:
                lift = (t - u) * (100 if kind == 'cls' else 1)
            row += f"  {lift:>+10.2f}"
        print(row)


if __name__ == "__main__":
    main()
