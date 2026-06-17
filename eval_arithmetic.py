#!/usr/bin/env python3
"""
eval_arithmetic.py — Chess2Vec position arithmetic tests.

Tests midpoint interpolation and color symmetry in the embedding space.

Usage:
    python3 eval_arithmetic.py
"""
import torch
import numpy as np
import csv
import os
from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


def load_model(arch, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('val_acc', 0)


def load_corpus(tsv_path, max_n=50000):
    """Load positions + pre-compute embeddings."""
    fens = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            fens.append(row['fen'])
            if i >= max_n:
                break
    return fens


def compute_corpus_embeddings(model, fens, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_embs = []
    for i in range(0, len(fens), batch_size):
        batch = fens[i:i+batch_size]
        x = torch.stack([torch.from_numpy(fen_to_bitboards(f)) for f in batch]).to(device)
        with torch.no_grad():
            emb = model.get_embedding(x).cpu().numpy()
        all_embs.append(emb)
    return np.vstack(all_embs)


def get_emb(model, fen):
    device = next(model.parameters()).device
    x = torch.from_numpy(fen_to_bitboards(fen)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.get_embedding(x).cpu().numpy()[0]


def nearest(target_emb, corpus_embs, corpus_fens, top_k=5):
    dists = np.linalg.norm(corpus_embs - target_emb, axis=1)
    idxs = np.argsort(dists)[:top_k]
    return [(corpus_fens[i], dists[i]) for i in idxs]


def run_arithmetic_tests(model, arch_name, corpus_fens, corpus_embs):
    print(f"\n{'='*70}")
    print(f"  CHESS2VEC ARITHMETIC: {arch_name}")
    print(f"{'='*70}")

    def midpoint_search(fen_a, fen_b, name_a, name_b):
        emb_a = get_emb(model, fen_a)
        emb_b = get_emb(model, fen_b)
        mid = (emb_a + emb_b) / 2
        results = nearest(mid, corpus_embs, corpus_fens, top_k=3)
        d_ab = np.linalg.norm(emb_a - emb_b)
        print(f"\n  {name_a} + {name_b} (dist={d_ab:.1f})")
        for fen, dist in results:
            print(f"    {dist:>5.1f}  {fen}")

    # ── Section 1: Strategic midpoints ────────────────────────
    print("\n  --- STRATEGIC MIDPOINTS ---")

    midpoint_search(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "Sicilian", "Queen's Gambit"
    )

    midpoint_search(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/5k2/8/8/8/8/4RK2/8 w - - 0 1",
        "Italian Game", "KR endgame"
    )

    midpoint_search(
        "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
        "rnbqkb1r/ppp1pppp/3p1n2/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
        "King's Gambit", "London System"
    )

    midpoint_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/4k3/8/8/8/4PK2/8 w - - 0 1",
        "Starting position", "KP endgame"
    )

    # ── Section 2: Color symmetry / material imbalance ────────
    print("\n\n  --- COLOR SYMMETRY / MATERIAL TESTS ---")

    # White winning (queen + 2 pawns up)
    white_wins = "8/5k2/8/8/8/8/2PPQK2/8 w - - 0 1"
    # Black winning (queen + 2 pawns up) — mirror
    black_wins = "8/2ppqk2/8/8/8/8/5K2/8 b - - 0 1"
    # Equal-ish endgame
    equal_end = "8/5k2/8/8/8/8/4PK2/8 w - - 0 1"

    emb_ww = get_emb(model, white_wins)
    emb_bw = get_emb(model, black_wins)
    emb_eq = get_emb(model, equal_end)

    d_ww_bw = np.linalg.norm(emb_ww - emb_bw)
    d_ww_eq = np.linalg.norm(emb_ww - emb_eq)
    d_bw_eq = np.linalg.norm(emb_bw - emb_eq)

    print(f"\n  White winning (Q+2P) vs Black winning (Q+2P):")
    print(f"    Distance: {d_ww_bw:.1f}")
    print(f"  White winning vs Equal KP end:")
    print(f"    Distance: {d_ww_eq:.1f}")
    print(f"  Black winning vs Equal KP end:")
    print(f"    Distance: {d_bw_eq:.1f}")

    mid_wb = (emb_ww + emb_bw) / 2
    results = nearest(mid_wb, corpus_embs, corpus_fens, top_k=5)
    print(f"\n  Midpoint(White winning + Black winning) = ???")
    print(f"  (hypothesis: should find roughly equal positions)")
    for fen, dist in results:
        print(f"    {dist:>5.1f}  {fen}")

    # ── Section 3: Piece addition/removal ─────────────────────
    print("\n\n  --- PIECE ARITHMETIC ---")

    # KR endgame + extra knight = ???
    kr_end = "8/5k2/8/8/8/8/4RK2/8 w - - 0 1"
    krn_end = "8/5k2/8/8/8/4N3/4RK2/8 w - - 0 1"

    emb_kr = get_emb(model, kr_end)
    emb_krn = get_emb(model, krn_end)
    diff = emb_krn - emb_kr  # "what does adding a knight do?"

    # Apply "add knight" to KP endgame
    kp_end = "8/5k2/8/8/8/8/4PK2/8 w - - 0 1"
    emb_kp = get_emb(model, kp_end)
    kpn_predicted = emb_kp + diff  # should find KPN endgame

    results = nearest(kpn_predicted, corpus_embs, corpus_fens, top_k=5)
    print(f"  KR + knight_vector = KRN endgame (control)")
    print(f"  KP + knight_vector = ??? (should find KPN-like)")
    for fen, dist in results:
        print(f"    {dist:>5.1f}  {fen}")


def run_sum_tests(model, arch_name, corpus_fens, corpus_embs):
    """Test: embed(A) + embed(B) → what position?"""
    print(f"\n{'='*70}")
    print(f"  EMBEDDING SUM TESTS: {arch_name}")
    print(f"{'='*70}")

    def sum_search(fen_a, fen_b, name_a, name_b):
        emb_a = get_emb(model, fen_a)
        emb_b = get_emb(model, fen_b)
        summed = emb_a + emb_b
        results = nearest(summed, corpus_embs, corpus_fens, top_k=5)
        print(f"\n  {name_a} + {name_b} =")
        for fen, dist in results:
            pieces = sum(1 for c in fen.split()[0] if c.isalpha())
            print(f"    {dist:>5.1f}  ({pieces}p) {fen}")

    # e4 position + e5 position = e4 e5 position? or Nf3?
    sum_search(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "after 1.e4 (white played)", "after 1...e5 (black played)"
    )

    # d4 + d5 = d4 d5? or QG?
    sum_search(
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "after 1.d4", "after 1...d5"
    )

    # Nf3 + d5 = Nf3 d5 (Reti)?
    sum_search(
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "after 1.Nf3", "after 1...d5"
    )

    # Sicilian (e4 c5) + d4 = Morra Gambit (e4 c5 d4)?
    sum_search(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "Sicilian (after e4 c5)", "after 1.d4"
    )

    # Italian + endgame pieces off = Italian endgame?
    sum_search(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/5k2/8/8/8/8/4PK2/8 w - - 0 1",
        "Italian middlegame", "KP endgame"
    )


def main():
    tsv = "lichess_data/prev_move_2400.tsv"
    print("Loading corpus...")
    corpus_fens = load_corpus(tsv, max_n=50000)

    for arch in ["mlp", "lstm", "cnn", "attention", "dual_mlp", "cnn_dual"]:
        path = f"models/prev_move_{arch}.pt"
        if not os.path.exists(path):
            continue
        model, val_acc = load_model(arch, path)
        print(f"\nComputing corpus embeddings for {arch}...")
        corpus_embs = compute_corpus_embeddings(model, corpus_fens)
        name = f"{arch} (val={val_acc:.1f}%)"
        run_arithmetic_tests(model, name, corpus_fens, corpus_embs)
        run_sum_tests(model, name, corpus_fens, corpus_embs)


if __name__ == "__main__":
    main()
