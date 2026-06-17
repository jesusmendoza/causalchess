#!/usr/bin/env python3
"""
eval_embeddings.py — Benchmark embedding quality across all trained models.

Tests strategic clustering: positions that are strategically similar should
have close embeddings, different ones should be far.

Usage:
    python3 eval_embeddings.py
"""
import torch
import numpy as np
import os
from prev_move_models import create_model
from prev_move_models import fen_to_bitboards


# ── Test positions ───────────────────────────────────────────────

POSITIONS = {
    # Transpositions (distance MUST be 0)
    "d4_Nf6_c4":      "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "c4_Nf6_d4":      "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",

    # First moves (strategic commitment)
    "e4":              "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "d4":              "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "Nf3":             "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
    "c4":              "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "g3":              "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",

    # Symmetric centers (strategically similar feel)
    "e4_e5":           "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "d4_d5":           "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",

    # e4 family openings (should cluster together)
    "sicilian":        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "french":          "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "italian_5":       "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "spanish_5":       "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",

    # d4 family openings (should cluster together)
    "queens_gambit":   "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "kings_indian":    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "london":          "rnbqkb1r/ppp1pppp/3p1n2/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    "nimzo":           "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",

    # Aggressive vs quiet (should be far)
    "fried_liver":     "r1bqkb1r/ppp2ppp/2n2n2/3Np3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5",
    "closed_catalan":  "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq - 4 5",

    # Endgames (should cluster together, far from openings/middlegames)
    "philidor":        "8/8/4k3/R7/5r2/4K3/8/8 w - - 0 1",
    "lucena":          "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
    "opp_bishops":     "8/8/4k3/8/2B5/8/4K3/5b2 w - - 0 1",
    "kp_end":          "8/5k2/8/8/8/8/4PK2/8 w - - 0 1",
    "rook_end":        "8/5k2/8/8/8/8/4RK2/8 w - - 0 1",
    "queen_end":       "8/5k2/8/8/8/8/3QK3/8 w - - 0 1",

    # Middlegame contrasts
    "open_mid":        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "closed_mid":      "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R b KQkq - 0 7",

    # Reference
    "startpos":        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
}


# ── Test suite ───────────────────────────────────────────────────

TESTS = [
    ("TRANSPOSITION (must be 0)", [
        ("d4_Nf6_c4", "c4_Nf6_d4", "same position different move order"),
    ]),
    ("SAME FAMILY — e4 openings (should be small)", [
        ("sicilian", "french", "Sicilian vs French"),
        ("italian_5", "spanish_5", "Italian vs Spanish"),
        ("sicilian", "italian_5", "Sicilian vs Italian"),
        ("e4", "e4_e5", "1.e4 vs 1.e4 e5"),
    ]),
    ("SAME FAMILY — d4 openings (should be small)", [
        ("queens_gambit", "kings_indian", "QG vs King's Indian"),
        ("queens_gambit", "london", "QG vs London"),
        ("queens_gambit", "nimzo", "QG vs Nimzo"),
        ("london", "nimzo", "London vs Nimzo"),
    ]),
    ("SAME FAMILY — endgames (should be small)", [
        ("philidor", "lucena", "Philidor vs Lucena (rook ends)"),
        ("kp_end", "rook_end", "KP vs KR endgame"),
        ("opp_bishops", "kp_end", "Opp bishops vs KP"),
        ("rook_end", "queen_end", "KR vs KQ endgame"),
    ]),
    ("CROSS FAMILY — e4 vs d4 (should be moderate-large)", [
        ("sicilian", "queens_gambit", "Sicilian vs QG"),
        ("italian_5", "london", "Italian vs London"),
        ("french", "kings_indian", "French vs King's Indian"),
        ("e4_e5", "d4_d5", "1.e4 e5 vs 1.d4 d5"),
    ]),
    ("CROSS PHASE — opening/mid vs endgame (should be very large)", [
        ("italian_5", "philidor", "Italian mid vs Philidor end"),
        ("sicilian", "kp_end", "Sicilian vs KP endgame"),
        ("startpos", "rook_end", "Startpos vs Rook endgame"),
        ("london", "opp_bishops", "London vs Opp bishops end"),
    ]),
    ("CHARACTER — aggressive vs quiet (should be large)", [
        ("fried_liver", "closed_catalan", "Fried Liver vs Catalan"),
        ("fried_liver", "london", "Fried Liver vs London"),
        ("open_mid", "closed_mid", "Open vs Closed middlegame"),
    ]),
    ("FIRST MOVES (strategic commitment)", [
        ("e4", "d4", "e4 vs d4"),
        ("e4", "Nf3", "e4 vs Nf3"),
        ("d4", "Nf3", "d4 vs Nf3"),
        ("e4", "g3", "e4 vs g3"),
        ("d4", "c4", "d4 vs c4"),
        ("Nf3", "c4", "Nf3 vs c4"),
    ]),
]


def run_benchmark(model, arch_name):
    model.eval()
    embs = {}
    with torch.no_grad():
        for name, fen in POSITIONS.items():
            x = torch.from_numpy(fen_to_bitboards(fen)).unsqueeze(0)
            embs[name] = model.get_embedding(x).numpy()[0]

    print(f"\n{'='*70}")
    print(f"  EMBEDDING BENCHMARK: {arch_name}")
    print(f"{'='*70}")

    category_avgs = []
    for category, pairs in TESTS:
        dists = []
        print(f"\n  {category}:")
        for a, b, desc in pairs:
            d = np.linalg.norm(embs[a] - embs[b])
            dists.append(d)
            print(f"    {d:>6.1f}  {desc}")
        avg = np.mean(dists)
        category_avgs.append((category.split("(")[0].strip(), avg))
        print(f"    {'avg':>6} = {avg:.1f}")

    print(f"\n  SUMMARY ({arch_name}):")
    print(f"  {'-'*50}")
    for cat, avg in category_avgs:
        bar_len = int(avg / 1.0)
        bar = "#" * min(bar_len, 50)
        print(f"    {avg:>5.1f}  {cat[:30]:<30} |{bar}")

    return category_avgs


def main():
    models_dir = "models"
    all_results = {}

    for arch in ["mlp", "attention", "lstm", "cnn", "dual_mlp", "cnn_dual"]:
        path = os.path.join(models_dir, f"prev_move_{arch}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        a = ckpt.get('arch', arch)
        model = create_model(a, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
        model.load_state_dict(ckpt['model'])
        avgs = run_benchmark(model, f"{arch} (val_acc={ckpt.get('val_acc', '?'):.1f}%)")
        all_results[arch] = avgs

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON ACROSS ARCHITECTURES")
        print(f"{'='*70}")
        cats = [c for c, _ in list(all_results.values())[0]]
        header = f"  {'Category':<35}"
        for arch in all_results:
            header += f" {arch:>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        for i, cat in enumerate(cats):
            row = f"  {cat[:35]:<35}"
            for arch in all_results:
                val = all_results[arch][i][1]
                row += f" {val:>10.1f}"
            print(row)


if __name__ == "__main__":
    main()
