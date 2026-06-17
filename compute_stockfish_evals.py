#!/usr/bin/env python3
"""
compute_stockfish_evals.py — Batch-evaluate corpus positions with Stockfish
for the strategic-distance test. One-time offline script.

For each FEN in the corpus, queries Stockfish (UCI) at fixed depth and
records the centipawn score (from White's perspective). Output TSV:

    fen <tab> eval_cp

Usage:
    python3 compute_stockfish_evals.py --n 10000 --depth 12 \
        --stockfish /usr/games/stockfish \
        --output lichess_data/stockfish_evals.tsv

Depth 10-12 at ~50ms/pos gives 10K positions in ~10 minutes.
"""
import argparse
import csv
import os
import time

import chess
import chess.engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="lichess_data/prev_move_2400.tsv",
                    help="TSV with a 'fen' column")
    ap.add_argument("--n", type=int, default=10000,
                    help="Number of positions to evaluate")
    ap.add_argument("--depth", type=int, default=12,
                    help="Stockfish search depth (10-14 typical)")
    ap.add_argument("--stockfish", default="stockfish",
                    help="Path to Stockfish binary")
    ap.add_argument("--output", default="lichess_data/stockfish_evals.tsv")
    ap.add_argument("--threads", type=int, default=1,
                    help="Stockfish threads (1 is usually fine for batch eval)")
    args = ap.parse_args()

    # Load positions
    print(f"Loading {args.n} FENs from {args.corpus}...")
    fens = []
    with open(args.corpus) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            fens.append(row['fen'])
            if len(fens) >= args.n:
                break
    print(f"  loaded {len(fens)} FENs")

    # Open Stockfish
    print(f"Launching Stockfish at {args.stockfish} (depth={args.depth}, threads={args.threads})...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": args.threads})

    t0 = time.time()
    skipped = 0
    with open(args.output, "w") as out:
        out.write("fen\teval_cp\n")
        for i, fen in enumerate(fens):
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                score = info["score"].white()
                if score.is_mate():
                    # Mate: map to large cp (+/- 10000) depending on side/moves
                    mate_in = score.mate()
                    cp = 10000 if mate_in > 0 else -10000
                else:
                    cp = score.score()
                out.write(f"{fen}\t{cp}\n")
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [warn] skip {i}: {e}")
                continue

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(fens) - i - 1) / rate
                print(f"  {i+1}/{len(fens)} ({100*(i+1)/len(fens):.1f}%) "
                      f"rate={rate:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)

    engine.quit()
    elapsed = time.time() - t0
    print(f"\nDone: {len(fens) - skipped} positions evaluated in {elapsed:.0f}s")
    if skipped > 0:
        print(f"Skipped: {skipped} (errors)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
