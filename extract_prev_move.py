#!/usr/bin/env python3
"""
extract_prev_move.py — Extract (position, previous_move) pairs from PGN.

For each position in each game (after move 2), outputs:
    FEN <tab> previous_move_UCI

This is the training data for previous-move prediction pre-training.

Usage:
    python3 extract_prev_move.py lichess_data/lichess_2013_01.pgn -o prev_move_data.tsv --max-games 100000
"""
import argparse
import chess
import chess.pgn
import io
import sys
import time


def process_game(game, min_ply=4, max_ply=200):
    """Extract (fen, prev_move_uci) pairs from a single game.
    Skip first min_ply positions (opening too uniform).
    Skip if game has fewer than min_ply moves."""
    board = game.board()
    moves = list(game.mainline_moves())

    if len(moves) < min_ply:
        return []

    pairs = []
    for i, move in enumerate(moves):
        board.push(move)
        if i >= min_ply - 1 and i < max_ply:
            # Position after this move; previous move = move
            fen = board.fen()
            prev_uci = move.uci()
            pairs.append((fen, prev_uci))

    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", help="PGN file to process")
    ap.add_argument("-o", "--output", default="prev_move_data.tsv")
    ap.add_argument("--max-games", type=int, default=0,
                    help="max games to process (0=all)")
    ap.add_argument("--min-ply", type=int, default=4,
                    help="skip positions before this ply (default 4)")
    ap.add_argument("--min-elo", type=int, default=1200,
                    help="skip games where either player below this Elo")
    args = ap.parse_args()

    t0 = time.time()
    n_games = 0
    n_pairs = 0
    n_skipped_elo = 0

    with open(args.pgn) as pgn_file, open(args.output, "w") as out:
        out.write("fen\tprev_move\n")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Elo filter
            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
                b_elo = int(game.headers.get("BlackElo", "0"))
                if w_elo < args.min_elo or b_elo < args.min_elo:
                    n_skipped_elo += 1
                    continue
            except (ValueError, TypeError):
                pass

            pairs = process_game(game, min_ply=args.min_ply)
            for fen, prev_move in pairs:
                out.write(f"{fen}\t{prev_move}\n")
                n_pairs += 1

            n_games += 1
            if n_games % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  {n_games} games, {n_pairs} pairs, "
                      f"{n_skipped_elo} skipped (elo), "
                      f"{elapsed:.0f}s ({n_games/elapsed:.0f} games/s)",
                      flush=True)

            if args.max_games > 0 and n_games >= args.max_games:
                break

    elapsed = time.time() - t0
    print(f"\nDone: {n_games} games → {n_pairs} pairs in {elapsed:.0f}s")
    print(f"Skipped {n_skipped_elo} games by Elo filter")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
