#!/usr/bin/env python3
"""
download_lichess.py — Stream Lichess DB, filter by Elo, extract prev_move pairs.

Streams directly from Lichess zst-compressed PGN, filtering games where
BOTH players are >= min_elo. Avoids downloading huge files to disk.

Usage:
    python3 download_lichess.py --months 2023-06 2023-07 2023-08 --min-elo 2400 --output lichess_data/prev_move_2400.tsv
"""
import argparse
import chess
import chess.pgn
import io
import subprocess
import sys
import time


def stream_pgn_from_url(url):
    """Stream decompressed PGN lines from a zst URL via curl | pzstd."""
    proc = subprocess.Popen(
        f'curl -s "{url}" | pzstd -d',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')


def process_game(game, min_ply=6, max_ply=200):
    """Extract (fen, prev_move_uci) pairs."""
    board = game.board()
    moves = list(game.mainline_moves())
    if len(moves) < min_ply:
        return []
    pairs = []
    for i, move in enumerate(moves):
        board.push(move)
        if i >= min_ply - 1 and i < max_ply:
            pairs.append((board.fen(), move.uci()))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2023-06"],
                    help="YYYY-MM months to download")
    ap.add_argument("--min-elo", type=int, default=2400)
    ap.add_argument("--output", default="lichess_data/prev_move_2400.tsv")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--min-ply", type=int, default=6)
    args = ap.parse_args()

    base_url = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"

    total_games = 0
    total_pairs = 0
    skipped_elo = 0
    t0 = time.time()

    with open(args.output, "w") as out:
        out.write("fen\tprev_move\n")

        for month in args.months:
            url = base_url.format(month)
            print(f"\nStreaming {month} from {url}...")
            print(f"  Filtering: both players >= {args.min_elo} Elo")

            stream = stream_pgn_from_url(url)

            while True:
                try:
                    game = chess.pgn.read_game(stream)
                except Exception:
                    continue
                if game is None:
                    break

                # Elo filter
                try:
                    w_elo = int(game.headers.get("WhiteElo", "0"))
                    b_elo = int(game.headers.get("BlackElo", "0"))
                except (ValueError, TypeError):
                    continue

                if w_elo < args.min_elo or b_elo < args.min_elo:
                    skipped_elo += 1
                    if skipped_elo % 100000 == 0:
                        elapsed = time.time() - t0
                        print(f"  ... {skipped_elo} skipped (elo), "
                              f"{total_games} kept, {total_pairs} pairs, "
                              f"{elapsed:.0f}s", flush=True)
                    continue

                pairs = process_game(game, min_ply=args.min_ply)
                for fen, prev_move in pairs:
                    out.write(f"{fen}\t{prev_move}\n")
                    total_pairs += 1

                total_games += 1
                if total_games % 5000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {total_games} games, {total_pairs} pairs, "
                          f"{skipped_elo} skipped, {elapsed:.0f}s "
                          f"({total_games/elapsed:.0f} g/s)", flush=True)

                if args.max_games > 0 and total_games >= args.max_games:
                    break

            stream.close()

    elapsed = time.time() - t0
    print(f"\nDone: {total_games} games → {total_pairs} pairs in {elapsed:.0f}s")
    print(f"Skipped {skipped_elo} games by Elo filter")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
