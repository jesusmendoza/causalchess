#!/usr/bin/env python3
"""
extract_elo_dataset.py — Stream lichess PGNs and build a per-player
dataset for Elo prediction.

For each player with at least `--games-per-player` games in a month,
we collect N random positions across their games and attach their Elo
at the time of play (using WhiteElo/BlackElo headers).

Output (directory default `data_elo/`):
  data_elo/chess_x.bin         N × 12 uint64   (bitboards per position)
  data_elo/chess_player_id.bin N × uint32      (player id per position)
  data_elo/chess_elo.bin       N × float32     (Elo at time of position)
  data_elo/chess_meta.txt      n_samples, n_players
  data_elo/player_names.txt    player_id → username mapping

The prediction task: given the concat of K position-embeddings for a
player, predict their mean Elo. Real ground truth.

Usage:
    python3 extract_elo_dataset.py --months 2023-06 --min-games 20 \\
        --positions-per-player 40 --outdir data_elo
"""
import argparse
import chess
import chess.pgn
import io
import subprocess
import os
import time
from collections import defaultdict
import numpy as np


def fen_to_bitboards(fen: str):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    boards = [0] * 12
    sq = 56
    for ch in fen.split()[0]:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            idx = piece_map.get(ch)
            if idx is not None:
                boards[idx] |= (1 << sq)
            sq += 1
    return boards


def stream_pgn_from_url(url):
    proc = subprocess.Popen(
        f'curl -s "{url}" | pzstd -d',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2023-06"])
    ap.add_argument("--outdir", default="data_elo")
    ap.add_argument("--min-games", type=int, default=20,
                    help="Only keep players with at least this many games collected.")
    ap.add_argument("--positions-per-player", type=int, default=40,
                    help="Random positions to sample per player (across their games).")
    ap.add_argument("--min-ply", type=int, default=8,
                    help="Skip first N plies (opening too uniform).")
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--min-elo", type=int, default=800,
                    help="Skip players below this Elo (too noisy).")
    ap.add_argument("--max-elo", type=int, default=3000)
    ap.add_argument("--max-games-per-month", type=int, default=1_000_000,
                    help="Safety cap on games processed per month.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # player_id -> list of (bitboards, elo_at_time)
    player_positions = defaultdict(list)
    player_elo_history = defaultdict(list)  # list of elos per game

    base_url = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
    total_games = 0

    t0 = time.time()
    for month in args.months:
        url = base_url.format(month)
        print(f"\nStreaming {month}...", flush=True)
        stream = stream_pgn_from_url(url)
        n_month = 0
        while True:
            try:
                game = chess.pgn.read_game(stream)
            except Exception:
                continue
            if game is None:
                break

            headers = game.headers
            w_name = headers.get("White", "?")
            b_name = headers.get("Black", "?")
            try:
                w_elo = int(headers.get("WhiteElo", "0"))
                b_elo = int(headers.get("BlackElo", "0"))
            except (ValueError, TypeError):
                continue
            if w_elo < args.min_elo or b_elo < args.min_elo:
                continue
            if w_elo > args.max_elo or b_elo > args.max_elo:
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < args.min_ply:
                continue

            # Random sample of a few positions from this game
            pos_plies = list(range(args.min_ply - 1, min(len(moves), args.max_ply)))
            if not pos_plies:
                continue
            n_take = min(3, len(pos_plies))  # up to 3 positions per game
            take_plies = rng.choice(pos_plies, size=n_take, replace=False)

            # Play through game, collect sampled positions
            for i, move in enumerate(moves):
                board.push(move)
                if i in take_plies:
                    bb = fen_to_bitboards(board.fen())
                    # The position is "after this move" so the side to move
                    # next is the opponent. Associate both players' Elo
                    # (we'll keep the side-to-move's Elo as label).
                    side_to_move_is_white = (i % 2 == 1)  # after move, other side moves
                    elo = w_elo if side_to_move_is_white else b_elo
                    player = w_name if side_to_move_is_white else b_name
                    player_positions[player].append((bb, elo))

            player_elo_history[w_name].append(w_elo)
            player_elo_history[b_name].append(b_elo)

            total_games += 1
            n_month += 1
            if total_games % 5000 == 0:
                n_players = len(player_positions)
                n_positions = sum(len(v) for v in player_positions.values())
                elapsed = time.time() - t0
                print(f"  {total_games} games, {n_players} players, "
                      f"{n_positions} positions, {elapsed:.0f}s", flush=True)

            if n_month >= args.max_games_per_month:
                break

        stream.close()

    # Filter players with enough games
    valid_players = [p for p, games in player_elo_history.items()
                     if len(games) >= args.min_games]
    print(f"\nPlayers: {len(valid_players)}/{len(player_positions)} "
          f"have ≥{args.min_games} games")

    # Sample up to --positions-per-player from each valid player
    x_list, p_list, e_list = [], [], []
    player_name_list = []
    for pid, pname in enumerate(valid_players):
        positions = player_positions[pname]
        if len(positions) == 0:
            continue
        n_take = min(args.positions_per_player, len(positions))
        idxs = rng.choice(len(positions), size=n_take, replace=False)
        for i in idxs:
            bb, elo = positions[i]
            x_list.append(bb)
            p_list.append(pid)
            e_list.append(elo)
        player_name_list.append(pname)

    n = len(x_list)
    print(f"\n{n} positions from {len(player_name_list)} players written")

    np.array(x_list, dtype=np.uint64).tofile(f"{args.outdir}/chess_x.bin")
    np.array(p_list, dtype=np.uint32).tofile(f"{args.outdir}/chess_player_id.bin")
    np.array(e_list, dtype=np.float32).tofile(f"{args.outdir}/chess_elo.bin")

    with open(f"{args.outdir}/chess_meta.txt", "w") as f:
        f.write(f"n_samples={n}\n")
        f.write(f"n_players={len(player_name_list)}\n")
        f.write(f"positions_per_player={args.positions_per_player}\n")
        f.write(f"min_games={args.min_games}\n")

    with open(f"{args.outdir}/player_names.txt", "w") as f:
        for i, name in enumerate(player_name_list):
            f.write(f"{i}\t{name}\t{np.mean(player_elo_history[name]):.0f}\n")

    print(f"Output: {args.outdir}/")


if __name__ == "__main__":
    main()
