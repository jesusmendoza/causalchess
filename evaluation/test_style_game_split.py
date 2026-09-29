#!/usr/bin/env python3
"""Re-evaluate the four-player endgame probe with games as split units.

Uses the same filtering and class balancing as test_style_endgame.py. Reports
balanced accuracy because holding out games does not preserve equal class sizes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess.pgn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from test_style_endgame import (
    MAX_MAT_DIFF, MAX_PIECES, MAX_POS_PER_GAME, MIN_PLY, PGN_PATH, PLAYERS,
    embed_batch, load_trained, load_untrained, material_diff, piece_count,
)


def extract_with_game_ids(path):
    data = {player: [] for player in PLAYERS}
    with open(path) as stream:
        game_id = 0
        while (game := chess.pgn.read_game(stream)) is not None:
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            if white in PLAYERS or black in PLAYERS:
                board = game.board()
                kept = 0
                for ply, move in enumerate(game.mainline_moves()):
                    if (ply >= MIN_PLY and piece_count(board) <= MAX_PIECES
                            and material_diff(board) <= MAX_MAT_DIFF):
                        player = white if ply % 2 == 0 else black
                        if player in PLAYERS:
                            data[player].append((board.fen(), game_id))
                            kept += 1
                        if kept >= MAX_POS_PER_GAME:
                            break
                    board.push(move)
            game_id += 1
    return data


def score(embedding, labels, train, valid):
    probe = LogisticRegression(max_iter=2000)
    probe.fit(embedding[train], labels[train])
    prediction = probe.predict(embedding[valid])
    return (100 * accuracy_score(labels[valid], prediction),
            100 * balanced_accuracy_score(labels[valid], prediction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt")
    parser.add_argument("--pgn", default=PGN_PATH)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    data = extract_with_game_ids(args.pgn)
    per_player = min(len(data[player]) for player in PLAYERS)
    fens = [fen for player in PLAYERS for fen, _ in data[player][:per_player]]
    games = np.array([gid for player in PLAYERS
                      for _, gid in data[player][:per_player]])
    labels = np.repeat(np.arange(len(PLAYERS)), per_player)
    print(f"Positions: {len(labels)}; distinct games: {len(set(games))}; "
          f"players: {len(PLAYERS)}; balanced positions/player: {per_player}")

    trained, checkpoint = load_trained(args.ckpt)
    untrained = load_untrained(checkpoint)
    trained_emb = embed_batch(trained, fens)
    untrained_emb = embed_batch(untrained, fens)

    train, valid = train_test_split(np.arange(len(labels)), test_size=0.2,
                                    random_state=42, stratify=labels)
    shared = sum(gid in set(games[train]) for gid in games[valid])
    print(f"Position split: {shared}/{len(valid)} validation positions "
          "share a game with training")
    print(f"  PMP accuracy/balanced: {score(trained_emb, labels, train, valid)}")
    print(f"  Untrained accuracy/balanced: {score(untrained_emb, labels, train, valid)}")

    group_split = GroupShuffleSplit(n_splits=args.splits, test_size=0.2,
                                    random_state=42)
    results = []
    print("Game split | train games | valid games | PMP acc/bal | untrained acc/bal")
    for index, (train, valid) in enumerate(group_split.split(trained_emb, labels, games), 1):
        if set(games[train]) & set(games[valid]):
            raise AssertionError("Game groups overlap")
        trained_score = score(trained_emb, labels, train, valid)
        untrained_score = score(untrained_emb, labels, train, valid)
        results.append((trained_score[1], untrained_score[1]))
        print(f"{index:>10} | {len(set(games[train])):>11} | "
              f"{len(set(games[valid])):>11} | "
              f"{trained_score[0]:.2f}/{trained_score[1]:.2f} | "
              f"{untrained_score[0]:.2f}/{untrained_score[1]:.2f}")
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0, ddof=1) if args.splits > 1 else np.zeros(2)
    print(f"Mean balanced accuracy +/- split SD: PMP {means[0]:.2f} +/- {stds[0]:.2f}; "
          f"untrained {means[1]:.2f} +/- {stds[1]:.2f}")


if __name__ == "__main__":
    main()
