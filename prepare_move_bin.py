#!/usr/bin/env python3
"""
prepare_move_bin.py — Pack (fen, move) TSV → binary training set.

Shared by Previous-Move (PMP) and Next-Move baselines. Same layout as the
historical prepare_prev_move_bin.py so prev_move_train.py --bin-dir works
unchanged.

Output (in --outdir):
  chess_x.bin      — N × 12 uint64 bitboards
  chess_y.bin      — N × uint16 move indices
  chess_phase.bin  — N × uint8 material-based phase (0/1/2; diagnostic only)
  chess_vocab.txt  — move\\tidx
  chess_meta.txt   — n_samples, n_moves, ...

Usage:
  # Previous-move (default; same as before)
  python3 prepare_move_bin.py \\
      --tsv lichess_data/prev_move_2400.tsv --move-col prev_move --outdir data

  # Next-move baseline (after full download)
  python3 prepare_move_bin.py \\
      --tsv lichess_data/next_move_2400.tsv --move-col next_move --outdir data_next
"""
import argparse
import csv
import os
import sys

import numpy as np


def fen_to_bitboards(fen: str) -> list:
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    boards = [0] * 12
    sq = 56
    for ch in fen.split()[0]:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            idx = piece_map.get(ch)
            if idx is not None:
                boards[idx] |= 1 << sq
            sq += 1
    return boards


def phase_from_bitboards(boards):
    """Material phase: opening >=70, middlegame >=40, endgame <40.

    Kings are excluded; these labels support dataset diagnostics and are not
    the previous-move training target.
    """
    weights = (1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0)
    material = sum(int(bb).bit_count() * w for bb, w in zip(boards, weights))
    return 0 if material >= 70 else 1 if material >= 40 else 2


def pack_tsv(tsv, move_col, outdir, max_samples=0):
    os.makedirs(outdir, exist_ok=True)

    print(f"Building vocab from {tsv} (column={move_col!r})...")
    moves = set()
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if move_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column {move_col!r} not in TSV headers: {reader.fieldnames}"
            )
        for row in reader:
            moves.add(row[move_col])
    vocab = {m: i for i, m in enumerate(sorted(moves))}
    n_moves = len(vocab)
    print(f"  {n_moves} unique moves")

    with open(os.path.join(outdir, "chess_vocab.txt"), "w") as f:
        for m, i in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{m}\t{i}\n")

    print("Converting to binary...")
    x_list = []
    y_list = []
    phase_list = []
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            move = row[move_col]
            if move not in vocab:
                continue
            boards = fen_to_bitboards(row["fen"])
            x_list.append(boards)
            y_list.append(vocab[move])
            phase_list.append(phase_from_bitboards(boards))
            if max_samples > 0 and len(x_list) >= max_samples:
                break
            if (i + 1) % 500000 == 0:
                print(f"  {i + 1} rows...", flush=True)

    n = len(x_list)
    print(f"  {n} samples")
    x_arr = np.array(x_list, dtype=np.uint64)
    y_arr = np.array(y_list, dtype=np.uint16)
    x_arr.tofile(os.path.join(outdir, "chess_x.bin"))
    y_arr.tofile(os.path.join(outdir, "chess_y.bin"))
    np.asarray(phase_list, dtype=np.uint8).tofile(
        os.path.join(outdir, "chess_phase.bin"))

    with open(os.path.join(outdir, "chess_meta.txt"), "w") as f:
        f.write(f"n_samples={n}\n")
        f.write(f"n_moves={n_moves}\n")
        f.write(f"in_words=12\n")
        f.write(f"in_bits=768\n")
        f.write(f"move_col={move_col}\n")
        f.write(f"source_tsv={tsv}\n")
        f.write("phase_definition=material_Q9_R5_B3_N3_P1_open70_middle40\n")

    print(f"Wrote {outdir}/chess_x.bin ({x_arr.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {outdir}/chess_y.bin ({y_arr.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {outdir}/chess_phase.bin, chess_vocab.txt, chess_meta.txt")


def main(argv=None):
    # Back-compat: prepare_prev_move_bin.py style positional args
    if argv is None and len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        tsv = sys.argv[1]
        max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        pack_tsv(tsv, "prev_move", "data", max_samples)
        return

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", default="", help="Input TSV with fen + move column")
    ap.add_argument(
        "--move-col",
        default="prev_move",
        choices=["prev_move", "next_move"],
        help="Label column (prev_move = PMP, next_move = next-move baseline)",
    )
    ap.add_argument(
        "--outdir",
        default="",
        help="Output dir (default: data for prev_move, data_next for next_move)",
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)
    if not args.tsv:
        args.tsv = (
            "lichess_data/next_move_2400.tsv"
            if args.move_col == "next_move"
            else "lichess_data/prev_move_2400.tsv"
        )
    if not args.outdir:
        args.outdir = "data_next" if args.move_col == "next_move" else "data"
    pack_tsv(args.tsv, args.move_col, args.outdir, args.max_samples)


if __name__ == "__main__":
    main()
