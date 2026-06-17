#!/usr/bin/env python3
"""Convert prev_move TSV to binary format for the C training pipeline.

Output:
  data/chess_x.bin  — N × 12 uint64 (bitboards per position)
  data/chess_y.bin  — N × uint16 (move index)
  data/chess_vocab.txt — move vocabulary (move\tidx per line)
"""
import csv
import numpy as np
import os
import struct
import sys


def fen_to_bitboards(fen: str) -> list:
    """Convert FEN to 12 uint64s."""
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    boards = [0] * 12
    parts = fen.split()
    sq = 56
    for ch in parts[0]:
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


def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else "lichess_data/prev_move_full.tsv"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    outdir = "data"
    os.makedirs(outdir, exist_ok=True)

    # First pass: build vocab
    print("Building vocab...")
    moves = set()
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            moves.add(row['prev_move'])
    vocab = {m: i for i, m in enumerate(sorted(moves))}
    n_moves = len(vocab)
    print(f"  {n_moves} unique moves")

    with open(f"{outdir}/chess_vocab.txt", "w") as f:
        for m, i in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{m}\t{i}\n")

    # Second pass: convert to binary
    print("Converting to binary...")
    x_list = []
    y_list = []
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            move = row['prev_move']
            if move not in vocab:
                continue
            boards = fen_to_bitboards(row['fen'])
            x_list.append(boards)
            y_list.append(vocab[move])
            if max_samples > 0 and len(x_list) >= max_samples:
                break
            if (i + 1) % 500000 == 0:
                print(f"  {i+1} rows...", flush=True)

    n = len(x_list)
    print(f"  {n} samples")

    # Write binary: x = N × 12 uint64, y = N × uint16
    x_arr = np.array(x_list, dtype=np.uint64)
    y_arr = np.array(y_list, dtype=np.uint16)

    x_arr.tofile(f"{outdir}/chess_x.bin")
    y_arr.tofile(f"{outdir}/chess_y.bin")

    # Write metadata
    with open(f"{outdir}/chess_meta.txt", "w") as f:
        f.write(f"n_samples={n}\n")
        f.write(f"n_moves={n_moves}\n")
        f.write(f"in_words=12\n")
        f.write(f"in_bits=768\n")

    print(f"Wrote {outdir}/chess_x.bin ({x_arr.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {outdir}/chess_y.bin ({y_arr.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {outdir}/chess_vocab.txt, {outdir}/chess_meta.txt")


if __name__ == "__main__":
    main()
