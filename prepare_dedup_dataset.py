#!/usr/bin/env python3
"""
prepare_dedup_dataset.py — deduplicate (position, prev_move) binomials.

This script keeps each unique (768-bit position, previous-move label) pair once.
It removes exact pair repetition while retaining distinct previous moves that
lead to the same board. The ablation measures sensitivity to repeated pairs;
it does not establish the absence of all memorization or prove generalization.

Outputs (do NOT overwrite originals):
  data/chess_x_dedup.bin      N_dedup × 12 uint64
  data/chess_y_dedup.bin      N_dedup × uint16
  data/chess_phase_dedup.bin  N_dedup × uint8
  data/chess_meta_dedup.txt
"""
import numpy as np
import os
import time


BIN_DIR = "data"
IN_X = os.path.join(BIN_DIR, "chess_x.bin")
IN_Y = os.path.join(BIN_DIR, "chess_y.bin")
IN_PHASE = os.path.join(BIN_DIR, "chess_phase.bin")
IN_META = os.path.join(BIN_DIR, "chess_meta.txt")
OUT_X = os.path.join(BIN_DIR, "chess_x_dedup.bin")
OUT_Y = os.path.join(BIN_DIR, "chess_y_dedup.bin")
OUT_PHASE = os.path.join(BIN_DIR, "chess_phase_dedup.bin")
OUT_META = os.path.join(BIN_DIR, "chess_meta_dedup.txt")


def main():
    with open(IN_META) as f:
        meta = dict(line.strip().split('=') for line in f if '=' in line)
    n = int(meta['n_samples'])
    n_moves = int(meta['n_moves'])
    if not os.path.exists(IN_PHASE):
        raise FileNotFoundError(
            f"{IN_PHASE} is missing. Repack the TSV with prepare_move_bin.py; "
            "it generates the material-based phase labels used for diagnostics."
        )
    print(f"Loading {n:,} samples from {IN_X}, {IN_Y}, {IN_PHASE}...")

    x = np.memmap(IN_X, dtype=np.uint64, mode='r', shape=(n, 12))
    y = np.memmap(IN_Y, dtype=np.uint16, mode='r', shape=(n,))
    phase = np.memmap(IN_PHASE, dtype=np.uint8, mode='r', shape=(n,))

    # Build combined byte view: each row = 96 bytes (x) + 2 bytes (y) = 98 bytes.
    # View as void-98 so np.unique can sort-and-dedup in one pass.
    print(f"\nBuilding combined byte array ({n * 98 / 1e9:.2f} GB)...")
    t0 = time.time()
    raw = np.empty((n, 98), dtype=np.uint8)
    raw[:, :96] = np.asarray(x).view(np.uint8).reshape(n, 96)
    raw[:, 96:] = np.asarray(y).view(np.uint8).reshape(n, 2)
    print(f"  built in {time.time()-t0:.1f}s")

    print("Running np.unique (sort-based dedup, may take 1-3 min)...")
    t0 = time.time()
    v = np.ascontiguousarray(raw).view(np.dtype((np.void, 98))).ravel()
    _, first_idx = np.unique(v, return_index=True)
    print(f"  unique done in {time.time()-t0:.1f}s")
    del v, raw

    # Preserve original ordering (game sequences stay contiguous, important for
    # sequential 90/10 train/val split that prev_move_train.py uses).
    first_idx.sort()
    n_dedup = len(first_idx)
    kept = 100 * n_dedup / n
    print(f"\nDedup: {n:,} -> {n_dedup:,} unique binomials ({kept:.2f}% retained)")
    print(f"  Removed {n - n_dedup:,} duplicates ({100 - kept:.2f}%)")

    print(f"\nWriting {OUT_X}...")
    np.asarray(x)[first_idx].tofile(OUT_X)
    print(f"Writing {OUT_Y}...")
    np.asarray(y)[first_idx].tofile(OUT_Y)
    print(f"Writing {OUT_PHASE}...")
    phase_dedup = np.asarray(phase)[first_idx]
    phase_dedup.tofile(OUT_PHASE)

    with open(OUT_META, "w") as f:
        f.write(f"n_samples={n_dedup}\n")
        f.write(f"n_moves={n_moves}\n")
        f.write(f"in_words=12\n")
        f.write(f"in_bits=768\n")
    print(f"Wrote {OUT_META}")

    print("\nPhase distribution comparison:")
    orig_counts = np.bincount(np.asarray(phase), minlength=3)
    dedup_counts = np.bincount(phase_dedup, minlength=3)
    labels = ["Opening", "Middle ", "Endgame"]
    print(f"  {'Phase':<8} {'Original':>12} {'Dedup':>12} {'Kept ratio':>12}")
    for i, lbl in enumerate(labels):
        o_pct = 100 * orig_counts[i] / n
        d_pct = 100 * dedup_counts[i] / n_dedup
        ratio = dedup_counts[i] / max(orig_counts[i], 1)
        print(f"  {lbl:<8} {o_pct:>11.2f}% {d_pct:>11.2f}% {ratio:>11.3f}")

    # Move-label distribution: how much does the class imbalance shrink?
    print("\nTop-100 class share (measures residual move-level imbalance):")
    y_dedup = np.asarray(y)[first_idx]
    orig_top100 = np.sort(np.bincount(np.asarray(y), minlength=n_moves))[::-1][:100].sum()
    dedup_top100 = np.sort(np.bincount(y_dedup, minlength=n_moves))[::-1][:100].sum()
    print(f"  Original: top-100 moves = {100*orig_top100/n:.2f}% of labels")
    print(f"  Dedup:    top-100 moves = {100*dedup_top100/n_dedup:.2f}% of labels")


if __name__ == "__main__":
    main()
