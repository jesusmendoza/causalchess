#!/usr/bin/env python3
"""
prepare_prev_move_bin.py — thin wrapper (back-compat).

Prefers the shared prepare_move_bin.py so prev-move and next-move stay in sync.

  python3 prepare_prev_move_bin.py [tsv] [max_samples]
  python3 prepare_prev_move_bin.py --tsv ... --move-col prev_move --outdir data
"""
from prepare_move_bin import main

if __name__ == "__main__":
    main()
