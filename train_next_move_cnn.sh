#!/usr/bin/env bash
# train_next_move_cnn.sh — Next-move CNN baseline (same arch as PMP).
#
# Run AFTER logs/download_next_full.log shows "Done: ... pairs"
# and wc -l lichess_data/next_move_2400.tsv is ~25M+ (not ~2.8M).
#
# Pipeline mirrors prev-move:
#   download_lichess_next.py  ↔  download_lichess.py
#   prepare_move_bin.py       ↔  (same binary layout as data/)
#   prev_move_train.py --task next_move  ↔  --task prev_move
set -euo pipefail
cd "$(dirname "$0")"

TSV="${TSV:-lichess_data/next_move_2400.tsv}"
OUTDIR="${OUTDIR:-data_next}"
CKPT="${CKPT:-models/next_move_cnn.pt}"
# Match PMP scale (~25.3M); download may be larger (stopped early).
MAX_SAMPLES="${MAX_SAMPLES:-25000000}"

if [[ ! -f "$TSV" ]]; then
  echo "Missing $TSV — wait for download to finish."
  exit 1
fi

NLINES=$(wc -l < "$TSV")
echo "TSV lines: $NLINES (need ≥ $MAX_SAMPLES pairs; will pack first $MAX_SAMPLES)"
if (( NLINES < MAX_SAMPLES )); then
  echo "WARNING: TSV has fewer than $MAX_SAMPLES lines. Aborting."
  exit 1
fi

echo "=== 1/2 Pack binary → $OUTDIR (max_samples=$MAX_SAMPLES) ==="
python3 prepare_move_bin.py \
  --tsv "$TSV" \
  --move-col next_move \
  --outdir "$OUTDIR" \
  --max-samples "$MAX_SAMPLES"

echo "=== 2/2 Train CNN (same hyperparams spirit as PMP production) ==="
python3 prev_move_train.py \
  --task next_move \
  --arch cnn \
  --bin-dir "$OUTDIR" \
  --output "$CKPT" \
  --epochs 100 \
  --patience 5 \
  --batch-size 4096 \
  --lr 1e-3 \
  --embed-dim 256 \
  --num-workers 4

echo "Done → $CKPT"
echo "Next: run probes with --ckpt-next $CKPT (extend test_probes_raw_baseline.py)"
