#!/bin/bash
# run_sealed_evaluation.sh — FINAL, ONE-SHOT evaluation on the sealed
# 2024-12 validation set. Run ONCE at paper finalization. Do NOT rerun with
# tweaks — this is the held-out number that goes in the paper.
#
# Usage:
#   bash evaluation/run_sealed_evaluation.sh <ckpt.pt> <outdir>
#
# Example:
#   bash evaluation/run_sealed_evaluation.sh models/prev_move_cnn_v2.pt \
#        evaluation/results_sealed_mem
#   bash evaluation/run_sealed_evaluation.sh models/prev_move_cnn_v2_dedup.pt \
#        evaluation/results_sealed_nomem

set -e
cd "$(dirname "$0")/.."   # → faichess_nn/

CKPT="${1:?usage: $0 <ckpt.pt> <outdir>}"
OUTDIR="${2:?usage: $0 <ckpt.pt> <outdir>}"
SEALED_PGN="lichess_data/validation_SEALED/val_2024_12_2400plus.pgn"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"; exit 1
fi
if [ ! -f "$SEALED_PGN" ]; then
    echo "ERROR: sealed PGN not found: $SEALED_PGN"; exit 1
fi

mkdir -p "$OUTDIR"
echo "=== SEALED evaluation ==="
echo "CKPT:   $CKPT"
echo "SEALED: $SEALED_PGN"
echo "OUT:    $OUTDIR"
echo

# 1) Transfer (next-move) on SEALED
echo "[1/4] Next-move transfer on SEALED..."
python3 evaluation/test_nextmove.py --ckpt "$CKPT" \
        --pgn "$SEALED_PGN" \
        > "$OUTDIR/nextmove_sealed.log" 2>&1
echo "    → $OUTDIR/nextmove_sealed.log"

# 2) Fine-tune on SEALED
echo "[2/4] Fine-tune (Tier 3) on SEALED..."
python3 evaluation/test_finetune.py --ckpt "$CKPT" \
        --label sealed \
        > "$OUTDIR/finetune_sealed.log" 2>&1
echo "    → $OUTDIR/finetune_sealed.log"

# 3) Style detection (if GM data present)
if [ -f "../data_style/gm_games.pgn" ]; then
    echo "[3/4] GM style detection (fixed GM set, not SEALED)..."
    python3 evaluation/test_style.py --ckpt "$CKPT" \
            > "$OUTDIR/style.log" 2>&1
    echo "    → $OUTDIR/style.log"
else
    echo "[3/4] Skipped style (no GM PGN)"
fi

# 4) Latent walk on SEALED anchors (qualitative paper figure)
echo "[4/4] Latent walk (qualitative demo)..."
python3 evaluation/test_latent_walk.py --ckpt "$CKPT" \
        > "$OUTDIR/latent_walk.log" 2>&1
echo "    → $OUTDIR/latent_walk.log"

echo
echo "SEALED eval complete. Results in $OUTDIR/"
echo "THESE ARE THE NUMBERS THAT GO IN THE PAPER. DO NOT RERUN."
