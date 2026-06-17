#!/bin/bash
# run_all_on_v2.sh — run all evaluation tests on the v2 CNN checkpoint.
# Save outputs to evaluation/results_v2/.
#
# Run from FAI/faichess_nn/:
#   bash evaluation/run_all_on_v2.sh
#
# Each test script supports an optional --ckpt argument. Scripts without it
# fall back to prev_move_cnn.pt; patch below redirects them to v2.

set -e
cd "$(dirname "$0")/.."   # move to faichess_nn/

CKPT="models/prev_move_cnn_v2.pt"
OUTDIR="evaluation/results_v2"
mkdir -p "$OUTDIR"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Training still running?"
    exit 1
fi

echo "=== Running all evaluations on $CKPT ==="
echo "Output dir: $OUTDIR"
echo

# Temp: swap model path in scripts by env var override
# Scripts read models/prev_move_cnn.pt by default; we backup and patch.
BACKUP_DIR="$(mktemp -d)"
for s in evaluation/test_probes.py evaluation/test_probes_strategic.py \
         evaluation/test_trajectories.py evaluation/test_landscape.py \
         evaluation/eval_arithmetic_compare.py evaluation/eval_neighborhoods.py; do
    [ -f "$s" ] && cp "$s" "$BACKUP_DIR/$(basename $s)"
done

# Patch to use v2 checkpoint
sed -i 's|models/prev_move_cnn.pt|models/prev_move_cnn_v2.pt|g' \
    evaluation/test_probes.py \
    evaluation/test_probes_strategic.py \
    evaluation/test_trajectories.py \
    evaluation/test_landscape.py 2>/dev/null || true

restore() {
    for s in "$BACKUP_DIR"/*.py; do
        cp "$s" "evaluation/$(basename $s)"
    done
    rm -rf "$BACKUP_DIR"
    echo "Restored original scripts."
}
trap restore EXIT

echo "[1/5] Linear probes (basic)..."
CUDA_VISIBLE_DEVICES= python3 evaluation/test_probes.py > "$OUTDIR/probes.log" 2>&1
echo "    → $OUTDIR/probes.log"

echo "[2/5] Linear probes (strategic + per-phase)..."
CUDA_VISIBLE_DEVICES= python3 evaluation/test_probes_strategic.py > "$OUTDIR/probes_strategic.log" 2>&1
echo "    → $OUTDIR/probes_strategic.log"

echo "[3/5] Trajectories..."
CUDA_VISIBLE_DEVICES= python3 evaluation/test_trajectories.py > "$OUTDIR/trajectories.log" 2>&1
# Copy figure
[ -f paper/figures/test_trajectories.png ] && cp paper/figures/test_trajectories.png "$OUTDIR/trajectories.png"
echo "    → $OUTDIR/trajectories.log + .png"

echo "[4/5] Landscape (PCA/t-SNE/UMAP)..."
CUDA_VISIBLE_DEVICES= python3 evaluation/test_landscape.py > "$OUTDIR/landscape.log" 2>&1
[ -f paper/figures/test_landscape.png ] && cp paper/figures/test_landscape.png "$OUTDIR/landscape.png"
echo "    → $OUTDIR/landscape.log + .png"

echo "[5/5] Chess2Vec arithmetic comparison..."
CUDA_VISIBLE_DEVICES= python3 evaluation/eval_arithmetic_compare.py > "$OUTDIR/arithmetic.log" 2>&1
echo "    → $OUTDIR/arithmetic.log"

echo
echo "All done. Inspect $OUTDIR/"
