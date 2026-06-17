#!/bin/bash
# run_all.sh — run the full evaluation suite on a checkpoint, write to outdir.
#
# Usage:
#   bash evaluation/run_all.sh <ckpt.pt> <outdir>
#
# Examples:
#   bash evaluation/run_all.sh models/prev_move_cnn_v2.pt       evaluation/results_mem
#   bash evaluation/run_all.sh models/prev_move_cnn_v2_dedup.pt evaluation/results_nomem
#
# Runs (in order):
#   1. test_probes          — linear probes (basic 9)
#   2. test_probes_strategic— linear probes (24 strategic, per-phase)
#   3. test_probes_mlp      — Tier 2: MLP probes (non-linear)
#   4. test_trajectories    — game trajectory smoothness
#   5. test_landscape       — PCA/t-SNE/UMAP figures
#   6. eval_arithmetic_compare — chess algebra vs untrained
#   7. eval_neighborhoods   — nearest-neighbor qualitative
#   8. test_latent_walk     — random walk in embedding space
#   9. test_nextmove        — transfer to next-move prediction
#  10. test_style           — GM style detection (requires GM PGN)
#  11. test_style_endgame   — GM style on endgames only
#  12. test_algebra_by_tier — algebra filtered by FEN uniqueness tier

set -e
cd "$(dirname "$0")/.."   # → faichess_nn/

CKPT="${1:?usage: $0 <ckpt.pt> <outdir>}"
OUTDIR="${2:?usage: $0 <ckpt.pt> <outdir>}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"; exit 1
fi

mkdir -p "$OUTDIR"
echo "=== Full evaluation suite ==="
echo "CKPT: $CKPT"
echo "OUT:  $OUTDIR"
echo

run_test() {
    local name="$1"; shift
    local logfile="$OUTDIR/${name}.log"
    echo "[$name]"
    if python3 "$@" --ckpt "$CKPT" > "$logfile" 2>&1; then
        echo "    ✓ → $logfile"
    else
        echo "    ✗ FAILED → $logfile"
    fi
}

run_test probes             evaluation/test_probes.py
run_test probes_strategic   evaluation/test_probes_strategic.py
run_test probes_mlp         evaluation/test_probes_mlp.py
run_test trajectories       evaluation/test_trajectories.py
run_test landscape          evaluation/test_landscape.py
run_test arithmetic         evaluation/eval_arithmetic_compare.py
run_test neighborhoods      evaluation/eval_neighborhoods.py
run_test latent_walk        evaluation/test_latent_walk.py
run_test nextmove           evaluation/test_nextmove.py
run_test algebra_by_tier    evaluation/test_algebra_by_tier.py

# Style tests require external GM PGN
if [ -f "../data_style/gm_games.pgn" ]; then
    run_test style              evaluation/test_style.py
    run_test style_endgame      evaluation/test_style_endgame.py
else
    echo "[style] skipped (no ../data_style/gm_games.pgn)"
fi

# Figures, if produced, are in paper/figures/. Copy to outdir.
for fig in test_trajectories test_landscape; do
    if [ -f "paper/figures/${fig}.png" ]; then
        cp "paper/figures/${fig}.png" "$OUTDIR/${fig}.png"
    fi
done

echo
echo "=== All done ==="
echo "Inspect $OUTDIR/"
