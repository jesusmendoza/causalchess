#!/usr/bin/env python3
"""
estimate_bayes_ceiling.py — Estimate the theoretical ceiling of prev-move
prediction accuracy via MLE and Bayesian posterior with an informative prior.

The issue: for positions that appear only once in training, MLE says "100%
accuracy" which is biased up — we have no evidence about the true distribution.
Bayesian smoothing with the global marginal over prev_moves as informative
prior addresses this.

Formula per position X with N observations and counts c[move]:

    post_mean[move] = (κ·p_global[move] + c[move]) / (κ + N)

Contribution to ceiling = max_move(post_mean) × N (this position contributes N
samples, each gets accuracy max(post_mean) in the limit).

Final ceiling = Σ contributions / Σ N.

We report ceilings for several κ:
  κ = 0   (pure MLE, biased up)
  κ = 1   (mild smoothing)
  κ = 10  (moderate)
  κ = 50  (strong)
  κ = 1000 (dominated by prior, lower bound)
"""
import numpy as np
import os
import time


BIN_DIR = "data"
IN_X = os.path.join(BIN_DIR, "chess_x.bin")
IN_Y = os.path.join(BIN_DIR, "chess_y.bin")
IN_META = os.path.join(BIN_DIR, "chess_meta.txt")


def main():
    with open(IN_META) as f:
        meta = dict(line.strip().split('=') for line in f if '=' in line)
    n = int(meta['n_samples'])
    n_moves = int(meta['n_moves'])
    print(f"Loading {n:,} samples, {n_moves} prev_move classes...")

    x = np.memmap(IN_X, dtype=np.uint64, mode='r', shape=(n, 12))
    y_np = np.fromfile(IN_Y, dtype=np.uint16).astype(np.int64)

    # Global marginal over prev_moves
    p_global = np.bincount(y_np, minlength=n_moves).astype(np.float64)
    p_global /= p_global.sum()
    print(f"Global marginal: max move freq = {p_global.max()*100:.3f}%, "
          f"entropy = {-np.sum(p_global * np.log2(p_global + 1e-12)):.2f} bits")

    # Sort samples by position bytes (96 bytes per row)
    print("\nSorting 25M samples by position bytes...")
    t0 = time.time()
    x_view = np.ascontiguousarray(x).view(np.dtype((np.void, 96))).ravel()
    order = np.argsort(x_view, kind='stable')
    print(f"  sorted in {time.time()-t0:.1f}s")

    # Group boundaries where position changes
    sorted_x = x_view[order]
    changes = np.where(sorted_x[1:] != sorted_x[:-1])[0] + 1
    boundaries = np.concatenate(([0], changes, [n]))
    n_groups = len(boundaries) - 1
    print(f"  {n_groups:,} unique positions")

    sorted_y = y_np[order]
    group_sizes = np.diff(boundaries)  # (n_groups,)

    # Singletons (N=1 positions): handle in bulk (fast path)
    print("\nProcessing singletons in bulk...")
    t0 = time.time()
    is_singleton = (group_sizes == 1)
    n_singletons = int(is_singleton.sum())
    # The single observation per singleton is sorted_y[boundaries[i]]
    singleton_y = sorted_y[boundaries[:-1][is_singleton]]
    # MLE contribution per singleton = 1
    # Bayesian contribution (κ, N=1): (κ·p_global[c_obs] + 1) / (κ + 1)

    # Multi-count positions
    multi_idx = np.where(~is_singleton)[0]
    n_multi = len(multi_idx)
    total_multi_samples = int(group_sizes[~is_singleton].sum())
    print(f"  {n_singletons:,} singletons ({100*n_singletons/n_groups:.1f}% of positions)")
    print(f"  {n_multi:,} multi-count positions, {total_multi_samples:,} samples ({100*total_multi_samples/n:.1f}% of data)")
    print(f"  singleton pass: {time.time()-t0:.1f}s")

    kappas = [0.0, 1.0, 10.0, 50.0, 200.0, 1000.0]

    # Initialize sums
    mle_sum = 0.0
    bayes_sums = {k: 0.0 for k in kappas}

    # Singletons — vectorized
    p_obs = p_global[singleton_y]   # (n_singletons,)
    mle_sum += n_singletons  # MLE: each contributes 1
    for k in kappas:
        if k == 0.0:
            # pure MLE: singleton contributes 1
            bayes_sums[k] += n_singletons
        else:
            # max post_mean for singleton is always the observed class
            # (as long as p_global_max_other * κ < κ·p_obs + 1, which holds for typical values)
            # To be safe, compute both the observed-class post_mean AND the global-argmax post_mean
            obs_mean = (k * p_obs + 1) / (k + 1)  # (n_singletons,)
            global_argmax_prob = p_global.max()  # max over classes
            global_mean = (k * global_argmax_prob + 0) / (k + 1)  # for non-observed classes
            # Max is obs_mean if it's ≥ global_mean, else global_mean
            max_mean = np.maximum(obs_mean, global_mean)
            bayes_sums[k] += float(max_mean.sum())

    # Multi-count positions — loop (there are few of them relative to singletons)
    print(f"\nProcessing {n_multi:,} multi-count positions...")
    t0 = time.time()
    report_every = max(1, n_multi // 20)
    counts = np.zeros(n_moves, dtype=np.float64)
    for loop_i, gi in enumerate(multi_idx):
        lo, hi = boundaries[gi], boundaries[gi+1]
        N_pos = hi - lo
        group_y = sorted_y[lo:hi]
        counts.fill(0)
        np.add.at(counts, group_y, 1.0)

        # MLE
        mle_sum += counts.max()

        # Bayesian for each κ
        for k in kappas:
            if k == 0.0:
                bayes_sums[k] += counts.max()
            else:
                # post_mean = (κ * p_global + counts) / (κ + N_pos)
                # We want max(post_mean), then multiply by N_pos for contribution
                post_max = ((k * p_global + counts).max()) / (k + N_pos)
                bayes_sums[k] += post_max * N_pos

        if (loop_i + 1) % report_every == 0:
            print(f"  {loop_i+1}/{n_multi} ({100*(loop_i+1)/n_multi:.0f}%) elapsed {time.time()-t0:.0f}s", flush=True)

    print(f"  multi-count pass: {time.time()-t0:.1f}s")

    # Results
    print("\n" + "="*65)
    print(f"  Total samples: {n:,}")
    print(f"  Unique positions: {n_groups:,}  (singletons: {n_singletons:,})")
    print(f"  Global marginal max: {p_global.max()*100:.3f}% (pure-prior ceiling)")
    print("="*65)
    print(f"  {'κ':<8} {'Bayes ceiling':>14} {'Description':<40}")
    print("-"*65)
    ceiling_mle = mle_sum / n * 100
    print(f"  {'0 (MLE)':<8} {ceiling_mle:>13.2f}% absolute upper bound, biased up")
    for k in kappas[1:]:
        c = bayes_sums[k] / n * 100
        if k == 1:     desc = "mild smoothing"
        elif k == 10:  desc = "moderate prior weight"
        elif k == 50:  desc = "strong prior weight"
        elif k == 200: desc = "very strong prior"
        else:          desc = "dominated by global prior (lower bound)"
        print(f"  κ={k:<6} {c:>13.2f}% {desc}")
    print("="*65)
    print(f"\n  Our CausalChess-mem: ~40.4% val accuracy")
    print(f"  (for interpretation: compare to Bayesian ceilings above)\n")


if __name__ == "__main__":
    main()
