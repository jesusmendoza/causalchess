#!/usr/bin/env python3
"""
fit_kappa_empirical_bayes.py — Fit the Dirichlet prior strength κ by maximum
marginal likelihood (empirical Bayes) on multi-count positions, then compute
the Bayes ceiling at the fit κ.

Model: for each position X, prev_move distribution p_X ~ Dirichlet(κ · p_global),
and observed counts c_X ~ Multinomial(N_X, p_X). Marginalizing p_X gives the
Dirichlet-multinomial likelihood, which we maximize over κ.

We use only multi-count positions (N ≥ 2) because singletons are uninformative
about κ (the likelihood is dominated by the prior itself).
"""
import numpy as np
import os
import time
from scipy.special import gammaln
from scipy.optimize import minimize_scalar


BIN_DIR = "data"
IN_X = os.path.join(BIN_DIR, "chess_x.bin")
IN_Y = os.path.join(BIN_DIR, "chess_y.bin")
IN_META = os.path.join(BIN_DIR, "chess_meta.txt")


def main():
    with open(IN_META) as f:
        meta = dict(line.strip().split('=') for line in f if '=' in line)
    n = int(meta['n_samples'])
    n_moves = int(meta['n_moves'])
    print(f"Loading {n:,} samples, {n_moves} move classes...")

    x = np.memmap(IN_X, dtype=np.uint64, mode='r', shape=(n, 12))
    y_np = np.fromfile(IN_Y, dtype=np.uint16).astype(np.int64)

    p_global = np.bincount(y_np, minlength=n_moves).astype(np.float64)
    p_global /= p_global.sum()

    # Sort by position bytes
    print("\nSorting samples by position...")
    t0 = time.time()
    x_view = np.ascontiguousarray(x).view(np.dtype((np.void, 96))).ravel()
    order = np.argsort(x_view, kind='stable')
    sorted_y = y_np[order]
    sorted_x = x_view[order]
    changes = np.where(sorted_x[1:] != sorted_x[:-1])[0] + 1
    boundaries = np.concatenate(([0], changes, [n]))
    group_sizes = np.diff(boundaries)
    print(f"  sorted in {time.time()-t0:.1f}s, {len(boundaries)-1:,} unique positions")

    # Extract multi-count positions (N >= 2)
    print("Extracting multi-count positions...")
    t0 = time.time()
    is_multi = group_sizes >= 2
    multi_idx = np.where(is_multi)[0]
    print(f"  {len(multi_idx):,} multi-count positions covering "
          f"{int(group_sizes[is_multi].sum()):,} samples")

    # Flatten counts representation:
    #   flat_moves:  concatenated nonzero move indices across positions
    #   flat_counts: corresponding counts
    #   pos_starts:  starting index of each position in the flat arrays
    #   N_per_pos:   total N for each position
    flat_moves_list = []
    flat_counts_list = []
    pos_starts = [0]
    N_per_pos = np.empty(len(multi_idx), dtype=np.int64)

    for i, gi in enumerate(multi_idx):
        lo, hi = boundaries[gi], boundaries[gi+1]
        group_y = sorted_y[lo:hi]
        unique_moves, counts = np.unique(group_y, return_counts=True)
        flat_moves_list.append(unique_moves)
        flat_counts_list.append(counts)
        pos_starts.append(pos_starts[-1] + len(unique_moves))
        N_per_pos[i] = counts.sum()

    flat_moves = np.concatenate(flat_moves_list).astype(np.int64)
    flat_counts = np.concatenate(flat_counts_list).astype(np.float64)
    pos_starts = np.array(pos_starts, dtype=np.int64)
    n_pos = len(multi_idx)
    p_global_at = p_global[flat_moves]  # (n_elements,)
    print(f"  {len(flat_moves):,} nonzero (position, move) entries")
    print(f"  extraction: {time.time()-t0:.1f}s")

    # Negative log-likelihood as function of log(κ)
    # L = sum_i [gammaln(κ) - gammaln(κ+N_i) + sum_j (gammaln(κ·p_j + c_j) - gammaln(κ·p_j))]
    def neg_ll(log_k):
        k = np.exp(log_k)
        # Per-position terms
        per_pos = gammaln(k) - gammaln(k + N_per_pos)  # (n_pos,)
        # Per-element terms
        alpha = k * p_global_at  # (n_elements,)
        per_elem = gammaln(alpha + flat_counts) - gammaln(alpha)  # (n_elements,)
        # Sum per_elem within each position via np.add.reduceat
        per_elem_sum = np.add.reduceat(per_elem, pos_starts[:-1])  # (n_pos,)
        total = per_pos.sum() + per_elem_sum.sum()
        return -total

    # Optimize in log-space, search over wide range
    print("\nOptimizing κ via Brent's method (1D minimization)...")
    t0 = time.time()
    # Quick scan for plausible region first
    log_ks = np.log(np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))
    nlls = np.array([neg_ll(lk) for lk in log_ks])
    print(f"  Initial scan:")
    for lk, nll in zip(log_ks, nlls):
        print(f"    κ={np.exp(lk):>10.4f}  -logL = {nll:>15,.1f}")
    # Bracket around minimum
    idx = np.argmin(nlls)
    lo_lk = log_ks[max(idx-1, 0)]
    hi_lk = log_ks[min(idx+1, len(log_ks)-1)]
    res = minimize_scalar(neg_ll, bracket=(lo_lk, log_ks[idx], hi_lk),
                          method='brent', options={'xtol': 1e-4})
    k_hat = float(np.exp(res.x))
    print(f"\n  FIT: κ = {k_hat:.4f}   -logL = {res.fun:,.1f}   ({time.time()-t0:.1f}s)")

    # Compute Bayes ceiling at fit κ
    print("\nComputing Bayes ceiling at fit κ...")
    t0 = time.time()
    ceiling_sum = 0.0

    # Singletons — closed-form contribution: max_a (κ·p_global[a] + 1{a=c_obs}) / (κ+1)
    # The max is always for a = c_obs unless κ·p_max > κ·p_obs + 1, i.e., p_max - p_obs > 1/κ.
    # For our κ < 1 probably, 1/κ > 1 > p_max, so always c_obs wins. Safe.
    is_singleton = (group_sizes == 1)
    singleton_start_idx = boundaries[:-1][is_singleton]
    singleton_y = sorted_y[singleton_start_idx]
    p_obs = p_global[singleton_y]
    global_max = p_global.max()
    obs_mean = (k_hat * p_obs + 1) / (k_hat + 1)
    global_mean = (k_hat * global_max) / (k_hat + 1)  # for unobserved classes
    ceiling_sum += np.maximum(obs_mean, global_mean).sum()

    # Multi-count: max_a (κ·p_global[a] + count[a]) / (κ + N)
    # Compute for each position: max of (κ·p_global_at + counts) element-wise
    # (over NONZERO entries only — but global max for zero-count moves also considered)
    alpha_flat = k_hat * p_global_at
    post_num_flat = alpha_flat + flat_counts
    # Max over nonzero entries per position
    max_nonzero = np.maximum.reduceat(post_num_flat, pos_starts[:-1])
    # For zero-count moves: post_num = κ·p_global[move] → max is κ·p_global.max()
    max_zero = k_hat * global_max
    max_per_pos = np.maximum(max_nonzero, max_zero)
    ceiling_per_pos = max_per_pos / (k_hat + N_per_pos)
    ceiling_sum += (ceiling_per_pos * N_per_pos).sum()

    ceiling = ceiling_sum / n * 100
    print(f"  ({time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "="*70)
    print(f"  EMPIRICAL BAYES FIT")
    print("="*70)
    print(f"  κ̂ (MLE)             : {k_hat:.4f}")
    print(f"  Interpretation      : prior worth {k_hat:.2f} pseudo-observations")
    print(f"  Bayes ceiling at κ̂  : {ceiling:.2f}%")
    print(f"  Our model           : 40.4%")
    print("="*70)

    # Context: previous κ sweep
    print("\n  For reference (from estimate_bayes_ceiling.py):")
    print(f"    κ=0 (MLE):      97.8%")
    print(f"    κ=1:            53.6%")
    print(f"    κ=10:           14.9%")
    print(f"    κ=50:            6.2%")
    print(f"    κ=1000:          2.1%")


if __name__ == "__main__":
    main()
