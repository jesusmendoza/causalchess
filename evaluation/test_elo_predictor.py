#!/usr/bin/env python3
import sys as _sys, os as _os
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PARENT)
_os.chdir(_PARENT)
"""
test_elo_predictor.py — Predict player Elo from a bag of K positions.

The ONE clear-validation downstream application from the paper: given K
positions played by a player, predict their Elo rating. Ground truth
is the player's actual rating from Lichess/Chess.com headers.

Setup:
  - Input: for each player, K=40 positions (random sample from their games).
  - Features: each position's 256-d embedding, averaged across K → 256-d
    player vector. (Could also try concat, but average is cleaner.)
  - Target: mean Elo across the collected positions.
  - Train regressor (Ridge or small MLP) on 80% of players, val on 20%.
  - Baseline: material-feature-average (6-d × K → average 6-d vector).

Requires `data_elo/` produced by extract_elo_dataset.py.

Usage:
    python3 test_elo_predictor.py --ckpt models/prev_move_cnn_v2.pt \\
        --bin-dir data_elo --k 40
"""
import argparse
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from prev_move_models import create_model
from prev_move_train import unpack_bitboards_gpu


def load_elo_data(bin_dir):
    with open(os.path.join(bin_dir, "chess_meta.txt")) as f:
        meta = dict(line.strip().split('=') for line in f if '=' in line)
    n = int(meta['n_samples'])
    n_players = int(meta['n_players'])
    x = np.memmap(os.path.join(bin_dir, "chess_x.bin"),
                  dtype=np.int64, mode='r', shape=(n, 12))
    p = np.memmap(os.path.join(bin_dir, "chess_player_id.bin"),
                  dtype=np.uint32, mode='r', shape=(n,))
    e = np.memmap(os.path.join(bin_dir, "chess_elo.bin"),
                  dtype=np.float32, mode='r', shape=(n,))
    return np.asarray(x), np.asarray(p), np.asarray(e), n_players


def compute_embeddings_gpu(model, x_packed, batch=1024):
    """Compute 256-d embeddings for all positions, on GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    n = len(x_packed)
    embs = np.empty((n, 256), dtype=np.float32)
    for i in range(0, n, batch):
        xb = torch.from_numpy(x_packed[i:i+batch]).to(device)
        xb_float = unpack_bitboards_gpu(xb)
        with torch.no_grad():
            embs[i:i+batch] = model.get_embedding(xb_float).cpu().numpy()
    return embs


def material_features_from_bitboards(x_packed):
    """Vectorized material features per position from packed uint64 bitboards.
    Returns (N, 6): piece_count, mat_total, mat_w, mat_b, mat_balance, phase_id
    """
    # popcount: unpack bytes and sum bits
    x_bytes = np.ascontiguousarray(x_packed).view(np.uint8).reshape(-1, 12, 8)
    bits = np.unpackbits(x_bytes, axis=-1).sum(axis=-1)  # (N, 12) counts per piece type
    # Layout: [P,N,B,R,Q,K, p,n,b,r,q,k]
    values = np.array([1, 3, 3, 5, 9, 0,  1, 3, 3, 5, 9, 0], dtype=np.float32)
    mat_total = (bits * values).sum(axis=1)
    mat_w = (bits[:, :6] * values[:6]).sum(axis=1)
    mat_b = (bits[:, 6:] * values[6:]).sum(axis=1)
    mat_bal = mat_w - mat_b
    piece_count = bits.sum(axis=1)
    phase = np.where(mat_total >= 70, 0, np.where(mat_total >= 40, 1, 2)).astype(np.float32)
    return np.stack([piece_count, mat_total, mat_w, mat_b, mat_bal, phase], axis=1)


def aggregate_per_player(features, player_ids):
    """For each player, mean feature across their K positions."""
    n_players = int(player_ids.max()) + 1
    agg = np.zeros((n_players, features.shape[1]), dtype=np.float32)
    counts = np.zeros(n_players, dtype=np.float32)
    for i in range(len(player_ids)):
        pid = int(player_ids[i])
        agg[pid] += features[i]
        counts[pid] += 1
    return agg / np.maximum(counts, 1)[:, None]


def fit_ridge(X_tr, y_tr, X_val, y_val, label):
    clf = Ridge(alpha=1.0)
    clf.fit(X_tr, y_tr)
    mae_val = mean_absolute_error(y_val, clf.predict(X_val))
    r2_val = r2_score(y_val, clf.predict(X_val))
    print(f"  {label:<38} R²={r2_val:+.4f}  MAE={mae_val:.1f} Elo")
    return r2_val, mae_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/prev_move_cnn_v2.pt")
    ap.add_argument("--bin-dir", default="data_elo")
    args = ap.parse_args()

    if not os.path.isdir(args.bin_dir):
        print(f"ERROR: {args.bin_dir} not found. Run extract_elo_dataset.py first.")
        return

    print(f"Loading Elo dataset from {args.bin_dir}...")
    x_packed, player_ids, elos, n_players = load_elo_data(args.bin_dir)
    n = len(x_packed)
    print(f"  {n} positions from {n_players} players")
    print(f"  Elo range: [{elos.min():.0f}, {elos.max():.0f}] mean={elos.mean():.0f}")

    # Per-player mean Elo (label)
    player_elo = np.zeros(n_players, dtype=np.float32)
    player_count = np.zeros(n_players, dtype=np.float32)
    for i in range(n):
        pid = int(player_ids[i])
        player_elo[pid] += elos[i]
        player_count[pid] += 1
    player_elo = player_elo / np.maximum(player_count, 1)

    # Material features per position → aggregate per player
    print("\nComputing material features (per position)...")
    mat_feats = material_features_from_bitboards(x_packed)
    mat_player = aggregate_per_player(mat_feats, player_ids)

    # Embedding features per position → aggregate per player
    print(f"\nLoading {args.ckpt} and computing embeddings...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get('arch', 'cnn')
    model = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    model.load_state_dict(ckpt['model'])
    emb_feats = compute_embeddings_gpu(model, x_packed)
    emb_player = aggregate_per_player(emb_feats, player_ids)

    # Untrained control
    print("\nComputing untrained control embeddings...")
    torch.manual_seed(42)
    untrained = create_model(arch, ckpt['n_moves'], embed_dim=ckpt['embed_dim'])
    emb_u_feats = compute_embeddings_gpu(untrained, x_packed)
    emb_u_player = aggregate_per_player(emb_u_feats, player_ids)

    # Train/val split at PLAYER level
    idx = np.arange(n_players)
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=42)
    y_tr, y_val = player_elo[idx_tr], player_elo[idx_val]

    print("\n" + "=" * 80)
    print("  Elo prediction (per-player, train 80% / val 20%)")
    print("=" * 80)
    fit_ridge(mat_player[idx_tr], y_tr, mat_player[idx_val], y_val,
              "Material mean (6-d)")
    fit_ridge(emb_u_player[idx_tr], y_tr, emb_u_player[idx_val], y_val,
              "Untrained embedding mean (256-d)")
    fit_ridge(emb_player[idx_tr], y_tr, emb_player[idx_val], y_val,
              f"Trained embedding mean ({arch}, 256-d)")
    fit_ridge(np.concatenate([mat_player, emb_player], axis=1)[idx_tr], y_tr,
              np.concatenate([mat_player, emb_player], axis=1)[idx_val], y_val,
              "Material + trained embedding (262-d)")
    print("=" * 80)
    print(f"\n  Elo std (val): {y_val.std():.0f}")
    print(f"  Elo range: [{y_val.min():.0f}, {y_val.max():.0f}]")


if __name__ == "__main__":
    main()
