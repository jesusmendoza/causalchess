"""Scale-aware descriptive metrics for embedding trajectories."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def summarize_trajectory(embeddings: np.ndarray) -> dict:
    """Report raw distances and per-position unit-L2 chord distances.

    Unit-L2 distances are Euclidean chord lengths on the unit sphere and lie
    in [0, 2]. They are a scale-control, not a cross-model alignment.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] < 2:
        raise ValueError("embeddings must be a 2-D array with at least two rows")
    if not np.all(np.isfinite(emb)):
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(emb, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("cannot L2-normalize an embedding with near-zero norm")
    unit = emb / norms[:, None]

    def summary(values: np.ndarray, drift: float) -> dict:
        return {
            "consecutive_mean": float(values.mean()),
            "consecutive_std_population": float(values.std()),
            "consecutive_n": int(values.size),
            "final_drift": float(drift),
        }

    raw_steps = np.linalg.norm(np.diff(emb, axis=0), axis=1)
    unit_steps = np.linalg.norm(np.diff(unit, axis=0), axis=1)
    raw_drift = float(np.linalg.norm(emb[-1] - emb[0]))
    unit_drift = float(np.linalg.norm(unit[-1] - unit[0]))
    return {
        "raw_embedding_units": summary(raw_steps, raw_drift),
        "per_position_unit_l2_chord": summary(unit_steps, unit_drift),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std_population": float(norms.std()),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
