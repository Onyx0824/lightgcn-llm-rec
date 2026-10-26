"""Helpers for Week 7 stratified / error analyses: load saved top-K lists, hit vectors, degree-quantile tiers."""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_top(emb_dir, prefix: str, ratio: str, alpha: str, k: int = 20) -> dict[int, np.ndarray]:
    """{seed: (n_users, k) int array} from results/week7/emb/<prefix>_r<ratio>_a<alpha>_s<seed>_top<k>.npy."""
    out = {}
    for f in glob.glob(str(Path(emb_dir) / f"{prefix}_r{ratio}_a{alpha}_s*_top{k}.npy")):
        m = re.search(rf"_s(\d+)_top{k}\.npy$", f)
        if m:
            out[int(m.group(1))] = np.load(f)
    return out


def hit_vector(top: np.ndarray, tu: np.ndarray, ti: np.ndarray) -> np.ndarray:
    """For each test (user, item) row: is the item inside that user's top-K list?"""
    return (top[tu] == ti[:, None]).any(axis=1)


def quantile_tiers(deg: np.ndarray, pool: np.ndarray, n_tiers: int, n_total: int) -> np.ndarray:
    """Tier id 0..n_tiers-1 (0 = lowest degree) for ids in `pool`, -1 elsewhere. Ties broken by id (reproducible)."""
    tier = np.full(n_total, -1)
    rank = pd.Series(deg[pool]).rank(method="first")
    tier[pool] = pd.qcut(rank, n_tiers, labels=False).to_numpy()
    return tier


def tier_labels(n: int) -> list[str]:
    names = {3: ["cold", "middle", "hot"], 4: ["coldest", "cool", "warm", "hottest"]}
    return [f"T{k + 1} ({names[n][k]})" if n in names else f"T{k + 1}" for k in range(n)]


def user_recall(hit: np.ndarray, tu: np.ndarray, n_users: int, n_pos: np.ndarray) -> np.ndarray:
    """Per-user Recall@K = hits / number of test positives."""
    return np.bincount(tu, weights=hit.astype(float), minlength=n_users) / np.maximum(n_pos, 1)


def user_hit_rate(hit: np.ndarray, tu: np.ndarray, n_users: int, n_pos: np.ndarray, k: int = 20) -> np.ndarray:
    """Normalised hit rate = hits / min(n_test, K) (max 1); better than Recall for choosing individual cases."""
    return np.bincount(tu, weights=hit.astype(float), minlength=n_users) / np.maximum(np.minimum(n_pos, k), 1)
