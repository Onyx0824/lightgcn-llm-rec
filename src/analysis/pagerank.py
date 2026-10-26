"""Bipartite user-item graph PageRank (power iteration; same definition as networkx.pagerank(alpha=d)).
Used by scripts/prune_pagerank.py (Week 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def bipartite_adj(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Symmetric (n_users+n_items)^2 adjacency; users 0..n_users-1, items follow. Duplicates collapsed."""
    R = sp.coo_matrix((np.ones(len(df)), (df["user"].to_numpy(), df["item"].to_numpy())),
                      shape=(n_users, n_items)).tocsr()
    R.data[:] = 1.0
    top = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    bot = sp.hstack([R.T.tocsr(), sp.csr_matrix((n_items, n_items))])
    return sp.vstack([top, bot]).tocsr()


def pagerank(A: sp.csr_matrix, d: float = 0.85, tol: float = 1e-10, max_iter: int = 500) -> tuple[np.ndarray, int]:
    """Power iteration. Dangling nodes (no edges) spread their mass uniformly. Returns (pr, n_iter)."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = deg == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / deg[~dangling]
    PT = (sp.diags(inv) @ A).T.tocsr()
    pr = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        new = d * (PT @ pr + pr[dangling].sum() / n) + (1 - d) / n
        err = np.abs(new - pr).sum()
        pr = new
        if err < tol:
            break
    return pr / pr.sum(), it
