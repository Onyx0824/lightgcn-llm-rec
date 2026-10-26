"""Embedding-space diagnostics used by scripts/visualize_embeddings.py."""
from __future__ import annotations

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_purity(x: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean share of each item's k nearest neighbours (cosine) that carry the same label."""
    x = l2(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return float((labels[idx] == labels[:, None]).mean())
