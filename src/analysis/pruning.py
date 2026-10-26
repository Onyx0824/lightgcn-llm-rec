"""Semantic-outlier scoring of LLM item embeddings (Week 5 'D group'). Used by scripts/prune_items.py."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def genre_list(g) -> list[str]:
    if isinstance(g, str):
        return [x for x in g.split("|") if x]
    return list(g or [])


def consistency_scores(emb: np.ndarray, has: np.ndarray, genres: list[list[str]], min_group: int = 3) -> np.ndarray:
    """Mean cosine to the leave-one-out genre centroid(s). NaN = not scorable (never pruned). Lower = more outlying."""
    ids = np.flatnonzero(has)
    sums, cnts = {}, defaultdict(int)
    for i in ids:
        for g in genres[i]:
            sums[g] = sums.get(g, 0) + emb[i]
            cnts[g] += 1
    scores = np.full(len(emb), np.nan, dtype=np.float64)
    for i in ids:
        vals = []
        for g in genres[i]:
            if cnts[g] < min_group:
                continue
            c = sums[g] - emb[i]
            n = np.linalg.norm(c)
            if n > 0:
                vals.append(float(emb[i] @ c / n))
        if vals:
            scores[i] = float(np.mean(vals))
    return scores
