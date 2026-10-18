"""Week 5 PageRank group: drop LLM item embeddings of low-/high-PageRank items.

PageRank is computed on the TRAIN user-item bipartite graph (undirected, damping 0.85,
power iteration; items without train interactions are dangling nodes).
Items whose PageRank is among the lowest (or highest) q% of items that HAVE an LLM embedding
lose their augmentation (item_has=False -> they fall back to ID-only embeddings in LightGCN).
Same output layout as scripts/prune_items.py, so train_augmented.py works unchanged.

Because bipartite PageRank is very close to node degree, --score degree is provided as a control:
if degree-based selection behaves the same, the effect is "popularity", not "PageRank".
The random-drop control is the existing results/week5/rand10 / rand20 (prune_items.py --mode random).

Usage (project root):
    python scripts/prune_pagerank.py --drop low  --frac 0.10 --out_dir results/week5/prlow10
    python scripts/prune_pagerank.py --drop high --frac 0.10 --out_dir results/week5/prhigh10
    python scripts/prune_pagerank.py --score degree --drop low --frac 0.10 --out_dir results/week5/deglow10
    python scripts/prune_pagerank.py --verify --out_dir results/week5/_tmp     # compare with networkx
Train with:  --aug_side item --aug_dir <out_dir> --tag <tag>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def bipartite_adj(df, n_users, n_items):
    """Symmetric (n_users+n_items)^2 adjacency; users 0..n_users-1, items follow. Duplicates collapsed."""
    R = sp.coo_matrix((np.ones(len(df)), (df["user"].to_numpy(), df["item"].to_numpy())),
                      shape=(n_users, n_items)).tocsr()
    R.data[:] = 1.0
    top = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    bot = sp.hstack([R.T.tocsr(), sp.csr_matrix((n_items, n_items))])
    return sp.vstack([top, bot]).tocsr()


def pagerank(A, d=0.85, tol=1e-10, max_iter=500):
    """Power iteration; same definition as networkx.pagerank(alpha=d). Returns (pr, n_iter)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--src_dir", default="results/week4", help="unfiltered embeddings")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--score", choices=["pagerank", "degree"], default="pagerank")
    ap.add_argument("--drop", choices=["low", "high"], required=True,
                    help="low = remove augmentation of the lowest-score items; high = highest-score items")
    ap.add_argument("--frac", type=float, required=True, help="fraction of augmented items to drop, e.g. 0.10")
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--verify", action="store_true", help="compare PageRank with networkx (pip install networkx)")
    args = ap.parse_args()

    src, out = Path(args.src_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    has = np.load(src / "item_has.npy").astype(bool)
    n_items = len(has)
    df = pd.read_csv(args.train)
    n_users = int(df["user"].max()) + 1

    A = bipartite_adj(df, n_users, n_items)
    pr, n_iter = pagerank(A, d=args.damping)
    item_pr = pr[n_users:]
    item_deg = np.asarray(A.sum(axis=1)).ravel()[n_users:]
    if args.verify:
        import networkx as nx
        nxp = nx.pagerank(nx.from_scipy_sparse_array(A), alpha=args.damping, tol=1e-12, max_iter=500)
        diff = np.abs(np.array([nxp[k] for k in range(A.shape[0])]) - pr).max()
        print(f"[verify] max abs diff vs networkx = {diff:.2e}")

    score = item_pr if args.score == "pagerank" else item_deg.astype(float)
    pool = np.flatnonzero(has)
    k = int(round(args.frac * len(pool)))
    # ties broken by item id (stable, reproducible)
    order = pool[np.lexsort((pool, score[pool]))]          # ascending score
    drop = order[:k] if args.drop == "low" else order[::-1][:k]

    new_has = has.copy()
    new_has[drop] = False
    for name in ["user_emb.npy", "user_has.npy", "item_emb.npy", "order.json"]:
        shutil.copy(src / name, out / name)
    np.save(out / "item_has.npy", new_has)

    from scipy.stats import spearmanr
    rho = spearmanr(item_pr[pool], item_deg[pool])[0]
    pd.DataFrame({"item": np.arange(n_items), "pagerank": item_pr, "degree": item_deg.astype(int),
                  "has_emb": has, "dropped": ~new_has & has}).to_csv(out / "pagerank_items.csv", index=False)
    stats = {
        "score": args.score, "drop": args.drop, "frac": args.frac, "pagerank_iters": n_iter,
        "items_with_embedding": int(has.sum()), "dropped": int(len(drop)), "items_usable_after": int(new_has.sum()),
        "spearman_pagerank_vs_degree": round(float(rho), 4),
        "degree_dropped_mean": round(float(item_deg[drop].mean()), 2),
        "degree_dropped_max": int(item_deg[drop].max()), "degree_all_mean": round(float(item_deg[pool].mean()), 2),
        "share_of_train_interactions_on_dropped_items": round(float(item_deg[drop].sum() / item_deg.sum()), 4),
    }
    with open(out / "prune_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
