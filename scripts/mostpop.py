"""
Step A - MostPop 基準：對每位 user 推薦「訓練集中最熱門、且該 user 在 train∪valid 沒互動過」的 20 部。
評估協定與 LightGCN 完全相同（沿用 src/evaluation/metrics.py 的 recall_at_k / ndcg_at_k）。
MostPop 是確定性的，不需要多 seed。結果存 results/mostpop_results.json。

用法（專案根目錄）：
    python scripts/mostpop.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_baseline import build_user_pos_items  # noqa: E402
from src.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402

K = 20


def main():
    d = ROOT / "data" / "processed"
    train, valid, test = (pd.read_csv(d / f"{n}.csv") for n in ("train", "valid", "test"))
    with open(d / "id_maps.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    n_users, n_items = len(m["user2idx"]), len(m["item2idx"])

    deg = np.bincount(train["item"].values, minlength=n_items).astype(np.float64)
    train_pos = build_user_pos_items(train, n_users)
    valid_pos = build_user_pos_items(valid, n_users)
    test_pos = {u: set(v) for u, v in build_user_pos_items(test, n_users).items()}

    topk = np.zeros((n_users, K), dtype=np.int64)
    for u in range(n_users):
        s = deg.copy()
        seen = list(train_pos.get(u, [])) + list(valid_pos.get(u, []))
        if seen:
            s[seen] = -np.inf
        topk[u] = np.argsort(-s, kind="stable")[:K]

    res = {"model": "MostPop", f"recall@{K}": recall_at_k(topk, test_pos, K),
           f"ndcg@{K}": ndcg_at_k(topk, test_pos, K),
           "n_test_users": int(sum(1 for s in test_pos.values() if s))}
    print(res)
    out = ROOT / "results" / "mostpop_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"已存到 {out}")


if __name__ == "__main__":
    main()
