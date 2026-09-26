"""
Week 2 - 評估函式：Recall@K、NDCG@K
先用「全量 ranking」版本（對每個user，排除train中看過的item，
在剩下的item裡排名），這是LightGCN論文的標準做法，
資料規模不大（MovieLens-1M 過濾後 user/item 數量都在萬級以下）算起來不會太慢，
不需要一開始就退而求其次用抽樣負例的簡化版。
"""

from __future__ import annotations

import numpy as np
import torch


def mask_seen_interactions(
    scores: torch.Tensor, user_pos_items: dict[int, list[int]]
) -> torch.Tensor:
    """把 train（或不該出現在候選排名中）的 user-item pair 分數設為 -inf，避免洩漏。"""
    scores = scores.clone()
    for u, items in user_pos_items.items():
        if items:
            scores[u, items] = -float("inf")
    return scores


def recall_at_k(topk_items: np.ndarray, test_pos_items: dict[int, set[int]], k: int) -> float:
    """
    topk_items: shape (n_users, k)，每個user分數前k高的item index
    test_pos_items: {user_idx: set(item_idx)}，該user在test set中的正樣本
    """
    recalls = []
    for u, pos_set in test_pos_items.items():
        if not pos_set:
            continue
        hit = len(set(topk_items[u]) & pos_set)
        recalls.append(hit / len(pos_set))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(topk_items: np.ndarray, test_pos_items: dict[int, set[int]], k: int) -> float:
    """NDCG@K：命中位置越前面分數越高，除以理想排序(IDCG)做正規化。"""
    ndcgs = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 位置1對應log2(2)

    for u, pos_set in test_pos_items.items():
        if not pos_set:
            continue
        hits = np.array([1.0 if item in pos_set else 0.0 for item in topk_items[u]])
        dcg = np.sum(hits * discounts)
        ideal_hits = min(len(pos_set), k)
        idcg = np.sum(discounts[:ideal_hits])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


@torch.no_grad()
def evaluate(
    model,
    train_user_pos_items: dict[int, list[int]],
    eval_user_pos_items: dict[int, set[int]],
    k: int = 20,
    batch_size: int = 1024,
) -> dict[str, float]:
    """
    對整個 user 集合分批計算 top-K，回傳 {"recall@k":..., "ndcg@k":...}。
    分批是為了避免一次算 (n_users, n_items) 全量分數矩陣時記憶體爆掉
    （MovieLens-1M量級通常不需要分批也撐得住，但保留擴充空間）。
    """
    all_scores = model.get_all_scores()  # (n_users, n_items)
    all_scores = mask_seen_interactions(all_scores, train_user_pos_items)

    n_users = all_scores.shape[0]
    topk_items = np.zeros((n_users, k), dtype=np.int64)
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_scores = all_scores[start:end]
        _, batch_topk = torch.topk(batch_scores, k=k, dim=1)
        topk_items[start:end] = batch_topk.cpu().numpy()

    recall = recall_at_k(topk_items, eval_user_pos_items, k)
    ndcg = ndcg_at_k(topk_items, eval_user_pos_items, k)
    return {f"recall@{k}": recall, f"ndcg@{k}": ndcg}
