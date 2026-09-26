"""
Week 2 - LightGCN 實作
核心簡化（相對於一般 GCN）：不含非線性轉換、不含每層的權重矩陣，
只做「鄰居平均聚合」，最後把各層的 embedding 取平均當作最終表示。
參考: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_norm_adj(
    n_users: int, n_items: int, train_user: np.ndarray, train_item: np.ndarray
) -> torch.sparse.Tensor:
    """
    建立對稱正規化的 user-item 二部圖鄰接矩陣（LightGCN 論文式 (7) 的 A_hat）。
    整個 user-item 圖用 (n_users+n_items) x (n_users+n_items) 的稀疏矩陣表示：
        A = [[0, R], [R^T, 0]]
    正規化：A_hat[i,j] = A[i,j] / sqrt(deg(i) * deg(j))
    """
    n_nodes = n_users + n_items
    item_offset = train_item + n_users

    # 雙向邊：user->item 和 item->user
    rows = np.concatenate([train_user, item_offset])
    cols = np.concatenate([item_offset, train_user])

    degree = np.zeros(n_nodes, dtype=np.float64)
    for r in rows:
        degree[r] += 1
    degree[degree == 0] = 1.0  # 避免除以0（理論上過濾過後不該有孤立節點）
    d_inv_sqrt = 1.0 / np.sqrt(degree)

    values = d_inv_sqrt[rows] * d_inv_sqrt[cols]

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values_t, size=(n_nodes, n_nodes)).coalesce()
    return adj


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.sparse.Tensor | None = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.norm_adj = norm_adj  # 訓練前用 build_norm_adj() 建好後指定進來

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        多層鄰居聚合，回傳（最終）user / item embedding。
        LightGCN 核心公式： e^(k+1) = A_hat @ e^(k)，最終 e = mean(e^(0), ..., e^(K))
        """
        assert self.norm_adj is not None, "請先呼叫 model.norm_adj = build_norm_adj(...)"
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_layer_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_layer_embeddings.append(ego_embeddings)

        stacked = torch.stack(all_layer_embeddings, dim=0)  # (K+1, N, d)
        final_embeddings = torch.mean(stacked, dim=0)  # 論文式(3)：取各層平均

        final_user = final_embeddings[: self.n_users]
        final_item = final_embeddings[self.n_users :]
        return final_user, final_item

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        回傳 (user_score_pos, user_score_neg, l2_reg) 供 BPR loss 使用。
        同時回傳 forward 前的 embedding（第0層）供 L2 正則化（LightGCN 慣例只對
        第0層 ego embedding 做正則化，而不是最終聚合後的 embedding）。
        """
        final_user, final_item = self.propagate()

        u_emb = final_user[users]
        pos_emb = final_item[pos_items]
        neg_emb = final_item[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        # L2 正則化用第0層（未聚合前）的 embedding，這是LightGCN原論文的作法
        u_ego = self.user_embedding(users)
        pos_ego = self.item_embedding(pos_items)
        neg_ego = self.item_embedding(neg_items)
        l2_reg = (
            u_ego.norm(2).pow(2) + pos_ego.norm(2).pow(2) + neg_ego.norm(2).pow(2)
        ) / users.shape[0]

        return pos_scores, neg_scores, l2_reg

    @torch.no_grad()
    def get_all_scores(self) -> torch.Tensor:
        """回傳 (n_users, n_items) 的完整評分矩陣，供評估階段做 top-K ranking。"""
        final_user, final_item = self.propagate()
        return final_user @ final_item.t()


def bpr_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, l2_reg: torch.Tensor, weight_decay: float = 1e-4
) -> torch.Tensor:
    """BPR loss：-log(sigmoid(pos_score - neg_score))，加上L2正則化。"""
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
    reg_loss = weight_decay * l2_reg
    return loss + reg_loss
