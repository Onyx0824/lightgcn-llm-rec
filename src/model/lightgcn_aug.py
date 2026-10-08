"""
Week 4 - LLM 增強版 LightGCN

做法（對照 LLMRec 的「把語意特徵接進 ID embedding」，但取最簡單的加權相加）：
    e0_user = id_embedding(u) + alpha * mask_u * W_u @ llm_emb_u
    e0_item = id_embedding(i) + alpha * mask_i * W_i @ llm_emb_i
之後的多層鄰居聚合與 baseline 完全相同。

- llm_emb 是凍結的（已 L2 normalize），只訓練線性投影 W（無 bias）。
- mask=0 的節點（沒生成 / 不在目前增強比例內）等同於 baseline，這讓 20%/50%/100% 消融直接成立。
- alpha=0 時模型與 baseline 數學上等價，可用來做 sanity check。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model.lightgcn import LightGCN


class LightGCNAug(LightGCN):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_feat: np.ndarray,
        item_feat: np.ndarray,
        user_mask: np.ndarray,
        item_mask: np.ndarray,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj=None,
        alpha: float = 1.0,
    ):
        super().__init__(n_users, n_items, embed_dim, n_layers, norm_adj)
        self.alpha = alpha
        self.register_buffer("user_feat", torch.as_tensor(user_feat, dtype=torch.float32))
        self.register_buffer("item_feat", torch.as_tensor(item_feat, dtype=torch.float32))
        self.register_buffer("user_mask", torch.as_tensor(user_mask, dtype=torch.float32).unsqueeze(1))
        self.register_buffer("item_mask", torch.as_tensor(item_mask, dtype=torch.float32).unsqueeze(1))
        self.user_proj = nn.Linear(user_feat.shape[1], embed_dim, bias=False)
        self.item_proj = nn.Linear(item_feat.shape[1], embed_dim, bias=False)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        u0 = self.user_embedding.weight + self.alpha * self.user_mask * self.user_proj(self.user_feat)
        i0 = self.item_embedding.weight + self.alpha * self.item_mask * self.item_proj(self.item_feat)
        ego = torch.cat([u0, i0], dim=0)
        layers = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            layers.append(ego)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]
