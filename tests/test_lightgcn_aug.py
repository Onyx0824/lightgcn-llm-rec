"""alpha=0 must make LightGCNAug mathematically identical to plain LightGCN (the control-group assumption)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch = pytest.importorskip("torch")
from src.model.lightgcn import LightGCN, build_norm_adj  # noqa: E402
from src.model.lightgcn_aug import LightGCNAug  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    n_u, n_i = 12, 15
    u = np.repeat(np.arange(n_u), 4)
    it = np.concatenate([rng.choice(n_i, 4, replace=False) for _ in range(n_u)])
    return n_u, n_i, build_norm_adj(n_u, n_i, u, it)


def test_alpha0_equals_baseline():
    n_u, n_i, adj = _toy()
    torch.manual_seed(0)
    base = LightGCN(n_u, n_i, 8, 2, adj)
    torch.manual_seed(0)
    aug = LightGCNAug(n_u, n_i, np.random.rand(n_u, 5), np.random.rand(n_i, 5),
                      np.ones(n_u), np.ones(n_i), embed_dim=8, n_layers=2, norm_adj=adj, alpha=0.0)
    bu, bi = base.propagate()
    au, ai = aug.propagate()
    assert torch.allclose(bu, au, atol=1e-6) and torch.allclose(bi, ai, atol=1e-6)


def test_alpha_positive_changes_embeddings():
    n_u, n_i, adj = _toy()
    torch.manual_seed(0)
    aug = LightGCNAug(n_u, n_i, np.random.rand(n_u, 5), np.random.rand(n_i, 5),
                      np.ones(n_u), np.ones(n_i), embed_dim=8, n_layers=2, norm_adj=adj, alpha=1.0)
    torch.manual_seed(0)
    base = LightGCN(n_u, n_i, 8, 2, adj)
    assert not torch.allclose(aug.propagate()[1], base.propagate()[1])
