"""Unit tests for Recall@K / NDCG@K (run from project root:  python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch = pytest.importorskip("torch")
from src.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402


def test_recall_partial_hit():
    top = np.array([[1, 2, 3]])
    assert recall_at_k(top, {0: {2, 9}}, k=3) == pytest.approx(0.5)


def test_recall_perfect_and_zero():
    top = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(top, {0: {1, 2, 3}, 1: {9}}, k=3) == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_ndcg_position_matters():
    first = ndcg_at_k(np.array([[7, 1, 2]]), {0: {7}}, k=3)
    second = ndcg_at_k(np.array([[1, 7, 2]]), {0: {7}}, k=3)
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1 / np.log2(3))
    assert first > second


def test_ndcg_two_positives_one_hit_at_rank2():
    val = ndcg_at_k(np.array([[1, 2, 3]]), {0: {2, 9}}, k=3)
    dcg = 1 / np.log2(3)
    idcg = 1 + 1 / np.log2(3)
    assert val == pytest.approx(dcg / idcg)
