"""Smoke test: sentence-transformers can load the project's embedding model and encode a sentence.
Downloads the model on first run (needs network); skipped if sentence-transformers is not installed."""
import pytest

st = pytest.importorskip("sentence_transformers")


def test_embedding_shape():
    model = st.SentenceTransformer("BAAI/bge-small-zh-v1.5")
    emb = model.encode("測試句子")
    assert emb.shape == (512,)
