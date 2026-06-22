import numpy as np
import pytest

from src.corpus_text import searchable_text
from src.dense_retrieval import DenseRetriever

CORPUS = [
    {
        "relevant_doc_tag": "A|Doc A",
        "relevant_article_tag": "A|Doc A|Điều 1",
        "doc_name": "Doc A",
        "article_id": "Điều 1",
        "text": "noi dung A",
    },
    {
        "relevant_doc_tag": "B|Doc B",
        "relevant_article_tag": "B|Doc B|Điều 2",
        "doc_name": "Doc B",
        "article_id": "Điều 2",
        "text": "noi dung B",
    },
    {
        "relevant_doc_tag": "C|Doc C",
        "relevant_article_tag": "C|Doc C|Điều 3",
        "doc_name": "Doc C",
        "article_id": "Điều 3",
        "text": "noi dung C",
    },
]

QUERY_A = "query about A"
QUERY_B = "query about B"
QUERY_C = "query about C"

VECTORS = {
    searchable_text(CORPUS[0]): [1.0, 0.0, 0.0],
    searchable_text(CORPUS[1]): [0.0, 1.0, 0.0],
    searchable_text(CORPUS[2]): [0.0, 0.0, 1.0],
    QUERY_A: [1.0, 0.0, 0.0],
    QUERY_B: [0.0, 1.0, 0.0],
    QUERY_C: [0.0, 0.0, 1.0],
}


class _StubEncoder:
    """Stands in for SentenceTransformer: returns pre-defined vectors keyed
    by exact text, so DenseRetriever can be unit tested without loading a
    real (network-dependent) embedding model."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.array(VECTORS[texts], dtype=np.float32)
        return np.array([VECTORS[t] for t in texts], dtype=np.float32)


def _build_retriever() -> DenseRetriever:
    return DenseRetriever(CORPUS, model=_StubEncoder())


def test_batch_search_returns_same_shape_as_search():
    retriever = _build_retriever()
    results = retriever.batch_search([QUERY_A], top_k=2)
    assert len(results) == 1
    assert len(results[0]) == 2
    assert "score" in results[0][0]
    assert isinstance(results[0][0]["score"], float)


def test_batch_search_ranks_each_query_independently():
    retriever = _build_retriever()
    results = retriever.batch_search([QUERY_A, QUERY_B, QUERY_C], top_k=1)
    assert results[0][0]["relevant_article_tag"] == "A|Doc A|Điều 1"
    assert results[1][0]["relevant_article_tag"] == "B|Doc B|Điều 2"
    assert results[2][0]["relevant_article_tag"] == "C|Doc C|Điều 3"


def test_batch_search_matches_per_query_search_exactly():
    """Same ranking whether queries go through search() one at a time or
    batch_search() all at once — proves the optimization doesn't change
    scores/ranking, only how the model is called."""
    retriever = _build_retriever()
    queries = [QUERY_A, QUERY_B, QUERY_C]

    per_query = [retriever.search(q, top_k=2) for q in queries]
    batched = retriever.batch_search(queries, top_k=2)

    per_query_tags = [[r["relevant_article_tag"] for r in lst] for lst in per_query]
    batched_tags = [[r["relevant_article_tag"] for r in lst] for lst in batched]
    assert per_query_tags == batched_tags

    per_query_scores = [[r["score"] for r in lst] for lst in per_query]
    batched_scores = [[r["score"] for r in lst] for lst in batched]
    for pq_row, b_row in zip(per_query_scores, batched_scores):
        assert pq_row == pytest.approx(b_row, abs=1e-6)


def test_batch_search_handles_empty_query_list():
    retriever = _build_retriever()
    assert retriever.batch_search([], top_k=2) == []


def test_batch_search_top_k_larger_than_corpus_returns_all():
    retriever = _build_retriever()
    results = retriever.batch_search([QUERY_A], top_k=100)
    assert len(results[0]) == len(CORPUS)
