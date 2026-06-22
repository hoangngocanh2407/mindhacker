from src.reranker import Reranker

CANDIDATES = [
    {"relevant_article_tag": "A|Doc A|Điều 1", "doc_name": "Doc A", "article_id": "Điều 1", "text": "alpha"},
    {"relevant_article_tag": "B|Doc B|Điều 2", "doc_name": "Doc B", "article_id": "Điều 2", "text": "beta"},
    {"relevant_article_tag": "C|Doc C|Điều 3", "doc_name": "Doc C", "article_id": "Điều 3", "text": "gamma"},
]


class _StubCrossEncoder:
    """Returns a fixed score per pair based on which body keyword appears in the
    paired text (the reranker passes searchable_text, e.g. 'Doc A Điều 1
    alpha'), so the reranker can be tested without the real GPU model. Scores
    chosen so 'gamma' > 'alpha' > 'beta'."""

    _SCORES = {"alpha": 0.5, "beta": 0.1, "gamma": 0.9}

    def predict(self, pairs):
        scores = []
        for _query, text in pairs:
            scores.append(next(s for kw, s in self._SCORES.items() if kw in text))
        return scores


def _reranker() -> Reranker:
    return Reranker(model=_StubCrossEncoder())


def test_rerank_orders_by_cross_encoder_score():
    out = _reranker().rerank("q", CANDIDATES, top_k=3)
    tags = [r["relevant_article_tag"] for r in out]
    assert tags == ["C|Doc C|Điều 3", "A|Doc A|Điều 1", "B|Doc B|Điều 2"]


def test_rerank_respects_top_k():
    out = _reranker().rerank("q", CANDIDATES, top_k=1)
    assert len(out) == 1
    assert out[0]["relevant_article_tag"] == "C|Doc C|Điều 3"


def test_rerank_attaches_rerank_score():
    out = _reranker().rerank("q", CANDIDATES, top_k=1)
    assert "rerank_score" in out[0]
    assert out[0]["rerank_score"] == 0.9


def test_rerank_empty_candidates_returns_empty():
    assert _reranker().rerank("q", [], top_k=3) == []
