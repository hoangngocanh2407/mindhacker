"""Cross-encoder reranker (GPU). Runs on Kaggle/Colab, not the local CPU box.

Model: BAAI/bge-reranker-v2-m3 — open-source, multilingual (incl. Vietnamese),
~568M params (well under the 14B cap), released 2024 (before the 2026-03-01
cutoff). A cross-encoder scores each (query, article) PAIR jointly, which is
far more accurate than bi-encoder/BM25 ranking but too slow on CPU — hence GPU.

Usage: BM25 (optionally unioned with dense) produces a wide candidate pool per
question; the reranker re-scores that pool and the top_k_final survivors become
the answer. The reranker can only promote articles that are IN the candidate
pool, so the pool should be wide (e.g. BM25 top-50) to lift the recall ceiling.
"""
from src.corpus_text import searchable_text


# Cross-encoder attention is O(seq_len^2). Legal articles run up to ~245k
# chars; feeding them whole blows up GPU memory (observed: 8 GiB single alloc
# OOM on a 16 GiB T4). Cap the sequence length and pre-truncate the candidate
# text so memory is bounded regardless of article size. 512 tokens is the
# standard passage-reranking window for bge-reranker and captures doc_name +
# article_id + the start of the article (the most query-relevant part).
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16
CANDIDATE_CHAR_CAP = 2000


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model=None,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH_SIZE,
    ):
        """`model` is an injection point for tests: any object with a
        `.predict(pairs, **kwargs) -> list[float]` works, so the reranker can be
        unit tested without loading the real (GPU/network) cross-encoder."""
        self._batch_size = batch_size
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """Re-score `candidates` (corpus-record dicts) against `query` with the
        cross-encoder and return the top_k, each with a `rerank_score` key,
        sorted by score descending. Empty pool -> empty list."""
        if not candidates:
            return []
        pairs = [[query, searchable_text(c)[:CANDIDATE_CHAR_CAP]] for c in candidates]
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        results = []
        for candidate, score in ranked[:top_k]:
            record = dict(candidate)
            record["rerank_score"] = float(score)
            results.append(record)
        return results
