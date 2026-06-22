"""Dense embedding retrieval using a Vietnamese sentence-embedding model.

Model: bkai-foundation-models/vietnamese-bi-encoder — open-source
(Apache 2.0), well under the 14B parameter cap, released well before the
2026-03-01 cutoff, with prior benchmarks on Vietnamese legal retrieval data
(Zalo Legal Text Retrieval 2021). See docs/future_tuning_parameters.md for
alternative models considered.

Note: sentence-transformers silently truncates any input text past the
model's max sequence length. Long law articles are not chunked (see
docs/future_tuning_parameters.md), so very long articles are only
partially represented in their embedding here.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.corpus_text import searchable_text
from src.embedding_cache import compute_corpus_hash, load_cached_embeddings, save_embeddings_cache

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


class DenseRetriever:
    def __init__(
        self,
        corpus: list[dict],
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_embeddings_path: str | None = None,
        cache_meta_path: str | None = None,
        model=None,
    ):
        """`model` is an injection point for tests: any object exposing
        `.encode(texts, normalize_embeddings=True, show_progress_bar=False)`
        can be passed in to avoid loading the real (network-dependent)
        SentenceTransformer model. Production code never passes it."""
        self.corpus = corpus
        self._model = model if model is not None else SentenceTransformer(model_name)
        texts = [searchable_text(record) for record in corpus]

        cached = None
        if cache_embeddings_path and cache_meta_path:
            corpus_hash = compute_corpus_hash(texts)
            cached = load_cached_embeddings(
                cache_embeddings_path, cache_meta_path, model_name, len(corpus), corpus_hash
            )

        if cached is not None:
            print(f"[DenseRetriever] Cache HIT — loading embeddings from {cache_embeddings_path}")
            self._embeddings = cached
        else:
            print("[DenseRetriever] Cache MISS — encoding corpus from scratch...")
            self._embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            if cache_embeddings_path and cache_meta_path:
                save_embeddings_cache(
                    cache_embeddings_path,
                    cache_meta_path,
                    self._embeddings,
                    model_name,
                    len(corpus),
                    compute_corpus_hash(texts),
                )

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        query_embedding = self._model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        scores = self._embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)[:top_k]
        return self._records_for(scores, ranked_indices)

    def batch_search(self, queries: list[str], top_k: int = 15) -> list[list[dict]]:
        """Same ranking as calling `search()` once per query, but encodes all
        queries in a single SentenceTransformer batch call and scores them
        against the corpus as one matrix multiply, instead of one query
        embedding + one matrix-vector product per question. Pure speed
        optimization — same similarity scores, same ranking, same output
        shape as `search()`, just batched.
        """
        if not queries:
            return []

        query_embeddings = self._model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        scores_matrix = query_embeddings @ self._embeddings.T  # (n_queries, n_corpus)

        n_corpus = scores_matrix.shape[1]
        k = min(top_k, n_corpus)
        results = []
        for scores in scores_matrix:
            if k < n_corpus:
                candidate_indices = np.argpartition(-scores, k - 1)[:k]
            else:
                candidate_indices = np.arange(n_corpus)
            ranked_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
            results.append(self._records_for(scores, ranked_indices))
        return results

    def _records_for(self, scores: np.ndarray, ranked_indices: np.ndarray) -> list[dict]:
        results = []
        for idx in ranked_indices:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results
