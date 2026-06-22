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
    ):
        self.corpus = corpus
        self._model = SentenceTransformer(model_name)
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
        results = []
        for idx in ranked_indices:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results
