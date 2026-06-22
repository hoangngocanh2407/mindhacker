"""Pure, model-free helpers for caching dense corpus embeddings to disk.

Kept separate from src/dense_retrieval.py so the cache validity logic
(hash comparison, shape checks) can be unit tested without loading a real
embedding model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def compute_corpus_hash(searchable_texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in searchable_texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def load_cached_embeddings(
    embeddings_path: str,
    meta_path: str,
    model_name: str,
    corpus_size: int,
    corpus_hash: str,
) -> np.ndarray | None:
    """Returns the cached embeddings array if the cache exists and its
    metadata matches the given model/corpus signature, else None."""
    embeddings_path_obj = Path(embeddings_path)
    meta_path_obj = Path(meta_path)
    if not embeddings_path_obj.exists() or not meta_path_obj.exists():
        return None

    meta = json.loads(meta_path_obj.read_text(encoding="utf-8"))
    if (
        meta.get("model_name") != model_name
        or meta.get("corpus_size") != corpus_size
        or meta.get("corpus_hash") != corpus_hash
    ):
        return None

    embeddings = np.load(embeddings_path_obj)
    if embeddings.shape[0] != corpus_size or embeddings.shape[1] != meta.get("embedding_dim"):
        return None
    return embeddings


def save_embeddings_cache(
    embeddings_path: str,
    meta_path: str,
    embeddings: np.ndarray,
    model_name: str,
    corpus_size: int,
    corpus_hash: str,
) -> None:
    embeddings_path_obj = Path(embeddings_path)
    meta_path_obj = Path(meta_path)
    embeddings_path_obj.parent.mkdir(parents=True, exist_ok=True)
    meta_path_obj.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path_obj, embeddings)
    meta_path_obj.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "corpus_size": corpus_size,
                "corpus_hash": corpus_hash,
                "embedding_dim": int(embeddings.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
