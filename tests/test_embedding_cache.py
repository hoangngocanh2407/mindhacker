import numpy as np

from src.embedding_cache import compute_corpus_hash, load_cached_embeddings, save_embeddings_cache


def test_compute_corpus_hash_differs_when_text_differs():
    h1 = compute_corpus_hash(["a", "b", "c"])
    h2 = compute_corpus_hash(["a", "b", "d"])
    assert h1 != h2


def test_compute_corpus_hash_stable_for_same_input():
    assert compute_corpus_hash(["a", "b"]) == compute_corpus_hash(["a", "b"])


def test_save_then_load_cached_embeddings_roundtrip(tmp_path):
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    emb_path = tmp_path / "emb.npy"
    meta_path = tmp_path / "meta.json"
    corpus_hash = compute_corpus_hash(["doc one", "doc two"])

    save_embeddings_cache(str(emb_path), str(meta_path), embeddings, "model-x", 2, corpus_hash)
    loaded = load_cached_embeddings(str(emb_path), str(meta_path), "model-x", 2, corpus_hash)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, embeddings)


def test_load_cached_embeddings_returns_none_when_files_missing(tmp_path):
    result = load_cached_embeddings(
        str(tmp_path / "missing.npy"), str(tmp_path / "missing.json"), "model-x", 2, "somehash"
    )
    assert result is None


def test_load_cached_embeddings_returns_none_when_model_name_differs(tmp_path):
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    emb_path = tmp_path / "emb.npy"
    meta_path = tmp_path / "meta.json"
    corpus_hash = compute_corpus_hash(["doc one"])
    save_embeddings_cache(str(emb_path), str(meta_path), embeddings, "model-x", 1, corpus_hash)

    result = load_cached_embeddings(str(emb_path), str(meta_path), "model-y", 1, corpus_hash)
    assert result is None


def test_load_cached_embeddings_returns_none_when_corpus_hash_differs(tmp_path):
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    emb_path = tmp_path / "emb.npy"
    meta_path = tmp_path / "meta.json"
    save_embeddings_cache(
        str(emb_path), str(meta_path), embeddings, "model-x", 1, compute_corpus_hash(["doc one"])
    )

    result = load_cached_embeddings(
        str(emb_path), str(meta_path), "model-x", 1, compute_corpus_hash(["doc two, changed"])
    )
    assert result is None


def test_load_cached_embeddings_returns_none_when_corpus_size_differs(tmp_path):
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    emb_path = tmp_path / "emb.npy"
    meta_path = tmp_path / "meta.json"
    corpus_hash = compute_corpus_hash(["doc one"])
    save_embeddings_cache(str(emb_path), str(meta_path), embeddings, "model-x", 1, corpus_hash)

    result = load_cached_embeddings(str(emb_path), str(meta_path), "model-x", 2, corpus_hash)
    assert result is None
