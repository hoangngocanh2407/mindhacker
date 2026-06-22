import json

from src.export import build_result_entry, write_results
from src.fusion import reciprocal_rank_fusion
from src.retrieval import BM25Retriever

QUESTION_TEXT_KEYS = ("question", "query", "question_text", "content")


def get_question_text(record: dict) -> str:
    for key in QUESTION_TEXT_KEYS:
        value = record.get(key)
        if value:
            return value
    raise ValueError(f"Cannot find question text field in record id={record.get('id')}")


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    questions_path: str,
    clean_corpus_path: str,
    output_path: str,
    top_k_retrieve: int = 15,
    top_k_final: int = 5,
    limit: int | None = None,
    dense_retriever=None,
    rrf_k: int = 60,
) -> list[dict]:
    """If `dense_retriever` is given (any object with a `.search(query, top_k)`
    method returning the same shape as BM25Retriever.search), its results are
    fused with BM25 via Reciprocal Rank Fusion. If omitted, behavior is
    identical to Phase 1 — BM25-only retrieval.
    """
    questions = _load_json(questions_path)
    if limit is not None:
        questions = questions[:limit]

    corpus = _load_json(clean_corpus_path)
    bm25_retriever = BM25Retriever(corpus)

    entries = []
    for question in questions:
        query = get_question_text(question)
        bm25_results = bm25_retriever.search(query, top_k=top_k_retrieve)

        if dense_retriever is not None:
            dense_results = dense_retriever.search(query, top_k=top_k_retrieve)
            top_articles = reciprocal_rank_fusion(
                [bm25_results, dense_results], k=rrf_k, top_k=top_k_retrieve
            )
        else:
            top_articles = bm25_results

        entries.append(build_result_entry(question["id"], top_articles, top_k_final=top_k_final))

    write_results(entries, output_path)
    return entries
