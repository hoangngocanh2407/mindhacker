import json

from src.export import build_result_entry, write_results
from src.fusion import reciprocal_rank_fusion
from src.query_processing import expand_query
from src.retrieval import BM25Retriever, segment_tokenize, tokenize

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


def _dedupe_by_tag(records: list[dict]) -> list[dict]:
    """Unique candidate records by relevant_article_tag, first-seen order.
    Order doesn't matter for reranking (the reranker re-scores everything),
    only uniqueness does."""
    seen = set()
    unique = []
    for record in records:
        tag = record["relevant_article_tag"]
        if tag not in seen:
            seen.add(tag)
            unique.append(record)
    return unique


def run_pipeline(
    questions_path: str,
    clean_corpus_path: str,
    output_path: str,
    top_k_retrieve: int = 15,
    top_k_final: int = 5,
    limit: int | None = None,
    dense_retriever=None,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    expand_abbreviations: bool = False,
    use_segmentation: bool = False,
    reranker=None,
) -> list[dict]:
    """If `dense_retriever` is given (any object with a `.search(query, top_k)`
    method returning the same shape as BM25Retriever.search), its results are
    fused with BM25 via Reciprocal Rank Fusion. If omitted, behavior is
    identical to Phase 1 — BM25-only retrieval.

    If `dense_retriever` also exposes `.batch_search(queries, top_k)`, it is
    used to encode/score all questions in one batch instead of one
    `.search()` call per question — pure speed optimization, same ranking
    (see tests/test_dense_retrieval.py for the equivalence check), no change
    to fusion or output.

    `dense_weight` (< 1.0) lowers dense's influence in the fusion; BM25 is
    fixed at weight 1.0. Default 1.0 = equal-weight RRF (Phase 2 behavior).

    `expand_abbreviations` (Phase 4): if True, legal abbreviations in each
    question (TNHH, GTGT...) are expanded to their full form before retrieval,
    applied once to the shared query text fed to both BM25 and dense. Default
    False = no expansion (unchanged behavior).

    `use_segmentation`: if True, BM25 (index + query) uses the Vietnamese
    word-segmenting tokenizer instead of the plain regex one. Affects BM25
    only — dense uses its own subword tokenizer. Default False.

    `reranker` (GPU): if given (object with `.rerank(query, candidates, top_k)`),
    it takes precedence over RRF fusion. The candidate pool is BM25's
    top_k_retrieve, unioned with dense's if a dense_retriever is also given
    (wider pool = higher recall ceiling), deduped by relevant_article_tag; the
    reranker re-scores it and the top results become the answer. Use a large
    top_k_retrieve (e.g. 50) to give the reranker room to recover correct
    articles BM25 ranked low. If the reranker also exposes
    `.rerank_batch(queries, candidate_lists, top_k)`, all questions are scored
    in one cross-encoder call (faster + a single progress bar) — same ranking.
    """
    questions = _load_json(questions_path)
    if limit is not None:
        questions = questions[:limit]

    corpus = _load_json(clean_corpus_path)
    bm25_tokenizer = segment_tokenize if use_segmentation else tokenize
    bm25_retriever = BM25Retriever(corpus, tokenizer=bm25_tokenizer)
    query_texts = [get_question_text(question) for question in questions]
    if expand_abbreviations:
        query_texts = [expand_query(text) for text in query_texts]

    dense_results_by_index = None
    if dense_retriever is not None and hasattr(dense_retriever, "batch_search"):
        dense_results_by_index = dense_retriever.batch_search(query_texts, top_k=top_k_retrieve)

    bm25_by_index = [bm25_retriever.search(q, top_k=top_k_retrieve) for q in query_texts]

    dense_by_index = [None] * len(questions)
    if dense_retriever is not None:
        for index in range(len(questions)):
            if dense_results_by_index is not None:
                dense_by_index[index] = dense_results_by_index[index]
            else:
                dense_by_index[index] = dense_retriever.search(query_texts[index], top_k=top_k_retrieve)

    if reranker is not None:
        candidate_pools = []
        for index in range(len(questions)):
            candidates = bm25_by_index[index]
            if dense_by_index[index] is not None:
                candidates = _dedupe_by_tag(bm25_by_index[index] + dense_by_index[index])
            candidate_pools.append(candidates)

        if hasattr(reranker, "rerank_batch"):
            top_articles_by_index = reranker.rerank_batch(
                query_texts, candidate_pools, top_k=top_k_retrieve
            )
        else:
            top_articles_by_index = [
                reranker.rerank(query_texts[i], candidate_pools[i], top_k=top_k_retrieve)
                for i in range(len(questions))
            ]
    else:
        top_articles_by_index = []
        for index in range(len(questions)):
            if dense_by_index[index] is not None:
                top_articles_by_index.append(
                    reciprocal_rank_fusion(
                        [bm25_by_index[index], dense_by_index[index]],
                        k=rrf_k,
                        top_k=top_k_retrieve,
                        weights=[1.0, dense_weight],
                    )
                )
            else:
                top_articles_by_index.append(bm25_by_index[index])

    entries = [
        build_result_entry(question["id"], top_articles_by_index[index], top_k_final=top_k_final)
        for index, question in enumerate(questions)
    ]

    write_results(entries, output_path)
    return entries
