import json

import pytest

from src.pipeline import get_question_text, run_pipeline
from src.validate import validate_entry

CORPUS = [
    {
        "relevant_doc_tag": "A|Doc A",
        "relevant_article_tag": "A|Doc A|Điều 1",
        "doc_name": "Doc A",
        "article_id": "Điều 1",
        "text": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu.",
    },
    {
        "relevant_doc_tag": "B|Doc B",
        "relevant_article_tag": "B|Doc B|Điều 2",
        "doc_name": "Doc B",
        "article_id": "Điều 2",
        "text": "Quy định về hợp đồng lao động và xử lý vi phạm kỷ luật.",
    },
]


def test_get_question_text_supports_question_key():
    assert get_question_text({"id": 1, "question": "Câu hỏi mẫu?"}) == "Câu hỏi mẫu?"


def test_get_question_text_supports_query_key():
    assert get_question_text({"id": 1, "query": "Câu hỏi mẫu?"}) == "Câu hỏi mẫu?"


def test_get_question_text_raises_value_error_when_no_known_key():
    with pytest.raises(ValueError, match="Cannot find question text field in record id=7"):
        get_question_text({"id": 7, "content": ""})


def test_run_pipeline_produces_valid_entries_with_question_key(tmp_path):
    questions = [
        {"id": 1, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"},
        {"id": 2, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1)

    assert len(entries) == 2
    for entry in entries:
        assert validate_entry(entry) == []
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == entries


def test_run_pipeline_produces_valid_entries_with_query_key(tmp_path):
    questions = [
        {"id": 1, "query": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"},
        {"id": 2, "query": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1)

    assert len(entries) == 2


def test_run_pipeline_raises_value_error_on_unrecognized_question_schema(tmp_path):
    questions = [{"id": 1, "unexpected_field": "schema mismatch"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot find question text field in record id=1"):
        run_pipeline(str(q_path), str(c_path), str(out_path))


class _StubDenseRetriever:
    """Duck-types BM25Retriever/DenseRetriever's .search() without loading a
    real embedding model, so fusion can be tested without network access."""

    def __init__(self, fixed_results: list[dict]):
        self._fixed_results = fixed_results

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        return self._fixed_results[:top_k]


def test_run_pipeline_fuses_dense_results_when_dense_retriever_given(tmp_path):
    questions = [{"id": 1, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    # Dense retriever "votes" for the article BM25 would otherwise rank lower.
    stub_dense = _StubDenseRetriever([CORPUS[0]])

    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=2,
        dense_retriever=stub_dense,
    )

    assert len(entries) == 1
    assert validate_entry(entries[0]) == []
    assert "A|Doc A|Điều 1" in entries[0]["relevant_articles"]


def test_run_pipeline_accepts_dense_weight_and_produces_valid_entries(tmp_path):
    questions = [{"id": 1, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    stub_dense = _StubDenseRetriever([CORPUS[0]])

    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=2,
        dense_retriever=stub_dense, dense_weight=0.0,
    )

    assert len(entries) == 1
    assert validate_entry(entries[0]) == []


class _StubBatchDenseRetriever:
    """Duck-types a DenseRetriever that also supports batch_search(), to
    verify run_pipeline prefers the batch path when available."""

    def __init__(self, fixed_results: list[dict]):
        self._fixed_results = fixed_results
        self.search_calls = 0
        self.batch_search_calls = 0

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        self.search_calls += 1
        return self._fixed_results[:top_k]

    def batch_search(self, queries: list[str], top_k: int = 15) -> list[list[dict]]:
        self.batch_search_calls += 1
        return [self._fixed_results[:top_k] for _ in queries]


def test_run_pipeline_uses_batch_search_when_dense_retriever_supports_it(tmp_path):
    questions = [
        {"id": 1, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
        {"id": 2, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"},
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    stub_dense = _StubBatchDenseRetriever([CORPUS[0]])

    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=2,
        dense_retriever=stub_dense,
    )

    assert len(entries) == 2
    assert stub_dense.batch_search_calls == 1
    assert stub_dense.search_calls == 0
    for entry in entries:
        assert validate_entry(entry) == []


class _StubReranker:
    """Reranks by a fixed preferred tag order, no model needed."""

    def __init__(self, preferred_tag: str):
        self.preferred_tag = preferred_tag
        self.seen_candidate_counts = []

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        self.seen_candidate_counts.append(len(candidates))
        ordered = sorted(
            candidates, key=lambda c: 0 if c["relevant_article_tag"] == self.preferred_tag else 1
        )
        return ordered[:top_k]


class _StubBatchReranker:
    """Reranker exposing rerank_batch, to verify run_pipeline prefers the
    batched path (one call) over per-question rerank."""

    def __init__(self, preferred_tag: str):
        self.preferred_tag = preferred_tag
        self.rerank_calls = 0
        self.rerank_batch_calls = 0

    def _order(self, candidates, top_k):
        ordered = sorted(
            candidates, key=lambda c: 0 if c["relevant_article_tag"] == self.preferred_tag else 1
        )
        return ordered[:top_k]

    def rerank(self, query, candidates, top_k=3):
        self.rerank_calls += 1
        return self._order(candidates, top_k)

    def rerank_batch(self, queries, candidate_lists, top_k=3):
        self.rerank_batch_calls += 1
        return [self._order(c, top_k) for c in candidate_lists]


def test_run_pipeline_prefers_rerank_batch_when_available(tmp_path):
    questions = [
        {"id": 1, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
        {"id": 2, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu?"},
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    reranker = _StubBatchReranker(preferred_tag="B|Doc B|Điều 2")
    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1,
        reranker=reranker,
    )

    assert reranker.rerank_batch_calls == 1
    assert reranker.rerank_calls == 0
    assert len(entries) == 2
    for entry in entries:
        assert validate_entry(entry) == []


def test_run_pipeline_reranker_takes_precedence_and_selects_preferred(tmp_path):
    questions = [{"id": 1, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    reranker = _StubReranker(preferred_tag="B|Doc B|Điều 2")
    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1,
        reranker=reranker,
    )

    assert entries[0]["relevant_articles"] == ["B|Doc B|Điều 2"]
    assert validate_entry(entries[0]) == []


def test_run_pipeline_reranker_pool_unions_dense_candidates(tmp_path):
    # BM25 alone surfaces Doc A; dense stub surfaces Doc B. The reranker should
    # receive BOTH as candidates (union), proving the wider recall pool.
    questions = [{"id": 1, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    dense_stub = _StubDenseRetriever([CORPUS[1]])  # votes Doc B
    reranker = _StubReranker(preferred_tag="B|Doc B|Điều 2")
    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=1, top_k_final=1,
        dense_retriever=dense_stub, reranker=reranker,
    )

    # reranker saw candidates from both BM25 (Doc A) and dense (Doc B)
    assert reranker.seen_candidate_counts[0] == 2
    assert entries[0]["relevant_articles"] == ["B|Doc B|Điều 2"]


def test_run_pipeline_without_dense_retriever_matches_phase1_bm25_only_behavior(tmp_path):
    questions = [{"id": 1, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1)

    assert entries[0]["relevant_articles"] == ["A|Doc A|Điều 1"]


def test_run_pipeline_respects_limit(tmp_path):
    questions = [
        {"id": 1, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"},
        {"id": 2, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), limit=1)

    assert len(entries) == 1
    assert entries[0]["id"] == 1


def test_run_pipeline_with_segmentation_produces_valid_entries(tmp_path):
    questions = [{"id": 1, "question": "Doanh nghiệp nhỏ và vừa được ưu đãi gì khi đấu thầu?"}]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1,
        use_segmentation=True,
    )

    assert len(entries) == 1
    assert validate_entry(entries[0]) == []
    assert entries[0]["relevant_articles"] == ["A|Doc A|Điều 1"]


def test_run_pipeline_with_abbreviation_expansion_matches_via_full_form(tmp_path):
    # Question uses "DNNVV"; corpus article uses the full form only. Without
    # expansion BM25 can't match; with expansion it should retrieve Doc A.
    questions = [{"id": 1, "question": "DNNVV được ưu đãi gì khi đấu thầu?"}]
    corpus = [
        {
            "relevant_doc_tag": "A|Doc A",
            "relevant_article_tag": "A|Doc A|Điều 1",
            "doc_name": "Doc A",
            "article_id": "Điều 1",
            "text": "Ưu đãi cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu.",
        },
        {
            "relevant_doc_tag": "B|Doc B",
            "relevant_article_tag": "B|Doc B|Điều 2",
            "doc_name": "Doc B",
            "article_id": "Điều 2",
            "text": "Quy định khác không liên quan.",
        },
    ]
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(
        str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1,
        expand_abbreviations=True,
    )

    assert entries[0]["relevant_articles"] == ["A|Doc A|Điều 1"]
    assert validate_entry(entries[0]) == []
