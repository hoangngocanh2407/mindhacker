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
