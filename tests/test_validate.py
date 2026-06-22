import json

from src.export import build_result_entry, write_results
from src.validate import validate_entry, validate_results

ARTICLE = {
    "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "article_id": "Điều 1",
    "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật.",
    "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
}


def test_build_result_entry_has_required_fields_and_valid_tags():
    entry = build_result_entry(1, [ARTICLE], top_k_final=5)
    assert entry["id"] == 1
    assert "Điều 1" in entry["answer"]
    assert entry["relevant_docs"] == [ARTICLE["relevant_doc_tag"]]
    assert entry["relevant_articles"] == [ARTICLE["relevant_article_tag"]]
    assert validate_entry(entry) == []


def test_build_result_entry_dedupes_relevant_docs_and_relevant_articles():
    # Same article passed in twice must collapse to a single entry in BOTH
    # output lists, preserving first-seen order.
    entry = build_result_entry(2, [ARTICLE, ARTICLE], top_k_final=5)
    assert entry["relevant_docs"] == [ARTICLE["relevant_doc_tag"]]
    assert entry["relevant_articles"] == [ARTICLE["relevant_article_tag"]]
    assert len(entry["relevant_articles"]) == 1


def test_validate_entry_flags_missing_field():
    errors = validate_entry({"id": 1, "answer": "x", "relevant_docs": ["a|b"]})
    assert any("relevant_articles" in e for e in errors)


def test_validate_entry_flags_malformed_article_tag():
    entry = {
        "id": 2,
        "answer": "...",
        "relevant_docs": ["X|Y"],
        "relevant_articles": ["chỉ có hai phần|không đủ"],
    }
    errors = validate_entry(entry)
    assert any("malformed" in e for e in errors)


def test_validate_entry_flags_empty_answer():
    entry = {"id": 3, "answer": "  ", "relevant_docs": ["a|b"], "relevant_articles": ["a|b|Điều 1"]}
    errors = validate_entry(entry)
    assert any("empty answer" in e for e in errors)


def test_validate_entry_flags_answer_that_does_not_cite_any_relevant_article():
    entry = {
        "id": 4,
        "answer": "Không rõ.",
        "relevant_docs": ["A|Doc A"],
        "relevant_articles": ["A|Doc A|Điều 1"],
    }
    errors = validate_entry(entry)
    assert "id=4: answer does not cite any relevant article" in errors


def test_validate_entry_passes_when_answer_cites_relevant_article():
    entry = {
        "id": 5,
        "answer": "Theo Điều 1 quy định như sau...",
        "relevant_docs": ["A|Doc A"],
        "relevant_articles": ["A|Doc A|Điều 1"],
    }
    assert validate_entry(entry) == []


def test_validate_results_detects_missing_ids(tmp_path):
    entries = [build_result_entry(1, [ARTICLE])]
    path = tmp_path / "results.json"
    write_results(entries, str(path))

    errors = validate_results(str(path), expected_count=2)

    assert any("expected 2 entries" in e for e in errors)
    assert any("missing ids" in e for e in errors)


def test_validate_results_passes_complete_valid_set(tmp_path):
    entries = [build_result_entry(1, [ARTICLE]), build_result_entry(2, [ARTICLE])]
    path = tmp_path / "results.json"
    write_results(entries, str(path))

    errors = validate_results(str(path), expected_count=2)

    assert errors == []
