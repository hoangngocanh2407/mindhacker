import json

from src.preprocess import dedupe_articles, drop_unknown, load_corpus, preprocess_corpus

SAMPLE_RECORDS = [
    {
        "doc_id": "80/2021/NĐ-CP",
        "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "article_id": "Điều 1",
        "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều.",
        "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
    },
    {
        "doc_id": "80/2021/NĐ-CP",
        "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "article_id": "Điều 1",
        "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều.",
        "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
    },
    {
        "doc_id": "UNKNOWN",
        "doc_name": "Nghị định 18-CP về Luật Đầu tư nước ngoài",
        "article_id": "Điều 5",
        "text": "Quy định về đầu tư nước ngoài tại Việt Nam.",
        "relevant_doc_tag": "UNKNOWN|Nghị định 18-CP về Luật Đầu tư nước ngoài",
        "relevant_article_tag": "UNKNOWN|Nghị định 18-CP về Luật Đầu tư nước ngoài|Điều 5",
    },
    {
        "doc_id": "59/2020/QH14",
        "doc_name": "Luật Doanh nghiệp 59/2020/QH14",
        "article_id": "Điều 4",
        "text": "Doanh nghiệp nhỏ và vừa được hiểu là doanh nghiệp có quy mô nhỏ.",
        "relevant_doc_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14",
        "relevant_article_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 4",
    },
]


def test_dedupe_articles_removes_exact_duplicates():
    deduped = dedupe_articles(SAMPLE_RECORDS)
    tags = [r["relevant_article_tag"] for r in deduped]
    assert len(deduped) == 3
    assert tags.count(
        "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1"
    ) == 1


def test_drop_unknown_removes_unknown_doc_id():
    clean, dropped = drop_unknown(SAMPLE_RECORDS)
    assert all(r["doc_id"] != "UNKNOWN" for r in clean)
    assert len(dropped) == 1
    assert dropped[0]["doc_id"] == "UNKNOWN"


def test_preprocess_corpus_writes_clean_file_and_returns_summary(tmp_path):
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "clean.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")

    summary = preprocess_corpus(str(raw_path), str(out_path))

    assert summary == {"total": 4, "after_dedupe": 3, "dropped_unknown": 1, "final": 2}
    clean = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(clean) == 2
    assert all(r["doc_id"] != "UNKNOWN" for r in clean)


def test_preprocess_corpus_creates_missing_nested_output_directory(tmp_path):
    raw_path = tmp_path / "raw.json"
    nested_out_path = tmp_path / "nested" / "processed" / "corpus_clean.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")

    assert not nested_out_path.parent.exists()

    preprocess_corpus(str(raw_path), str(nested_out_path))

    assert nested_out_path.exists()
    assert (nested_out_path.parent / "corpus_clean.json.dropped.json").exists()


def test_load_corpus_returns_list_of_dicts(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")
    records = load_corpus(str(raw_path))
    assert len(records) == 4
    assert records[0]["doc_id"] == "80/2021/NĐ-CP"
