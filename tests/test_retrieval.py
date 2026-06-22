from src.retrieval import BM25Retriever, tokenize

TEXT_MATCH_CORPUS = [
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
    {
        "relevant_doc_tag": "C|Doc C",
        "relevant_article_tag": "C|Doc C|Điều 3",
        "doc_name": "Doc C",
        "article_id": "Điều 3",
        "text": "Thủ tục chuyển đổi hộ kinh doanh thành doanh nghiệp.",
    },
]

# Body text deliberately does NOT contain "Luật Doanh nghiệp 59/2020/QH14" —
# only doc_name does. Proves the combined index actually uses doc_name.
DOC_NAME_MATCH_CORPUS = [
    {
        "relevant_doc_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14",
        "relevant_article_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 7",
        "doc_name": "Luật Doanh nghiệp 59/2020/QH14",
        "article_id": "Điều 7",
        "text": "Cổ đông có quyền tham dự và phát biểu trong các cuộc họp.",
    },
    {
        "relevant_doc_tag": "X|Luật Đấu thầu",
        "relevant_article_tag": "X|Luật Đấu thầu|Điều 9",
        "doc_name": "Luật Đấu thầu",
        "article_id": "Điều 9",
        "text": "Quy định về hồ sơ dự thầu và thời hạn nộp hồ sơ.",
    },
]

# Body text deliberately does NOT contain "Điều 4" — only article_id does.
ARTICLE_ID_MATCH_CORPUS = [
    {
        "relevant_doc_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14",
        "relevant_article_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 4",
        "doc_name": "Luật Doanh nghiệp 59/2020/QH14",
        "article_id": "Điều 4",
        "text": "Doanh nghiệp nhỏ và vừa được xác định theo quy mô lao động và doanh thu.",
    },
    {
        "relevant_doc_tag": "X|Luật Đấu thầu",
        "relevant_article_tag": "X|Luật Đấu thầu|Điều 9",
        "doc_name": "Luật Đấu thầu",
        "article_id": "Điều 9",
        "text": "Quy định về hồ sơ dự thầu và thời hạn nộp hồ sơ.",
    },
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    tokens = tokenize("Ưu đãi, thuế và đất đai?")
    assert tokens == ["ưu", "đãi", "thuế", "và", "đất", "đai"]


def test_search_returns_most_relevant_article_first():
    retriever = BM25Retriever(TEXT_MATCH_CORPUS)
    results = retriever.search("ưu đãi thuế khi tham gia đấu thầu cho doanh nghiệp", top_k=2)
    assert results[0]["relevant_article_tag"] == "A|Doc A|Điều 1"
    assert len(results) <= 2


def test_search_result_includes_score_key():
    retriever = BM25Retriever(TEXT_MATCH_CORPUS)
    results = retriever.search("hợp đồng lao động vi phạm kỷ luật", top_k=1)
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_search_matches_via_doc_name_when_body_text_does_not_contain_query():
    retriever = BM25Retriever(DOC_NAME_MATCH_CORPUS)
    results = retriever.search("Luật Doanh nghiệp 59/2020/QH14", top_k=1)
    assert results[0]["relevant_article_tag"] == "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 7"


def test_search_matches_via_article_id_when_body_text_does_not_contain_it():
    retriever = BM25Retriever(ARTICLE_ID_MATCH_CORPUS)
    results = retriever.search("Điều 4", top_k=1)
    assert results[0]["relevant_article_tag"] == "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 4"
