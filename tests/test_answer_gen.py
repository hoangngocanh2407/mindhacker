from src.answer_gen import generate_answer

ARTICLE = {
    "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "article_id": "Điều 1",
    "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật.",
    "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
}


def test_generate_answer_includes_article_citation():
    answer = generate_answer([ARTICLE])
    assert "Điều 1" in answer
    assert "Nghị định 80/2021/NĐ-CP" in answer


def test_generate_answer_handles_empty_list():
    answer = generate_answer([])
    assert "không tìm thấy điều luật" in answer.lower()


def test_generate_answer_truncates_long_text():
    long_article = dict(ARTICLE)
    long_article["text"] = "a" * 1000
    answer = generate_answer([long_article])
    assert "..." in answer
    assert len(answer) < 1000


def test_generate_answer_respects_max_articles():
    articles = [ARTICLE, ARTICLE, ARTICLE]
    answer = generate_answer(articles, max_articles=1)
    assert answer.count("Điều 1") == 1
