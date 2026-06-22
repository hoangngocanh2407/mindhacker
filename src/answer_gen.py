def generate_answer(top_articles: list[dict], max_articles: int = 5) -> str:
    if not top_articles:
        return "Không tìm thấy điều luật liên quan trong cơ sở dữ liệu để trả lời câu hỏi này."

    parts = []
    for article in top_articles[:max_articles]:
        snippet = article["text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "..."
        parts.append(f"Theo {article['article_id']} {article['doc_name']}: {snippet}")
    return " ".join(parts)
