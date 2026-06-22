def searchable_text(record: dict) -> str:
    """Combined text used for indexing by both BM25 and dense retrieval.

    Including doc_name and article_id (not just body text) lets a question
    that names a law/article match even when the article body itself
    doesn't repeat those words.
    """
    return f"{record.get('doc_name', '')} {record.get('article_id', '')} {record.get('text', '')}"
