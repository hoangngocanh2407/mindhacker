def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60, top_k: int = 15
) -> list[dict]:
    """Merge ranked result lists (e.g. BM25 + dense) by Reciprocal Rank Fusion.

    Each record's fused score is the sum, across lists, of 1 / (k + rank)
    where rank is 1-based position in that list. Records are deduped by
    `relevant_article_tag` — if the same article appears in multiple lists,
    its scores are summed (rewarding agreement between retrievers) and the
    first-seen copy of the record is kept.
    """
    scores: dict[str, float] = {}
    records: dict[str, dict] = {}

    for results in result_lists:
        for rank, record in enumerate(results, start=1):
            tag = record["relevant_article_tag"]
            scores[tag] = scores.get(tag, 0.0) + 1.0 / (k + rank)
            if tag not in records:
                records[tag] = record

    ranked_tags = sorted(scores, key=lambda tag: scores[tag], reverse=True)

    fused = []
    for tag in ranked_tags[:top_k]:
        record = dict(records[tag])
        record["fused_score"] = scores[tag]
        fused.append(record)
    return fused
