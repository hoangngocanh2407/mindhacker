def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 15,
    weights: list[float] | None = None,
) -> list[dict]:
    """Merge ranked result lists (e.g. BM25 + dense) by Reciprocal Rank Fusion.

    Each record's fused score is the sum, across lists, of
    weight * 1 / (k + rank), where rank is 1-based position in that list and
    weight is the per-list weight (default 1.0 for every list, i.e. classic
    equal-weight RRF). Records are deduped by `relevant_article_tag` — if the
    same article appears in multiple lists, its weighted scores are summed
    (rewarding agreement) and the first-seen copy of the record is kept.

    Lowering a list's weight reduces how much that retriever can pull an
    article up the ranking — used to down-weight dense, which is noisy at the
    article level on this corpus (see docs/reports/phase2_report.md).
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match result_lists length "
            f"({len(result_lists)})"
        )

    scores: dict[str, float] = {}
    records: dict[str, dict] = {}

    for results, weight in zip(result_lists, weights):
        for rank, record in enumerate(results, start=1):
            tag = record["relevant_article_tag"]
            scores[tag] = scores.get(tag, 0.0) + weight * (1.0 / (k + rank))
            if tag not in records:
                records[tag] = record

    ranked_tags = sorted(scores, key=lambda tag: scores[tag], reverse=True)

    fused = []
    for tag in ranked_tags[:top_k]:
        record = dict(records[tag])
        record["fused_score"] = scores[tag]
        fused.append(record)
    return fused
