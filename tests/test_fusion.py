import pytest

from src.fusion import reciprocal_rank_fusion

ARTICLE_A = {"relevant_article_tag": "A|Doc A|Điều 1", "doc_name": "Doc A"}
ARTICLE_B = {"relevant_article_tag": "B|Doc B|Điều 2", "doc_name": "Doc B"}
ARTICLE_C = {"relevant_article_tag": "C|Doc C|Điều 3", "doc_name": "Doc C"}


def test_fusion_ranks_item_found_by_both_lists_above_single_list_top_hit():
    # B is rank 1 in list 1 only. A is rank 2 in list 1 and rank 1 in list 2 —
    # agreement between both retrievers should outrank a single top-1 hit.
    bm25_results = [ARTICLE_B, ARTICLE_A]
    dense_results = [ARTICLE_A, ARTICLE_C]

    fused = reciprocal_rank_fusion([bm25_results, dense_results], top_k=3)

    tags = [r["relevant_article_tag"] for r in fused]
    assert tags[0] == "A|Doc A|Điều 1"


def test_fusion_dedupes_by_relevant_article_tag():
    fused = reciprocal_rank_fusion([[ARTICLE_A], [ARTICLE_A]], top_k=5)
    tags = [r["relevant_article_tag"] for r in fused]
    assert tags.count("A|Doc A|Điều 1") == 1


def test_fusion_respects_top_k():
    fused = reciprocal_rank_fusion([[ARTICLE_A, ARTICLE_B, ARTICLE_C]], top_k=2)
    assert len(fused) == 2


def test_fusion_result_includes_fused_score():
    fused = reciprocal_rank_fusion([[ARTICLE_A]], top_k=1)
    assert "fused_score" in fused[0]
    assert isinstance(fused[0]["fused_score"], float)


def test_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_fusion_weight_zero_for_dense_excludes_dense_only_article():
    # weights=[1.0, 0.0]: list 2 (dense) contributes nothing, so a dense-only
    # article scores 0 and a bm25-only article ranks above it.
    bm25_results = [ARTICLE_B]   # bm25-only
    dense_results = [ARTICLE_A]  # dense-only
    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results], top_k=5, weights=[1.0, 0.0]
    )
    assert fused[0]["relevant_article_tag"] == "B|Doc B|Điều 2"
    assert fused[-1]["relevant_article_tag"] == "A|Doc A|Điều 1"
    assert fused[-1]["fused_score"] == 0.0


def test_fusion_weight_flips_winner_between_sources():
    bm25_results = [ARTICLE_B]   # bm25-only
    dense_results = [ARTICLE_A]  # dense-only
    # Favor dense → dense-only article wins.
    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results], top_k=5, weights=[0.0, 1.0]
    )
    assert fused[0]["relevant_article_tag"] == "A|Doc A|Điều 1"


def test_fusion_weights_none_matches_explicit_equal_weights():
    bm25_results = [ARTICLE_B, ARTICLE_A]
    dense_results = [ARTICLE_A, ARTICLE_C]
    default = reciprocal_rank_fusion([bm25_results, dense_results], top_k=3)
    explicit = reciprocal_rank_fusion(
        [bm25_results, dense_results], top_k=3, weights=[1.0, 1.0]
    )
    assert [r["relevant_article_tag"] for r in default] == [
        r["relevant_article_tag"] for r in explicit
    ]
    for d, e in zip(default, explicit):
        assert d["fused_score"] == e["fused_score"]


def test_fusion_raises_when_weights_length_mismatches():
    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion([[ARTICLE_A], [ARTICLE_B]], weights=[1.0])
