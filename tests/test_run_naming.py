from src.run_naming import run_slug


def test_slug_bm25_default():
    assert run_slug("bm25", 5) == "bm25_kf5"


def test_slug_bm25_with_expand():
    assert run_slug("bm25", 10, expand_query=True) == "bm25_kf10_expand"


def test_slug_hybrid_includes_dense_weight():
    assert run_slug("hybrid", 5, dense_weight=0.3) == "hybrid_kf5_dw0.3"


def test_slug_hybrid_with_all_options():
    assert run_slug("hybrid", 15, dense_weight=1.0, expand_query=True) == "hybrid_kf15_dw1.0_expand"


def test_slug_bm25_omits_dense_weight():
    # dense_weight is irrelevant for bm25 mode and must not appear in the slug.
    assert "dw" not in run_slug("bm25", 5, dense_weight=0.3)


def test_distinct_configs_produce_distinct_slugs():
    slugs = {
        run_slug("bm25", 5),
        run_slug("bm25", 10),
        run_slug("bm25", 5, expand_query=True),
        run_slug("hybrid", 5, dense_weight=0.3),
        run_slug("hybrid", 5, dense_weight=1.0),
    }
    assert len(slugs) == 5
