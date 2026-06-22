from src.query_processing import LEGAL_ABBREVIATIONS, expand_query


def test_expand_query_appends_full_form_for_known_abbreviation():
    result = expand_query("Công ty TNHH cần làm gì?")
    assert "TNHH" in result  # original kept
    assert "trách nhiệm hữu hạn" in result  # full form appended


def test_expand_query_leaves_query_without_abbreviation_unchanged():
    q = "Doanh nghiệp nhỏ và vừa được hỗ trợ gì?"
    assert expand_query(q) == q


def test_expand_query_does_not_match_substring_inside_longer_token():
    # "TNHHX" is not the whole-word abbreviation "TNHH".
    q = "Mã TNHHX là gì?"
    assert "trách nhiệm hữu hạn" not in expand_query(q)


def test_expand_query_does_not_match_lowercase():
    # Abbreviations only expand when written uppercase as in the dict.
    q = "tnhh viết thường không phải viết tắt"
    assert expand_query(q) == q


def test_expand_query_handles_multiple_abbreviations():
    result = expand_query("Thuế GTGT và BHXH áp dụng thế nào?")
    assert "giá trị gia tăng" in result
    assert "bảo hiểm xã hội" in result


def test_expand_query_appends_full_form_only_once_for_repeated_abbreviation():
    result = expand_query("BHXH và lại BHXH nữa")
    assert result.count("bảo hiểm xã hội") == 1


def test_legal_abbreviations_dict_is_uppercase_keys():
    for abbr in LEGAL_ABBREVIATIONS:
        assert abbr == abbr.upper()
