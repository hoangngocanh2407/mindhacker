"""Query-side preprocessing for retrieval (Phase 4).

Only abbreviation expansion is implemented: legal abbreviations that questions
sometimes use (TNHH, GTGT, BHXH...) are expanded to the full form the corpus
almost always uses, so BM25 can match the key terms. The full form is
APPENDED (not substituted), keeping the original abbreviation too — lowest
risk, no signal lost. Synonym expansion was deliberately left out (it tends
to dilute precision, which is already the weak point).

Dictionary is curated from abbreviations actually seen in the question set
whose full form is well-represented in the corpus (verified on real data).
"""
import re

LEGAL_ABBREVIATIONS: dict[str, str] = {
    "TNHH": "trách nhiệm hữu hạn",
    "TNDN": "thu nhập doanh nghiệp",
    "TNCN": "thu nhập cá nhân",
    "GTGT": "giá trị gia tăng",
    "BHXH": "bảo hiểm xã hội",
    "BHYT": "bảo hiểm y tế",
    "BHTN": "bảo hiểm thất nghiệp",
    "DNNVV": "doanh nghiệp nhỏ và vừa",
    "SHTT": "sở hữu trí tuệ",
    "UBND": "ủy ban nhân dân",
}


def expand_query(query: str, abbreviations: dict[str, str] = LEGAL_ABBREVIATIONS) -> str:
    """Append the full form for each known legal abbreviation found in `query`.

    Matches are whole-word and case-sensitive (the uppercase abbreviation),
    so substrings like "TNHHX" and lowercase "tnhh" are not expanded. Each
    full form is appended at most once even if its abbreviation repeats.
    A query with no known abbreviation is returned unchanged.
    """
    additions = []
    for abbr, full in abbreviations.items():
        if re.search(rf"\b{re.escape(abbr)}\b", query):
            additions.append(full)
    if not additions:
        return query
    return query + " " + " ".join(additions)
