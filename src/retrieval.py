import re
from typing import Callable

from rank_bm25 import BM25Okapi

from src.corpus_text import searchable_text


def tokenize(text: str) -> list[str]:
    # Baseline tokenizer: lowercase + \w+ split. No Vietnamese word
    # segmentation — "doanh nghiệp nhỏ và vừa" becomes 5 separate tokens.
    return re.findall(r"\w+", text.lower())


def segment_tokenize(text: str) -> list[str]:
    """Vietnamese word-segmenting tokenizer (pyvi). Joins compound words with
    underscores ("doanh nghiệp" -> "doanh_nghiệp") so multi-word legal terms
    become single BM25 tokens, sharpening matching vs the plain regex tokenizer.

    `re.findall(r"\\w+", ...)` keeps the underscore (it's a word char) so the
    compound survives as one token and punctuation is dropped. pyvi is imported
    lazily so the dependency is only needed when segmentation is actually used.
    """
    from pyvi import ViTokenizer

    return re.findall(r"\w+", ViTokenizer.tokenize(text).lower())


class BM25Retriever:
    def __init__(self, corpus: list[dict], tokenizer: Callable[[str], list[str]] = tokenize):
        self.corpus = corpus
        self._tokenizer = tokenizer
        self._tokenized = [tokenizer(searchable_text(record)) for record in corpus]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        scores = self._bm25.get_scores(self._tokenizer(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results
