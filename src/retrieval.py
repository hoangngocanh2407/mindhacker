import re

from rank_bm25 import BM25Okapi

from src.corpus_text import searchable_text


def tokenize(text: str) -> list[str]:
    # Baseline tokenizer for Vietnamese, Phase 1 only: lowercase + \w+ split.
    # No proper word segmentation, stemming, or stopword removal yet.
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokenized = [tokenize(searchable_text(record)) for record in corpus]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results
