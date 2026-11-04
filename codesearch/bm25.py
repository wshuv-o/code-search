"""BM25 over an inverted index, with no search engine to run."""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable

K1 = 1.2
B = 0.75


class BM25:
    def __init__(self, docs: list[str], tokenize: Callable[[str], list[str]]) -> None:
        self.tokenize = tokenize
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        for i, doc in enumerate(docs):
            tokens = tokenize(doc)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.avg_length = sum(self.lengths) / len(docs) if docs else 0.0

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        n = len(self.lengths)
        scores: dict[int, float] = {}
        for term in set(self.tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc, tf in postings:
                norm = 1 - B + B * self.lengths[doc] / self.avg_length
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (K1 + 1) / (tf + K1 * norm)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
