"""Build, save and search an index, and pack results into a fixed context budget."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .bm25 import BM25
from .chunking import CHUNKERS, Chunk
from .tokens import code_tokens, naive_tokens

TOKENIZERS = {"code": code_tokens, "naive": naive_tokens}
RRF_K = 60
CANDIDATES = 100
Embed = Callable[[list[str]], np.ndarray]


def _key(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.search_text.encode()).hexdigest()


def read_sources(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(root.rglob("*.py")) if p.is_file()}


def fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. It uses ranks only, so BM25 and cosine scores never need a common scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Index:
    def __init__(self, chunks: list[Chunk], tokenizer: str = "code", vectors: np.ndarray | None = None,
                 embed: Embed | None = None, embedded: int = 0) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.vectors = vectors
        self.embed = embed
        self.embedded = embedded  # how many chunks this build actually had to embed
        self.bm25 = BM25([c.keyword_text for c in chunks], TOKENIZERS[tokenizer])

    @classmethod
    def build(cls, sources: dict[str, str], chunker: str = "ast", tokenizer: str = "code", embed: Embed | None = None,
              previous: "Index | None" = None) -> "Index":
        """Only chunks whose text changed are embedded again; the rest reuse `previous`."""
        chunks = [c for path, source in sources.items() for c in CHUNKERS[chunker](path, source)]
        if embed is None:
            return cls(chunks, tokenizer)
        known: dict[str, np.ndarray] = {}
        if previous is not None and previous.vectors is not None:
            known = {_key(c): v for c, v in zip(previous.chunks, previous.vectors)}
        keys = [_key(c) for c in chunks]
        missing = sorted({k for k in keys if k not in known})
        if missing:
            texts = {_key(c): c.search_text for c in chunks}
            known.update(zip(missing, embed([texts[k] for k in missing])))
        vectors = np.stack([known[k] for k in keys]) if keys else np.zeros((0, 384), dtype=np.float32)
        return cls(chunks, tokenizer, vectors, embed, len(missing))

    def rank(self, query: str, mode: str = "hybrid", k: int = 10) -> list[int]:
        keyword = lambda: [doc for doc, _ in self.bm25.search(query, CANDIDATES)]
        if mode == "bm25":
            return keyword()[:k]
        if self.vectors is None or self.embed is None:
            raise ValueError(f"mode {mode!r} needs an index built with embeddings")
        similarity = self.vectors @ self.embed([query])[0]
        dense = np.argsort(-similarity, kind="stable")[:CANDIDATES].tolist()
        if mode == "dense":
            return dense[:k]
        if mode == "hybrid":
            return [doc for doc, _ in fuse([keyword(), dense])][:k]
        raise ValueError(f"unknown mode {mode!r}")

    def search(self, query: str, mode: str = "hybrid", k: int = 10) -> list[Chunk]:
        return [self.chunks[i] for i in self.rank(query, mode, k)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(json.dumps(
            {"tokenizer": self.tokenizer, "chunks": [asdict(c) for c in self.chunks]}), encoding="utf-8")
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory: Path, embed: Embed | None = None) -> "Index":
        data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        path = directory / "vectors.npy"
        vectors = np.load(path) if path.exists() else None
        return cls([Chunk(**c) for c in data["chunks"]], data["tokenizer"], vectors, embed)


def estimate_tokens(text: str) -> int:
    """About 4 characters a token: an estimate, close enough to budget with, not a billing figure."""
    return math.ceil(len(text) / 4)


def _overlaps(a: Chunk, b: Chunk) -> bool:
    return a.path == b.path and a.start <= b.end and b.start <= a.end


def pack(ranked: list[Chunk], budget: int) -> list[Chunk]:
    """Best first, skipping repeats of lines already included and anything that no longer fits."""
    chosen: list[Chunk] = []
    left = budget
    for chunk in ranked:
        cost = estimate_tokens(chunk.search_text)
        if cost <= left and not any(_overlaps(chunk, c) for c in chosen):
            chosen.append(chunk)
            left -= cost
    return chosen


def render(chunks: list[Chunk]) -> str:
    """Grouped by file and in line order, so a model reads code the way it is laid out."""
    parts = []
    for c in sorted(chunks, key=lambda c: (c.path, c.start)):
        parts.append(f"### {c.path}:{c.start}-{c.end} ({c.symbol})\n{c.text}")
    return "\n\n".join(parts)
