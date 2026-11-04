"""Tokenizers for keyword search. Code names hide words, so `parseHTTPHeader` must also match "header"."""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_]+")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def naive_tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def code_tokens(text: str) -> list[str]:
    """Each identifier whole, plus its snake_case and camelCase parts."""
    out: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower()
        out.append(whole)
        parts = [p.lower() for p in _PART.findall(word)]
        if len(parts) > 1:
            out.extend(parts)
    return out
