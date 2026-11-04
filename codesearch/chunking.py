"""Split source into retrievable chunks. The strategy is an eval variable, so all three live here."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_LINES = 120
WINDOW = 40
OVERLAP = 10
SYMBOL_BOOST = 5
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Chunk:
    path: str
    symbol: str
    start: int  # 1-indexed, inclusive
    end: int
    text: str

    @property
    def search_text(self) -> str:
        """The location is searchable too: a query for "cookies" should find http/cookies.py."""
        return f"{self.path} {self.symbol}\n{self.text}"

    @property
    def keyword_text(self) -> str:
        """The name counts more than the body: on the dev split this took identifier queries from 0.72 to 0.96 MRR."""
        return f"{self.path} {(self.symbol + ' ') * SYMBOL_BOOST}\n{self.text}"


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _first_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decorators] + [node.lineno])


def _emit(chunks: list[Chunk], path: str, symbol: str, lines: list[str], start: int, end: int) -> None:
    """Oversized regions become overlapping pieces that keep the same symbol."""
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    if start > end:
        return
    if end - start + 1 <= MAX_LINES:
        chunks.append(Chunk(path, symbol, start, end, "\n".join(lines[start - 1:end])))
        return
    at = start
    while at <= end:
        stop = min(at + MAX_LINES - 1, end)
        chunks.append(Chunk(path, symbol, at, stop, "\n".join(lines[at - 1:stop])))
        if stop == end:
            break
        at = stop - OVERLAP + 1


def _walk(chunks: list[Chunk], path: str, lines: list[str], body: list[ast.stmt], symbol: str, start: int, end: int) -> None:
    """Definitions get their own chunks; the code between them belongs to the enclosing symbol."""
    cursor = start
    for node in body:
        if not isinstance(node, _DEFS):
            continue
        first = _first_line(node)
        _emit(chunks, path, symbol, lines, cursor, first - 1)
        name = f"{symbol}.{node.name}" if symbol else node.name
        if isinstance(node, ast.ClassDef):
            _walk(chunks, path, lines, node.body, name, first, node.end_lineno)
        else:
            _emit(chunks, path, name, lines, first, node.end_lineno)
        cursor = node.end_lineno + 1
    _emit(chunks, path, symbol, lines, cursor, end)


def ast_chunks(path: str, source: str) -> list[Chunk]:
    """One chunk per function or method. Anything that doesn't parse as Python falls back to windows."""
    if not path.endswith(".py"):
        return window_chunks(path, source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return window_chunks(path, source)
    lines = source.splitlines()
    chunks: list[Chunk] = []
    _walk(chunks, path, lines, tree.body, module_name(path), 1, len(lines))
    return chunks


def window_chunks(path: str, source: str) -> list[Chunk]:
    """The usual baseline: fixed line windows that ignore the code's structure."""
    lines = source.splitlines()
    chunks: list[Chunk] = []
    at = 1
    while at <= len(lines):
        stop = min(at + WINDOW - 1, len(lines))
        text = "\n".join(lines[at - 1:stop])
        if text.strip():
            chunks.append(Chunk(path, f"{module_name(path)}:{at}", at, stop, text))
        if stop == len(lines):
            break
        at = stop - OVERLAP + 1
    return chunks


def file_chunks(path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    return [Chunk(path, module_name(path), 1, len(lines), source)] if source.strip() else []


CHUNKERS = {"ast": ast_chunks, "window": window_chunks, "file": file_chunks}


def symbol_ranges(path: str, source: str) -> dict[str, tuple[int, int]]:
    """Where each definition lives, independent of how any chunker cut the file. Evals label against this."""
    ranges: dict[str, tuple[int, int]] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, _DEFS):
                name = f"{prefix}.{node.name}"
                ranges[name] = (_first_line(node), node.end_lineno)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name)

    visit(ast.parse(source).body, module_name(path))
    return ranges
