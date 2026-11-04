"""Compares retrieval setups on labelled queries over pinned CPython source.

    python -m evals.run --split dev               # for tuning
    python -m evals.run --split test --gate       # for reporting; exits 1 below the floors

Tune on dev only. Test is the number you report.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from codesearch.chunking import Chunk, symbol_ranges
from codesearch.embed import cache_dir, fetch
from codesearch.index import Index, pack

CORPUS_TAG = "v3.12.8"
CORPUS_URL = f"https://github.com/python/cpython/archive/refs/tags/{CORPUS_TAG}.tar.gz"
CORPUS_SHA = "b4387264881e09bbd3dbb703750adb472335c98c1ca1ffebfef34e3665b3df19"
QUERIES = Path(__file__).with_name("queries.yaml")
BUDGET = 2000
MODULES = """argparse ast asyncio base64 bisect calendar collections concurrent configparser contextlib copy csv dataclasses
difflib dis email enum fnmatch ftplib functools getopt gettext glob gzip hashlib heapq hmac html http imaplib inspect
ipaddress json locale logging mimetypes optparse pathlib pickle poplib pprint queue random secrets selectors shlex
shutil smtplib socket socketserver sqlite3 ssl statistics string subprocess tarfile tempfile textwrap threading timeit
tokenize traceback typing unittest urllib uuid xml zipfile _pydatetime""".split()

# Regression floors on the test split, set from the first full run. A drop below them fails CI.
FLOORS = {"hybrid": {"mrr": 0.42, "hit5": 0.58, "context": 0.65}, "bm25": {"mrr": 0.32, "hit5": 0.41, "context": 0.44}}

CONFIGS = [  # name, chunker, tokenizer, mode
    ("window + plain keywords", "window", "naive", "bm25"),
    ("file + code keywords", "file", "code", "bm25"),
    ("window + code keywords", "window", "code", "bm25"),
    ("ast + plain keywords", "ast", "naive", "bm25"),
    ("ast + code keywords (bm25)", "ast", "code", "bm25"),
    ("ast + embeddings (dense)", "ast", "code", "dense"),
    ("ast + hybrid", "ast", "code", "hybrid"),
]


@dataclass(frozen=True)
class Query:
    text: str
    targets: tuple[str, ...]
    kind: str

    @property
    def split(self) -> str:
        return "dev" if hashlib.sha256(self.text.encode()).digest()[0] % 2 == 0 else "test"


def load_queries() -> list[Query]:
    data = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    return [Query(str(row["q"]), tuple(row["a"]), kind) for kind, rows in data.items() for row in rows]


def corpus_sources() -> dict[str, str]:
    """Only the listed modules, without their test suites: tests quote the code and would answer for it."""
    archive = fetch(CORPUS_URL, cache_dir() / "cpython.tar.gz", CORPUS_SHA)
    prefix = f"cpython-{CORPUS_TAG[1:]}/Lib/"
    sources: dict[str, str] = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if not (member.isfile() and member.name.startswith(prefix) and member.name.endswith(".py")):
                continue
            rel = member.name[len(prefix):]
            parts = rel.split("/")
            if parts[0].removesuffix(".py") in MODULES and not {"test", "tests"} & set(parts):
                sources[rel] = tar.extractfile(member).read().decode("utf-8", errors="replace")
    return sources


def locate(sources: dict[str, str]) -> dict[str, tuple[str, int, int]]:
    where = {}
    for path, source in sources.items():
        try:
            ranges = symbol_ranges(path, source)
        except SyntaxError:
            continue
        where.update({symbol: (path, *span) for symbol, span in ranges.items()})
    return where


def is_relevant(chunk: Chunk, target: tuple[str, int, int]) -> bool:
    """Shared by every chunker: the overlap must cover half of the target or half of the chunk, whichever is smaller."""
    path, start, end = target
    if chunk.path != path:
        return False
    overlap = min(chunk.end, end) - max(chunk.start, start) + 1
    return overlap > 0 and overlap * 2 >= min(end - start + 1, chunk.end - chunk.start + 1)


def first_hit(chunks: list[Chunk], targets: list[tuple[str, int, int]]) -> int | None:
    for rank, chunk in enumerate(chunks, 1):
        if any(is_relevant(chunk, t) for t in targets):
            return rank
    return None


def bootstrap_diff(a: list[float], b: list[float], resamples: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of b - a with a 95% interval, resampling queries in pairs."""
    diff = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(index: Index, mode: str, queries: list[Query], where: dict) -> dict:
    reciprocal, hit1, hit5, hit10, fits, latencies = [], [], [], [], [], []
    for q in queries:
        targets = [where[s] for s in q.targets]
        start = time.perf_counter()
        ranked = index.search(q.text, mode, 50)
        latencies.append(time.perf_counter() - start)
        rank = first_hit(ranked[:10], targets)
        reciprocal.append(1 / rank if rank else 0.0)
        hit1.append(float(rank == 1))
        hit5.append(float(bool(rank and rank <= 5)))
        hit10.append(float(rank is not None))
        fits.append(float(first_hit(pack(ranked, BUDGET), targets) is not None))
    return {"mrr": statistics.mean(reciprocal), "hit1": statistics.mean(hit1), "hit5": statistics.mean(hit5),
            "hit10": statistics.mean(hit10), "context": statistics.mean(fits),
            "p95_ms": float(np.percentile(latencies, 95) * 1000), "per_query": reciprocal}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)

    from codesearch.embed import Embedder

    sources = corpus_sources()
    where = locate(sources)
    everything = load_queries()
    unknown = sorted({s for q in everything for s in q.targets if s not in where})
    if unknown:
        print("Labels that don't exist in the corpus:", *unknown, sep="\n  ")
        return 2
    queries = [q for q in everything if q.split == args.split]

    embed = Embedder()
    indexes: dict[tuple[str, str], Index] = {}
    start = time.perf_counter()
    for _, chunker, tokenizer, mode in CONFIGS:
        needs_vectors = mode != "bm25"
        key = (chunker, tokenizer)
        if key not in indexes or (needs_vectors and indexes[key].vectors is None):
            indexes[key] = Index.build(sources, chunker, tokenizer, embed if needs_vectors else None)
    built = time.perf_counter() - start
    ast_index = indexes[("ast", "code")]

    lines = [f"Corpus: {len(sources)} files from CPython {CORPUS_TAG}, {len(ast_index.chunks)} AST chunks. Index build {built:.0f} s on CPU.",
             f"Split: {args.split}, {len(queries)} queries ({sum(q.kind == 'paraphrase' for q in queries)} paraphrase, "
             f"{sum(q.kind == 'identifier' for q in queries)} identifier). Context = target inside a {BUDGET}-token pack.", "",
             "| Setup | MRR@10 | Hit@1 | Hit@5 | Hit@10 | In context | p95 ms |", "|---|---|---|---|---|---|---|"]
    results = {}
    for name, chunker, tokenizer, mode in CONFIGS:
        r = evaluate(indexes[(chunker, tokenizer)], mode, queries, where)
        results[name] = r
        lines.append(f"| {name} | {r['mrr']:.3f} | {r['hit1']:.2f} | {r['hit5']:.2f} | {r['hit10']:.2f} | {r['context']:.2f} | {r['p95_ms']:.1f} |")

    lines += ["", "| By query type | BM25 MRR | Dense MRR | Hybrid MRR |", "|---|---|---|---|"]
    for kind in ("paraphrase", "identifier"):
        subset = [q for q in queries if q.kind == kind]
        cells = [evaluate(ast_index, m, subset, where)["mrr"] for m in ("bm25", "dense", "hybrid")]
        lines.append(f"| {kind} ({len(subset)}) | {cells[0]:.3f} | {cells[1]:.3f} | {cells[2]:.3f} |")

    mean, low, high = bootstrap_diff(results["ast + code keywords (bm25)"]["per_query"], results["ast + hybrid"]["per_query"])
    lines += ["", f"Hybrid minus BM25, MRR: {mean:+.3f} (95% CI {low:+.3f} to {high:+.3f}, paired bootstrap over queries)."]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")

    if args.gate:
        names = {"hybrid": "ast + hybrid", "bm25": "ast + code keywords (bm25)"}
        failed = [f"{m} {metric} {results[names[m]][metric]:.3f} < {floor}" for m, floors in FLOORS.items()
                  for metric, floor in floors.items() if results[names[m]][metric] < floor]
        print("Gate: FAIL\n  " + "\n  ".join(failed) if failed else "Gate: PASS")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
