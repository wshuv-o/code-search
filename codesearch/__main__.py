"""
    python -m codesearch index SOURCE_DIR
    python -m codesearch search "where do we retry failed uploads"
    python -m codesearch context "where do we retry failed uploads" --budget 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .index import Index, estimate_tokens, pack, read_sources, render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codesearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=Path(".codesearch"))
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("index")
    build.add_argument("source", type=Path)
    build.add_argument("--keyword-only", action="store_true", help="skip embeddings; needs no model download")
    for name in ("search", "context"):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default=None)
        if name == "search":
            p.add_argument("-k", type=int, default=10)
        else:
            p.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.command == "index":
        embed = None
        previous = None
        if not args.keyword_only:
            from .embed import Embedder
            embed = Embedder()
            if (args.index / "chunks.json").exists():
                previous = Index.load(args.index)
        index = Index.build(read_sources(args.source), embed=embed, previous=previous)
        index.save(args.index)
        print(f"{len(index.chunks)} chunks, {index.embedded} embedded, index in {args.index}")
        return 0

    embed = None
    if (args.index / "vectors.npy").exists() and args.mode != "bm25":
        from .embed import Embedder
        embed = Embedder()
    index = Index.load(args.index, embed)
    mode = args.mode or ("hybrid" if embed else "bm25")
    if args.command == "search":
        for c in index.search(args.query, mode, args.k):
            print(f"{c.path}:{c.start}-{c.end}  {c.symbol}")
        return 0
    chosen = pack(index.search(args.query, mode, 50), args.budget)
    print(render(chosen))
    print(f"\n{len(chosen)} chunks, about {sum(estimate_tokens(c.search_text) for c in chosen)} of {args.budget} tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
