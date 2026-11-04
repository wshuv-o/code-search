# code-search

Retrieval for code, with an eval that decides what's worth keeping. It indexes Python source, searches it with keywords, local embeddings or both, and packs the results into a fixed context budget for a model. It needs no API key and no paid service: embeddings run on CPU with ONNX Runtime, so code never leaves the machine.

```bash
pip install -r requirements.txt
python -m codesearch index path/to/source
python -m codesearch search "where do we retry failed uploads"
python -m codesearch context "where do we retry failed uploads" --budget 2000
```

Add `--keyword-only` to `index` to skip the 23 MB model download. Re-indexing only embeds chunks whose text changed.

## Results

110 labelled queries over 190 files of pinned CPython 3.12.8 source (9,260 chunks). These are the 51 test queries, which played no part in tuning. "In context" means the right code made it into a 2,000-token pack, which is what a model would actually receive.

| Setup | MRR@10 | Hit@5 | Hit@10 | In context | p95 ms |
|---|---|---|---|---|---|
| Fixed windows, plain keywords | 0.319 | 0.39 | 0.43 | 0.37 | 6.7 |
| Whole files, code keywords | 0.539 | 0.63 | 0.76 | **0.02** | 0.5 |
| Fixed windows, code keywords | 0.381 | 0.43 | 0.51 | 0.43 | 7.0 |
| AST chunks, plain keywords | 0.305 | 0.43 | 0.49 | 0.41 | 6.4 |
| AST chunks, code keywords (BM25) | 0.351 | 0.45 | 0.51 | 0.49 | 6.3 |
| AST chunks, embeddings | 0.450 | 0.63 | 0.71 | 0.73 | 4.3 |
| AST chunks, hybrid | 0.456 | 0.63 | 0.76 | 0.71 | 14.7 |

| Query type | BM25 MRR | Embeddings MRR | Hybrid MRR |
|---|---|---|---|
| Paraphrase, avoids the code's own words (44) | 0.270 | 0.407 | 0.415 |
| Identifier, names the code (7) | 0.857 | 0.714 | 0.714 |

Hybrid minus BM25, MRR: +0.105, 95% CI -0.007 to +0.218 (paired bootstrap over queries).

What this shows:

- **Ranking alone is misleading.** Whole files have the best MRR, because a file is an easy target to hit, and are nearly useless in practice: the answer fits in the budget 2% of the time. That is why the eval measures the pack as well as the ranking.
- **Embeddings help most when the query doesn't use the code's words**, and keywords win when it does. Neither is enough alone.
- **Hybrid is the default because it is close to the best on both query types, not because it wins.** It does not beat embeddings alone here, and its lead over BM25 falls just short of significance on this many queries. It costs about 3x the query latency of embeddings alone.
- **AST chunks don't beat fixed windows on rank in this data.** Windows have the higher MRR with code keywords on test (0.381 against 0.351), the reverse of dev. AST chunks do fit the budget more often and give exact symbols, but this eval doesn't show a clear ranking gain.

## How the eval stays honest

- **Split by hash.** Dev or test comes from a hash of the query text, so nobody chooses which queries are in the test set.
- **Tuned on dev only.** A dev comparison led to one change: counting the symbol name more heavily (identifier MRR 0.72 to 0.96 on dev). Stemming and stopword removal didn't help and were left out. After the test run nothing was tuned, and the floors below were set from it.
- **One relevance rule for every chunker.** A chunk counts if it overlaps at least half of the target or half of the chunk, whichever is smaller.
- **Labels are checked.** The eval refuses to run if a labelled symbol isn't in the corpus.
- **Everything is pinned.** The corpus and model are fetched by exact version and rejected if the SHA-256 differs.
- **CI gate.** `evals/run.py --split test --gate` fails below fixed floors. They guard against regressions, and because they were set from this test run, passing them isn't fresh evidence of quality.

## Limits

- The queries were written by the author with Claude, knowing the code, not taken from real user logs. Treat the absolute numbers as a comparison between setups on this set, not as expected production performance.
- 7 identifier queries in the test split is few, so that row is only a rough signal.
- One corpus. CPython's standard library is well named and well documented, and a messier codebase may score differently.
- AST chunking is Python-only; other files and unparseable Python fall back to line windows, and `index` currently reads `*.py` only.
- Small comment-only chunks can use up part of a tight budget, since packing has no minimum-usefulness check.
- Search is brute force over vectors in memory. That is fine for thousands of chunks and not designed for millions. There is no reranker, and no eval of a model's final answers.

## Layout

| Path | What it does |
|---|---|
| `codesearch/chunking.py` | AST, window and whole-file chunkers |
| `codesearch/tokens.py`, `bm25.py` | Code-aware tokenizer and BM25 |
| `codesearch/embed.py` | Pinned model download and ONNX embedding |
| `codesearch/index.py` | Incremental build, fusion, context packing |
| `evals/queries.yaml`, `evals/run.py` | Labelled queries, comparison and gate |
| `tests/` | Unit tests; embeddings are faked, so no downloads |
