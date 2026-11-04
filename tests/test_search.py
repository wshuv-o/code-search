"""No downloads: embeddings are faked, so these run anywhere."""
import numpy as np
import pytest

from codesearch.bm25 import BM25
from codesearch.chunking import MAX_LINES, ast_chunks, file_chunks, module_name, symbol_ranges, window_chunks
from codesearch.embed import fetch, sha256
from codesearch.index import Index, estimate_tokens, fuse, pack, render
from codesearch.tokens import code_tokens, naive_tokens
from evals.run import Query, bootstrap_diff, first_hit, is_relevant, load_queries

SOURCE = '''"""Module docstring."""
import os

LIMIT = 3


@decorated
def top(a):
    return a


class Thing:
    """Holds things."""
    size = 1

    def method(self):
        return 1

    class Inner:
        def deep(self):
            return 2


TRAILER = 9
'''


class FakeEmbed:
    """Vectors from a fixed hash of the words, and a count of what it was asked to embed."""
    def __init__(self):
        self.seen = []

    def __call__(self, texts):
        self.seen.extend(texts)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in naive_tokens(text):
                out[i, sum(map(ord, word)) % 384] += 1
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


def test_tokens_split_identifiers_and_keep_the_whole_name():
    assert code_tokens("parseHTTPHeader") == ["parsehttpheader", "parse", "http", "header"]
    assert code_tokens("max_retry_count") == ["max_retry_count", "max", "retry", "count"]
    assert naive_tokens("parseHTTPHeader") == ["parsehttpheader"]


def test_module_name():
    assert module_name("http/cookies.py") == "http.cookies"
    assert module_name("json/__init__.py") == "json"


def test_ast_chunks_give_every_definition_its_own_chunk():
    symbols = [c.symbol for c in ast_chunks("pkg/mod.py", SOURCE)]
    assert symbols == ["pkg.mod", "pkg.mod.top", "pkg.mod.Thing", "pkg.mod.Thing.method", "pkg.mod.Thing.Inner", "pkg.mod.Thing.Inner.deep", "pkg.mod"]


def test_ast_chunks_lose_no_line_of_code():
    covered = set()
    for c in ast_chunks("m.py", SOURCE):
        covered.update(range(c.start, c.end + 1))
    code = {i for i, line in enumerate(SOURCE.splitlines(), 1) if line.strip()}
    assert code <= covered


def test_decorator_stays_with_its_function():
    chunk = next(c for c in ast_chunks("m.py", SOURCE) if c.symbol == "m.top")
    assert chunk.text.startswith("@decorated")


def test_oversized_function_splits_with_same_symbol_and_full_coverage():
    body = "\n".join(f"    x{i} = {i}" for i in range(400))
    chunks = ast_chunks("m.py", f"def big():\n{body}\n")
    assert len(chunks) > 1 and {c.symbol for c in chunks} == {"m.big"}
    assert all(c.end - c.start + 1 <= MAX_LINES for c in chunks)
    assert chunks[0].start == 1 and chunks[-1].end == 401
    assert all(b.start <= a.end + 1 for a, b in zip(chunks, chunks[1:]))


@pytest.mark.parametrize("path,source", [("broken.py", "def (:\n  oops\n"), ("notes.txt", "plain text\n")])
def test_unparseable_input_falls_back_to_windows(path, source):
    chunks = ast_chunks(path, source)
    assert chunks and chunks[0].start == 1


def test_window_chunks_overlap_and_reach_the_end():
    chunks = window_chunks("m.py", "\n".join(f"line{i}" for i in range(100)))
    assert chunks[-1].end == 100 and chunks[1].start == chunks[0].end - 9


def test_empty_file_gives_no_chunks():
    assert ast_chunks("m.py", "") == [] and window_chunks("m.py", "") == [] and file_chunks("m.py", "  \n") == []


def test_bm25_prefers_rare_term_and_ignores_unknown():
    docs = ["common common rare", "common common common", "common other"]
    bm25 = BM25(docs, naive_tokens)
    assert bm25.search("rare common", 3)[0][0] == 0
    assert bm25.search("absent", 3) == []


def test_bm25_finds_part_of_an_identifier_only_with_code_tokens():
    docs = ["def parseHeader(): pass", "def other(): pass"]
    assert BM25(docs, code_tokens).search("header", 2)[0][0] == 0
    assert BM25(docs, naive_tokens).search("header", 2) == []


def test_a_name_match_beats_the_same_word_used_in_a_body():
    index = Index.build({"a.py": "def other():\n    retry = 1\n    return retry\n", "b.py": "def retry():\n    return 1\n"})
    assert index.search("retry", "bm25", 1)[0].symbol == "b.retry"


def test_fusion_rewards_agreement_over_one_top_rank():
    assert fuse([[1, 2, 3], [4, 2, 5]])[0][0] == 2


def test_incremental_build_embeds_only_what_changed():
    embed = FakeEmbed()
    first = Index.build({"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"}, embed=embed)
    assert first.embedded == 2
    embed.seen.clear()
    second = Index.build({"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 99\n"}, embed=embed, previous=first)
    assert second.embedded == 1 and len(embed.seen) == 1 and "99" in embed.seen[0]
    np.testing.assert_array_equal(second.vectors[0], first.vectors[0])


def test_deleted_file_leaves_the_index():
    embed = FakeEmbed()
    first = Index.build({"a.py": "def a(): pass\n", "b.py": "def b(): pass\n"}, embed=embed)
    second = Index.build({"a.py": "def a(): pass\n"}, embed=embed, previous=first)
    assert {c.path for c in second.chunks} == {"a.py"} and second.vectors.shape[0] == 1


def test_save_and_load_give_the_same_results(tmp_path):
    embed = FakeEmbed()
    index = Index.build({"a.py": SOURCE}, embed=embed)
    index.save(tmp_path)
    loaded = Index.load(tmp_path, embed)
    assert loaded.search("method", "hybrid", 3) == index.search("method", "hybrid", 3)


def test_dense_mode_without_vectors_fails_loudly():
    with pytest.raises(ValueError, match="embeddings"):
        Index.build({"a.py": SOURCE}).search("x", "dense")


def test_unknown_mode_fails_loudly():
    with pytest.raises(ValueError, match="unknown"):
        Index.build({"a.py": SOURCE}, embed=FakeEmbed()).search("x", "typo")


def test_pack_never_exceeds_budget_and_skips_rather_than_stops():
    chunks = ast_chunks("m.py", "def big():\n" + "    x = 1\n" * 100 + "\ndef small():\n    return 2\n")
    big, small = chunks[0], chunks[1]
    budget = estimate_tokens(small.search_text) + 1
    assert pack([big, small], budget) == [small]


def test_pack_drops_overlapping_repeats():
    chunks = window_chunks("m.py", "\n".join(f"line{i}" for i in range(60)))
    assert pack(chunks[:2], 100000) == [chunks[0]]


def test_render_is_grouped_and_in_line_order():
    a, b = ast_chunks("m.py", "def a():\n    pass\n\ndef b():\n    pass\n")
    text = render([b, a])
    assert text.index("(m.a)") < text.index("(m.b)")


def test_relevance_is_fair_across_chunk_sizes():
    chunk = lambda s, e: type("C", (), {"path": "m.py", "start": s, "end": e})()
    target = ("m.py", 10, 20)
    assert is_relevant(chunk(10, 20), target) and is_relevant(chunk(1, 100), target)
    assert is_relevant(chunk(15, 40), target) and not is_relevant(chunk(19, 40), target)
    assert not is_relevant(chunk(21, 30), target)
    assert not is_relevant(type("C", (), {"path": "n.py", "start": 10, "end": 20})(), target)


def test_first_hit_is_one_indexed_and_none_when_absent():
    chunks = ast_chunks("m.py", "def a():\n    pass\n\ndef b():\n    pass\n")
    assert first_hit(chunks, [("m.py", 4, 5)]) == 2
    assert first_hit(chunks, [("z.py", 1, 2)]) is None


def test_bootstrap_interval_excludes_zero_only_for_a_real_difference():
    assert bootstrap_diff([0.0] * 50, [1.0] * 50)[1] > 0
    mean, low, high = bootstrap_diff([0.0, 1.0] * 25, [1.0, 0.0] * 25)
    assert mean == 0 and low < 0 < high


def test_split_is_stable_and_uses_both_sides():
    assert Query("same text", (), "paraphrase").split == Query("same text", ("x",), "identifier").split
    queries = load_queries()
    assert len(queries) >= 100
    assert {q.split for q in queries} == {"dev", "test"}
    assert len({q.text for q in queries}) == len(queries)


def test_symbol_ranges_include_nested_methods():
    ranges = symbol_ranges("m.py", SOURCE)
    assert ranges["m.Thing.Inner.deep"][0] > ranges["m.Thing"][0]
    assert ranges["m.top"][0] == 7


def test_download_with_the_wrong_hash_is_rejected(tmp_path):
    source = tmp_path / "src"
    source.write_bytes(b"payload")
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="sha256"):
        fetch(source.as_uri(), dest, "0" * 64)
    assert not dest.exists() and not list(tmp_path.glob("*.partial"))
    assert fetch(source.as_uri(), dest, sha256(source)).read_bytes() == b"payload"
