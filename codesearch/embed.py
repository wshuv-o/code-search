"""Local embeddings with ONNX Runtime: no API key, no per-call cost, and code never leaves the machine."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
BASE = f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/{REVISION}"
FILES = {
    "model_quantized.onnx": ("onnx/model_quantized.onnx", "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"),
    "tokenizer.json": ("tokenizer.json", "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
}
MAX_TOKENS = 256
BATCH = 32


def cache_dir() -> Path:
    return Path(os.environ.get("CODESEARCH_CACHE", ".cache"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected: str) -> Path:
    """A pinned download: a file that doesn't match its hash is deleted and never used."""
    if dest.exists() and sha256(dest) == expected:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    urllib.request.urlretrieve(url, partial)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        raise ValueError(f"{url}: expected sha256 {expected}, got {actual}")
    partial.replace(dest)
    return dest


class Embedder:
    name = f"all-MiniLM-L6-v2@{REVISION[:8]}"

    def __init__(self) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        paths = {name: fetch(f"{BASE}/{remote}", cache_dir() / name, digest) for name, (remote, digest) in FILES.items()}
        self.tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        self.session = onnxruntime.InferenceSession(str(paths["model_quantized.onnx"]), providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Unit vectors in input order. Sorting by length first keeps padding, and so the runtime, down."""
        out = np.zeros((len(texts), 384), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for at in range(0, len(order), BATCH):
            batch = order[at:at + BATCH]
            encoded = self.tokenizer.encode_batch([texts[i][:4000] for i in batch])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            out[batch] = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return out
