"""
Embeddings (Part 6).

The principle from Part 6 that actually matters: use the *same* embedding model
on both sides (ingestion and query), and let the domain — not the dimension —
drive the choice. The function below is a small, deterministic, dependency-free
stand-in so the vector store, RAG path, and eval suite run offline and
reproducibly. In a real deployment you would swap `embed` for a sentence-
transformer or a provider embedding endpoint; nothing else in the codebase
changes, which is the whole point of keeping one embed function behind one name.
"""

from __future__ import annotations

import hashlib
import math
import re

DIM = 256
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stable_hash(salt: str, tok: str) -> int:
    # Process-stable (unlike builtin hash()), so vectors persist across runs.
    digest = hashlib.md5(f"{salt}:{tok}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def embed(text: str) -> list[float]:
    """Hashed bag-of-words → L2-normalized vector. Same model both sides."""
    vec = [0.0] * DIM
    for tok in _tokens(text):
        # Two hashes per token reduce collision artifacts a little.
        vec[_stable_hash("a", tok) % DIM] += 1.0
        vec[_stable_hash("b", tok) % DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
