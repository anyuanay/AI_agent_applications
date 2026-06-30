"""Semantic embeddings for meaning-level grouping (OpenAI).

Stage 2 groups synonymous mentions **by meaning**, not by string. The signal it
uses is the one the method calls for: an embedding of each candidate *together
with its surrounding context*. Context is what lets an abbreviation meet its
expansion: ``IC`` and ``incident commander`` look nothing alike as strings, but
the sentences they appear in do, so their context-enriched embeddings land close
together.

Embeddings are computed with OpenAI's ``text-embedding-3-small`` (1536 dims by
default; the model supports a shorter ``dimensions`` projection). The API key is
read from a ``.env`` file (``OPENAI_API_KEY``). Vectors are cached on disk so
re-running at a different merge threshold never re-embeds.

A run with ``--no-embeddings`` skips this module entirely and falls back to
lexical grouping; see ``group.py``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536          # model default; the API also accepts shorter dims


def load_api_key(env_path: Optional[Path] = None) -> str:
    """Read OPENAI_API_KEY from the environment or a .env file.

    Search order: explicit ``env_path`` -> ``OPENAI_API_KEY`` already in the
    environment -> the nearest ``.env`` walking up from this file.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")

    for c in candidates:
        if c and c.is_file():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    raise RuntimeError(
        "OPENAI_API_KEY not found in environment or any .env file. "
        "Add OPENAI_API_KEY=... to the skill's .env, or run with --no-embeddings."
    )


def context_text(candidate: dict, source: str, n_windows: int = 3,
                 width: int = 60) -> str:
    """Build the text to embed for one candidate: its surface plus local context.

    For a grounded candidate we take up to ``n_windows`` occurrences and splice a
    ``width``-character window around each into one string. Context is what makes
    grouping meaning-aware rather than string-aware. An implicit candidate (no
    literal occurrence) embeds on its surface alone, which is the best signal it
    has.
    """
    surface = candidate.get("canonical") or candidate.get("key") or ""
    occ = candidate.get("occurrences") or []
    if not occ:
        return surface
    windows = []
    for o in occ[:n_windows]:
        s = max(0, int(o["start"]) - width)
        e = min(len(source), int(o["end"]) + width)
        snippet = " ".join(source[s:e].split())
        windows.append(snippet)
    return f"{surface} || " + " ... ".join(windows)


class EmbeddingCache:
    """On-disk cache of text -> vector, so re-runs never re-embed.

    Keyed by a hash of ``(model, dim, text)`` and stored as a single ``.npz`` (a
    hash list plus a stacked vector array). The cache makes threshold tuning
    instant: embed once, re-cluster for free.
    """

    def __init__(self, path: Optional[Path], model: str, dim: int):
        self.path = Path(path) if path else None
        self.model = model
        self.dim = dim
        self._store: dict[str, np.ndarray] = {}
        if self.path and self.path.is_file():
            try:
                data = np.load(self.path, allow_pickle=False)
                for h, v in zip(data["hashes"], data["vectors"]):
                    self._store[str(h)] = v
                print(f"    [cache] loaded {len(self._store)} vectors "
                      f"from {self.path.name}")
            except Exception as exc:  # noqa: BLE001 - a bad cache is not fatal
                print(f"    [cache] could not read {self.path.name}: "
                      f"{str(exc)[:80]}")

    def key(self, text: str) -> str:
        h = hashlib.sha1(f"{self.model}|{self.dim}|{text}".encode("utf-8"))
        return h.hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._store.get(self.key(text))

    def put(self, text: str, vec: np.ndarray) -> None:
        self._store[self.key(text)] = vec

    def save(self) -> None:
        if not self.path or not self._store:
            return
        hashes = np.array(list(self._store.keys()))
        vectors = np.vstack(list(self._store.values())).astype(np.float32)
        try:
            np.savez(self.path, hashes=hashes, vectors=vectors)
        except Exception as exc:  # noqa: BLE001
            print(f"    [cache] could not write {self.path.name}: {str(exc)[:80]}")


class Embedder:
    """OpenAI embedding client with batching, on-disk cache, and 429 backoff.

    OpenAI accepts up to 2048 inputs per request, so a few large batches cover the
    whole corpus well within Tier-1 limits (3,000 RPM / 1,000,000 TPM). A rate
    limit is transient: we wait the suggested delay and retry the same batch.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 dim: int = DEFAULT_DIM, batch_size: int = 256,
                 delay: float = 0.0, max_retries: int = 6,
                 cache: Optional["EmbeddingCache"] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.delay = delay
        self.max_retries = max_retries
        self.cache = cache

    def _create(self, batch: list[str]):
        # text-embedding-3-* accept a `dimensions` arg; pass it only when set.
        kwargs = dict(model=self.model, input=batch)
        if self.dim:
            kwargs["dimensions"] = self.dim
        return self.client.embeddings.create(**kwargs)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from openai import RateLimitError, APIError, APITimeoutError

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._create(texts)
                # API returns items in input order, but sort by index to be safe.
                items = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in items]
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_exc = exc
                wait = min(2.0 ** attempt, 60.0)
                ra = getattr(getattr(exc, "response", None), "headers", None)
                if ra and ra.get("retry-after"):
                    try:
                        wait = min(float(ra.get("retry-after")) + 1.0, 65.0)
                    except (TypeError, ValueError):
                        pass
                print(f"    [embed] {type(exc).__name__}; waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001 - isolate a bad item by splitting
                last_exc = exc
                if len(texts) > 1:
                    mid = len(texts) // 2
                    return (self._embed_batch(texts[:mid])
                            + self._embed_batch(texts[mid:]))
                print(f"    [embed] giving up on 1 text: {str(exc)[:120]}")
                return [[0.0] * self.dim]
        print(f"    [embed] exhausted retries on {len(texts)} texts: "
              f"{str(last_exc)[:120]}")
        return [[0.0] * self.dim for _ in texts]

    def embed(self, texts: list[str], progress: bool = True) -> np.ndarray:
        """Embed all texts, returning an (n, dim) float32 array (L2-normalized).

        Cached texts are served from disk; only cache misses cost an API request.
        """
        n = len(texts)
        out: list[Optional[np.ndarray]] = [None] * n

        miss_idx: list[int] = []
        miss_text_to_rows: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            cached = self.cache.get(t) if self.cache else None
            if cached is not None:
                out[i] = cached
            else:
                if t not in miss_text_to_rows:
                    miss_idx.append(i)
                miss_text_to_rows.setdefault(t, []).append(i)

        unique_misses = [texts[i] for i in miss_idx]
        if self.cache and (n - len(unique_misses)):
            print(f"    [cache] {n - len(unique_misses)} hit, "
                  f"{len(unique_misses)} to embed")

        done = 0
        for j in range(0, len(unique_misses), self.batch_size):
            batch = unique_misses[j:j + self.batch_size]
            vecs = self._embed_batch(batch)
            for t, v in zip(batch, vecs):
                arr = np.asarray(v, dtype=np.float32)
                if self.cache:
                    self.cache.put(t, arr)
                for row in miss_text_to_rows[t]:
                    out[row] = arr
            done += len(batch)
            if progress:
                print(f"    embedded {done}/{len(unique_misses)}")
            if self.delay:
                time.sleep(self.delay)

        if self.cache:
            self.cache.save()

        arr = np.asarray([o if o is not None else np.zeros(self.dim, np.float32)
                          for o in out], dtype=np.float32)
        # text-embedding-3 returns unit vectors; normalize defensively so cosine
        # distance reduces to a dot product downstream.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
