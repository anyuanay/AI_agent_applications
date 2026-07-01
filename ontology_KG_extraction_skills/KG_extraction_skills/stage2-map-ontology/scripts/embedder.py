"""OpenAI text-embedding-3-small wrapper with in-memory caching and batch support."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "text-embedding-3-small"
_API_BATCH = 500     # safe upper bound per OpenAI request


def load_api_key(env_path: Optional[Path] = None) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")

    for c in candidates:
        if c.is_file():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    raise RuntimeError(
        "OPENAI_API_KEY not found in environment or any .env file. "
        "Add it to .env or pass --env /path/to/.env"
    )


def cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class Embedder:
    """Embeds texts via OpenAI text-embedding-3-small with in-memory caching."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model  = model
        self._cache: dict[str, list[float]] = {}
        self._calls  = 0
        self._cached = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one embedding per text (cache-aware)."""
        results: list[Optional[list[float]]] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
                self._cached += 1
            else:
                to_embed.append((i, t))

        for batch_start in range(0, len(to_embed), _API_BATCH):
            batch = to_embed[batch_start : batch_start + _API_BATCH]
            raw   = [t for _, t in batch]
            resp  = self._client.embeddings.create(model=self._model, input=raw)
            self._calls += 1
            for (orig_idx, text), item in zip(batch, resp.data):
                emb = item.embedding
                self._cache[text] = emb
                results[orig_idx]  = emb

        return results  # type: ignore[return-value]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def top_match(
        self,
        query_emb: list[float],
        candidate_embs: list[list[float]],
    ) -> tuple[int, float]:
        """Return (index, cosine_score) of the best-matching candidate."""
        best_idx, best_score = 0, -1.0
        for i, emb in enumerate(candidate_embs):
            s = cosine(query_emb, emb)
            if s > best_score:
                best_score, best_idx = s, i
        return best_idx, best_score

    def stats(self) -> dict:
        return {"api_calls": self._calls, "cache_hits": self._cached,
                "unique_texts_cached": len(self._cache)}
