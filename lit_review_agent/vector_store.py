"""
A tiny persistent vector store (Part 6).

Just enough to demonstrate the RAG path: add (vector, text, id, metadata),
search by cosine top-k, and persist to a JSON file so memory survives across
runs. In production this is Pinecone / pgvector / Chroma / FAISS; the interface
is the same, which is why the harness above it never has to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from embeddings import cosine


@dataclass
class Hit:
    id: str
    text: str
    score: float
    meta: dict


class VectorStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._rows: list[dict] = []
        if self.path and self.path.exists():
            self._rows = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, vec: list[float], text: str, id: str, meta: dict | None = None) -> None:
        self._rows.append({"id": id, "vec": vec, "text": text, "meta": meta or {}})
        self._persist()

    def search(self, vec: list[float], k: int = 5) -> list[Hit]:
        scored = [
            Hit(id=r["id"], text=r["text"], score=cosine(vec, r["vec"]), meta=r["meta"])
            for r in self._rows
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._rows)

    def _persist(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._rows), encoding="utf-8")
