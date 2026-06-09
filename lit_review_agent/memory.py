"""
The agent's memory layer (Part 6).

Working memory is the context window (the `messages` list in the harness).
Everything here is *storage*: a persistent vector store the agent loads from and
writes to across runs. `remember`/`recall` are the two tools that bridge the
tiers; `trim_for_window` is the short-term move that pages a bulky tool result
out to a side store before it bloats the window.
"""

from __future__ import annotations

import json
from pathlib import Path

from embeddings import embed
from textutil import chunk, reorder_for_window
from vector_store import VectorStore

STORE_PATH = Path(__file__).parent / "memory_store.json"

# Side store for full paper bodies paged out of the window (short-term memory).
PAPER_STORE: dict[str, dict] = {}


class PaperMemory:
    """A thin persistent memory over a vector store (Part 6, Section 9)."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store  # persisted: survives across runs

    def remember(self, note: str, paper_id: str, metadata: dict | None = None) -> str:
        """Store a one-line finding so future sessions can retrieve it."""
        for piece in chunk(note):  # §4: right-sized chunks
            vec = embed(piece)  # §5: one model, both sides
            self.store.add(vec, text=piece, id=paper_id, meta=metadata or {})
        return f"Remembered: {note}"

    def recall(self, query: str, k: int = 5) -> str:
        """Retrieve relevant findings from PAST sessions.
        Use at the START of a task before searching from scratch."""
        vec = embed(query)  # §5: same embed model
        hits = self.store.search(vec, k=k)  # §3: top-k similarity
        hits = reorder_for_window(hits)  # §6: lost-in-the-middle
        return json.dumps([{"id": h.id, "note": h.text} for h in hits])


def trim_for_window(tool_name: str, result: str) -> str:
    """Trim a bulky tool result before it enters the window (Part 6, Section 2).

    Full paper text goes to `PAPER_STORE` (storage); a compact note goes into
    the window (working memory). The agent can re-fetch the full body by ID.
    """
    if tool_name == "fetch_paper":
        try:
            paper = json.loads(result)
        except json.JSONDecodeError:
            return result
        if "id" not in paper:
            return result
        PAPER_STORE[paper["id"]] = paper  # full text -> storage
        abstract = paper.get("abstract") or ""
        return json.dumps({  # compact note -> window
            "id": paper["id"],
            "title": paper.get("title"),
            "year": paper.get("year"),
            "citations": paper.get("citations"),
            "abstract_first_line": abstract.split(". ")[0],
        })
    return result


# One shared memory instance, lazily constructed so importing this module is cheap.
_MEMORY: PaperMemory | None = None


def default_memory() -> PaperMemory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = PaperMemory(VectorStore(STORE_PATH))
    return _MEMORY
