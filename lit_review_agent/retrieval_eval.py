"""
Retrieval evaluation (Part 6, Section 7).

The retrieval-shaped slice of evaluation, introduced in Part 6 and generalized
in Part 12. A tiny, honest harness: for each query with a known relevant set,
what fraction did the retriever return in its top k? This runs entirely offline
against the local vector store and the stand-in embeddings.

    python retrieval_eval.py
"""

from __future__ import annotations

from embeddings import embed
from vector_store import VectorStore

# A handful of cases with known-relevant ids (the gold set).
EVAL = [
    {"query": "graph neural networks for molecules",
     "relevant_ids": {"kipf2017", "gilmer2017", "ying2021"}},
    {"query": "attention mechanisms in transformers",
     "relevant_ids": {"vaswani2017"}},
]

# A small fixed corpus so the demo is self-contained.
_CORPUS = {
    "kipf2017": "Semi-supervised classification with graph convolutional networks for molecules",
    "gilmer2017": "Neural message passing for quantum chemistry and molecular property prediction",
    "ying2021": "Hierarchical graph representation learning for molecules and proteins",
    "vaswani2017": "Attention is all you need: the transformer and self-attention mechanisms",
    "he2016": "Deep residual learning for image recognition",
}


def _build_store() -> VectorStore:
    store = VectorStore()  # in-memory, no persistence for the eval
    for pid, text in _CORPUS.items():
        store.add(embed(text), text=text, id=pid)
    return store


def recall_at_k(retrieve, k: int = 5) -> float:
    scores = []
    for case in EVAL:
        got = {h.id for h in retrieve(case["query"])[:k]}
        hit = len(got & case["relevant_ids"]) / len(case["relevant_ids"])
        scores.append(hit)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    store = _build_store()
    retrieve = lambda q: store.search(embed(q), k=5)
    print(f"Recall@5 over {len(EVAL)} cases: {recall_at_k(retrieve, k=5):.2f}")
