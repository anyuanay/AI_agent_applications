"""
Graph query tools (Part 6, Section 8).

Two flavours, framed as the Part 5 enum-vs-free-text tradeoff:
  - typed traversals (preferred, safer): papers_citing, papers_bridging_topics,
    citation_path — a constrained surface the model is unlikely to misuse;
  - a flexible passthrough (powerful, riskier): query_graph.

Plus hybrid_recall: vector finds the entry nodes, the graph expands the
neighborhood (GraphRAG).
"""

from __future__ import annotations

import json

from embeddings import embed
from graph import GRAPH


# ----- typed traversals (preferred) -----
def papers_citing(paper_ids: list[str], year: str | None = None) -> str:
    """Papers that cite ALL of the given papers, newest first.
    Use to find recent work built on foundations you already have."""
    return json.dumps(GRAPH.citing(paper_ids, year=year)[:20])


def papers_bridging_topics(topic_a: str, topic_b: str) -> str:
    """Papers tagged with BOTH topics — the work connecting two areas.
    Use to find cross-over papers."""
    return json.dumps(GRAPH.bridging(topic_a, topic_b)[:20])


def citation_path(from_id: str, to_id: str) -> str:
    """Shortest CITES path between two papers, if one exists.
    Use to trace how an idea propagated."""
    return json.dumps(GRAPH.shortest_path(from_id, to_id, rel="CITES"))


# ----- flexible passthrough (riskier) -----
def query_graph(cypher: str) -> str:
    """Run a read-only Cypher query against the paper graph.
    Use for open-ended relational questions the typed tools do not cover."""
    try:
        rows = GRAPH.run(cypher, read_only=True, timeout=5)
        return json.dumps(rows[:25])  # cap rows -> cap tokens
    except NotImplementedError as e:
        return json.dumps({"error": "unsupported", "message": str(e)})


# ----- hybrid GraphRAG -----
def hybrid_recall(query: str, k: int = 3) -> str:
    """Vector finds the door, the graph walks the rooms."""
    from memory import default_memory  # lazy import; memory owns the store

    seeds = default_memory().store.search(embed(query), k=k)
    ids = [s.id for s in seeds]
    nbrs = GRAPH.neighborhood(ids, hops=1)
    return json.dumps({"seeds": ids, "context": nbrs})
