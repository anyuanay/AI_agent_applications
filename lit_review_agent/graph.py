"""
The lit-review property graph (Part 6, Section 8).

Node types:  Paper, Author, Venue, Topic
Edge types:  CITES (Paper->Paper), WROTE (Author->Paper),
             PUBLISHED_IN (Paper->Venue), ON_TOPIC (Paper->Topic)

This is an in-memory graph — enough to make the typed traversals real and the
token-efficiency argument concrete. In a real deployment this is Neo4j or a
property-graph service, and `run()` is where a Cypher passthrough would land.
Part 14 names this schema for what it already is: an ontology (the types and
legal relations) populated by instances (the graph).
"""

from __future__ import annotations

from collections import defaultdict, deque


class PaperGraph:
    def __init__(self) -> None:
        # node key -> attributes. Keys are namespaced, e.g. "Paper:abc", "Topic:gnn".
        self.nodes: dict[str, dict] = {}
        # adjacency: rel -> {src_key -> set(dst_key)}
        self.out: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.inc: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    # ----- ingestion -----
    def add_node(self, ntype: str, key: str, **attrs) -> str:
        nkey = f"{ntype}:{key}"
        node = self.nodes.setdefault(nkey, {"type": ntype, "key": key})
        node.update({k: v for k, v in attrs.items() if v is not None})
        return nkey

    def add_edge(self, rel: str, src: str, dst: str) -> None:
        self.out[rel][src].add(dst)
        self.inc[rel][dst].add(src)

    def add_paper(self, paper: dict) -> str:
        """Ingest a paper dict (Semantic Scholar shape) and its relations."""
        pid = paper.get("id") or paper.get("paperId")
        pkey = self.add_node(
            "Paper", pid,
            title=paper.get("title"),
            year=paper.get("year"),
            citations=paper.get("citations", paper.get("citationCount")),
        )
        for name in paper.get("authors", []) or []:
            akey = self.add_node("Author", name, name=name)
            self.add_edge("WROTE", akey, pkey)
        venue = paper.get("venue")
        if venue:
            vkey = self.add_node("Venue", venue, name=venue)
            self.add_edge("PUBLISHED_IN", pkey, vkey)
        for topic in paper.get("topics", []) or []:
            tkey = self.add_node("Topic", topic, name=topic)
            self.add_edge("ON_TOPIC", pkey, tkey)
        for cited in paper.get("references", []) or []:
            # CITES edges are free from Semantic Scholar references.
            ckey = self.add_node("Paper", cited, title=None)
            self.add_edge("CITES", pkey, ckey)
        return pkey

    # ----- typed traversals (preferred, safe; Part 6 §8) -----
    def citing(self, paper_ids: list[str], year: str | None = None) -> list[dict]:
        """Papers that cite ALL of the given papers, newest first."""
        targets = [f"Paper:{pid}" for pid in paper_ids]
        citers: set | None = None
        for t in targets:
            who = self.inc["CITES"].get(t, set())
            citers = who if citers is None else (citers & who)
        rows = [self.nodes[c] for c in (citers or set()) if c in self.nodes]
        if year:
            rows = [r for r in rows if _year_match(r.get("year"), year)]
        rows.sort(key=lambda r: r.get("year") or 0, reverse=True)
        return [_paper_row(r) for r in rows]

    def bridging(self, topic_a: str, topic_b: str) -> list[dict]:
        """Papers tagged with BOTH topics — the work connecting two areas."""
        a = self.inc["ON_TOPIC"].get(f"Topic:{topic_a}", set())
        b = self.inc["ON_TOPIC"].get(f"Topic:{topic_b}", set())
        return [_paper_row(self.nodes[p]) for p in (a & b) if p in self.nodes]

    def shortest_path(self, from_id: str, to_id: str, rel: str = "CITES") -> list[str]:
        """Shortest path between two papers along `rel`, if one exists."""
        start, goal = f"Paper:{from_id}", f"Paper:{to_id}"
        if start not in self.nodes or goal not in self.nodes:
            return []
        seen = {start}
        queue: deque[list[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return [n.split(":", 1)[1] for n in path]
            for nxt in self.out[rel].get(node, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(path + [nxt])
        return []

    def neighborhood(self, paper_ids: list[str], hops: int = 1) -> list[dict]:
        """All papers within `hops` of the seeds along any relation."""
        frontier = {f"Paper:{pid}" for pid in paper_ids if f"Paper:{pid}" in self.nodes}
        seen = set(frontier)
        for _ in range(hops):
            nxt: set = set()
            for node in frontier:
                for rel in self.out:
                    nxt |= self.out[rel].get(node, set())
                for rel in self.inc:
                    nxt |= self.inc[rel].get(node, set())
            frontier = nxt - seen
            seen |= nxt
        return [
            _paper_row(self.nodes[n])
            for n in seen
            if n in self.nodes and self.nodes[n]["type"] == "Paper"
        ]

    def run(self, cypher: str, read_only: bool = True, timeout: int = 5):
        """Flexible Cypher passthrough.

        Faithful to Part 6: the typed traversals above are preferred and safe.
        The free-text passthrough requires a real graph backend (Neo4j); here it
        returns a structured 'unsupported' result rather than pretending. This is
        the enum-vs-free-text tradeoff from Part 5 made literal.
        """
        raise NotImplementedError(
            "query_graph requires a Cypher backend; prefer the typed tools "
            "(papers_citing, papers_bridging_topics, citation_path)."
        )


def _year_match(year, spec: str) -> bool:
    if year is None:
        return False
    if "-" in spec:
        lo, hi = spec.split("-")
        return int(lo) <= int(year) <= int(hi)
    return str(year) == spec


def _paper_row(node: dict) -> dict:
    return {
        "id": node.get("key"),
        "title": node.get("title"),
        "year": node.get("year"),
        "citations": node.get("citations"),
    }


# One shared graph instance the tools ingest into and the graph tools read from.
GRAPH = PaperGraph()
