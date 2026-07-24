"""Taxonomy utilities and a lightweight reasoner.

Two jobs:

  1. **DAG utilities** the rest of Stage 3 needs: transitive ancestors, node
     depth, and the lowest common ancestor (LCA) used to lift a relation's
     domain/range to "the most general class that is still correct".

  2. **A lightweight consistency check** that closes the stage. A full OWL
     reasoner (HermiT/ELK) needs Java; here we check the two contradictions Stage
     3 can actually produce: a cycle in the ``subClassOf`` DAG (which makes its
     classes mutually sub/super, i.e. unsatisfiable) and a disjointness axiom that
     contradicts the subsumption it sits in. Following the article, an
     inconsistency is repaired by **relaxing the most suspect axiom first** (a
     coined disjointness before an extracted subClassOf), never by silently
     dropping a class. Every relaxation is recorded.
"""

from __future__ import annotations

from model import Axiom


def build_parent_map(classes: list, coined: list) -> dict[str, set[str]]:
    """child_id -> set(parent_id), across extracted parent edges and coined families."""
    pmap: dict[str, set[str]] = {}
    for c in classes:
        pmap.setdefault(c.id, set())
        for e in c.parents:
            pmap[c.id].add(e.parent)
    for cp in coined:
        pmap.setdefault(cp.id, set())
        for child in cp.children:
            pmap.setdefault(child, set()).add(cp.id)
    return pmap


def ancestors(node: str, pmap: dict[str, set[str]],
              _cache: dict[str, set[str]] | None = None) -> set[str]:
    """All transitive ancestors of ``node`` (not including itself). Cycle-safe."""
    if _cache is not None and node in _cache:
        return _cache[node]
    seen: set[str] = set()
    stack = list(pmap.get(node, ()))
    while stack:
        p = stack.pop()
        if p in seen or p == node:
            continue
        seen.add(p)
        stack.extend(pmap.get(p, ()))
    if _cache is not None:
        _cache[node] = seen
    return seen


def depth(node: str, pmap: dict[str, set[str]], _cache: dict[str, int] | None = None,
          _stack: frozenset | None = None) -> int:
    """Longest distance from ``node`` up to a root (root depth 0). Cycle-safe."""
    _cache = _cache if _cache is not None else {}
    _stack = _stack or frozenset()
    if node in _cache:
        return _cache[node]
    parents = [p for p in pmap.get(node, ()) if p not in _stack and p != node]
    if not parents:
        return 0
    d = 1 + max(depth(p, pmap, _cache, _stack | {node}) for p in parents)
    _cache[node] = d
    return d


def lca(nodes: list[str], pmap: dict[str, set[str]]) -> str | None:
    """Lowest common ancestor of ``nodes``: the most specific class covering all.

    Each node counts as an ancestor of itself, so the LCA of a single node is that
    node, and the LCA of siblings is their shared parent. Returns None when the
    nodes share no ancestor at all.
    """
    nodes = [n for n in nodes if n]
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    acache: dict[str, set[str]] = {}
    sets = []
    for n in nodes:
        sets.append({n} | ancestors(n, pmap, acache))
    common = set.intersection(*sets) if sets else set()
    if not common:
        return None
    dcache: dict[str, int] = {}
    return max(common, key=lambda c: depth(c, pmap, dcache))


def break_cycles(classes: list, pmap: dict[str, set[str]]) -> list[dict]:
    """Detect ``subClassOf`` cycles and break each by dropping its weakest edge.

    The weakest edge is the lowest-confidence parent edge on a node in the cycle.
    Returns a list of issue records (the cycle and the edge removed).
    """
    issues: list[dict] = []
    color: dict[str, int] = {}      # 0=unvisited,1=in-stack,2=done
    by_id = {c.id: c for c in classes}

    def visit(node: str, path: list[str]):
        color[node] = 1
        path.append(node)
        for p in list(pmap.get(node, ())):
            if color.get(p, 0) == 1:
                # Found a cycle: the slice of path from p onward, plus node->p.
                cyc = path[path.index(p):] + [p]
                _drop_weakest(cyc)
                return
            if color.get(p, 0) == 0:
                visit(p, path)
        path.pop()
        color[node] = 2

    def _drop_weakest(cycle: list[str]):
        # Consider the edges (child->parent) along the cycle; drop the weakest
        # that we own as an extracted parent edge.
        worst = None  # (confidence, child_id, parent_id)
        for i in range(len(cycle) - 1):
            child, parent = cycle[i], cycle[i + 1]
            c = by_id.get(child)
            if not c:
                continue
            for e in c.parents:
                if e.parent == parent:
                    if worst is None or e.confidence < worst[0]:
                        worst = (e.confidence, child, parent)
        if worst is None:
            return
        _, child, parent = worst
        c = by_id[child]
        c.parents = [e for e in c.parents if e.parent != parent]
        pmap.get(child, set()).discard(parent)
        c.flags.append("cycle_edge_removed")
        issues.append({"type": "cycle", "cycle": cycle,
                       "removed_edge": [child, parent]})

    for cid in list(pmap.keys()):
        if color.get(cid, 0) == 0:
            visit(cid, [])
    return issues


def check_disjointness(axioms: list[Axiom], pmap: dict[str, set[str]]) -> list[dict]:
    """Relax any disjointness axiom that contradicts the subsumption hierarchy.

    A ``A disjointWith B`` is contradictory if one is an ancestor of the other.
    Such an axiom is the most suspect (it is a coined sibling disjointness), so we
    relax it (mark ``relaxed``) rather than touch the class hierarchy.
    """
    issues: list[dict] = []
    acache: dict[str, set[str]] = {}
    for ax in axioms:
        if ax.type != "disjointWith" or len(ax.classes) != 2:
            continue
        a, b = ax.classes
        if b in ancestors(a, pmap, acache) or a in ancestors(b, pmap, acache):
            ax.relaxed = True
            ax.flagged = True
            issues.append({"type": "disjoint_contradiction", "classes": [a, b]})
    return issues


def disjoint_axioms_for_family(child_ids: list[str], cap: int = 6) -> list[Axiom]:
    """Pairwise disjointness among the children of a coined family.

    Emitted only for small families (the clear sibling case); flagged as suspect
    so the reasoner relaxes it first if it ever contradicts the hierarchy.
    """
    if not (2 <= len(child_ids) <= cap):
        return []
    out: list[Axiom] = []
    for i in range(len(child_ids)):
        for j in range(i + 1, len(child_ids)):
            out.append(Axiom(type="disjointWith",
                             classes=[child_ids[i], child_ids[j]],
                             source="coined_family_siblings", flagged=True))
    return out
