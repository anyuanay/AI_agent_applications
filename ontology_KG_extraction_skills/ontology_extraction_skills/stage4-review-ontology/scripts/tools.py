"""The tools the review agent calls (it tests, it does not guess).

The article gives the agent four tools; this module implements the three
deterministic ones (the fourth, the ontology editor, is the set of Refine edits in
``review_ontology.py``):

  - **grounding-check**: is an element grounded *by its kind*? An extracted
    concept is grounded by a recorded mention (``mention_count > 0``); a coined
    parent by its grounded children; a relationship by its predicate.
  - **corpus search**: re-ground a concept whose mention count is zero by looking
    for any of its surfaces in the source text. This is how a borderline concept
    earns its place rather than being rejected outright.
  - **reasoner**: the lightweight consistency check (cycles + disjointness),
    re-run after every edit, repairing by relaxing the most suspect axiom first.

Grounding by kind is the whole point: what counts as justification depends on what
the element is.
"""

from __future__ import annotations

import re


class Corpus:
    """A searchable view of the source text for the corpus-search tool."""

    def __init__(self, source: str):
        self.lc = source.casefold()

    def contains(self, surface: str) -> bool:
        s = (surface or "").strip().casefold()
        if not s:
            return False
        return s in self.lc

    def any_surface(self, surfaces: list[str]) -> bool:
        return any(self.contains(s) for s in surfaces)


def concept_grounded(node: dict, corpus: "Corpus") -> tuple[bool, str]:
    """Grounding-check for an extracted concept.

    Grounded if it has a recorded literal mention (``mention_count > 0``), or if
    the corpus-search tool finds one of its surfaces verbatim. Returns
    (grounded, how).
    """
    if node.get("mention_count", 0) > 0:
        return True, "mention"
    surfaces = list(node.get("alt_labels") or [])
    if corpus.any_surface(surfaces):
        return True, "corpus_search"
    return False, "none"


# --- lightweight reasoner -------------------------------------------------

def build_parent_map(classes: list[dict], coined: list[dict]) -> dict[str, set]:
    pmap: dict[str, set] = {}
    for c in classes:
        pmap.setdefault(c["id"], set())
        for e in c.get("parents", []):
            pmap[c["id"]].add(e["parent"])
    for cp in coined:
        pmap.setdefault(cp["id"], set())
        for ch in cp.get("children", []):
            pmap.setdefault(ch, set()).add(cp["id"])
    return pmap


def ancestors(node: str, pmap: dict[str, set], cache: dict[str, set]) -> set:
    if node in cache:
        return cache[node]
    seen: set = set()
    stack = list(pmap.get(node, ()))
    while stack:
        p = stack.pop()
        if p in seen or p == node:
            continue
        seen.add(p)
        stack.extend(pmap.get(p, ()))
    cache[node] = seen
    return seen


def find_cycles(pmap: dict[str, set]) -> list[list[str]]:
    """Return any subClassOf cycles (each as a node path). Should be empty."""
    color: dict[str, int] = {}
    cycles: list[list[str]] = []

    def visit(n: str, path: list[str]):
        color[n] = 1
        path.append(n)
        for p in list(pmap.get(n, ())):
            if color.get(p, 0) == 1:
                cycles.append(path[path.index(p):] + [p])
            elif color.get(p, 0) == 0:
                visit(p, path)
        path.pop()
        color[n] = 2

    for n in list(pmap.keys()):
        if color.get(n, 0) == 0:
            visit(n, [])
    return cycles


def relax_contradictory_disjointness(axioms: list[dict],
                                     pmap: dict[str, set]) -> list[dict]:
    """Mark any disjointness axiom that contradicts the hierarchy as relaxed.

    ``A disjointWith B`` contradicts the DAG if one is an ancestor of the other. A
    coined sibling disjointness is the most suspect axiom, so it is relaxed (not
    the hierarchy). Returns the issues found.
    """
    issues: list[dict] = []
    cache: dict[str, set] = {}
    for ax in axioms:
        if ax.get("type") != "disjointWith" or len(ax.get("classes", [])) != 2:
            continue
        a, b = ax["classes"]
        if b in ancestors(a, pmap, cache) or a in ancestors(b, pmap, cache):
            ax["relaxed"] = True
            ax["flagged"] = True
            issues.append({"type": "disjoint_contradiction", "classes": [a, b]})
    return issues
