"""Data model for Stage 3 (synthesize structure).

Stage 3 is the first stage that **commits to structure**. Where Stage 2 produced
a flat vocabulary of named concepts and relations, Stage 3 wires them into a
shape: an ``rdfs:subClassOf`` DAG, a domain and range on every relationship, and
axioms, plus a consistency report from a (lightweight) reasoner.

The model carries the three invariants forward:

  - **Recall-first.** A disagreement is never resolved by dropping an edge. It is
    kept and ``flagged`` for Stage 4 review. Orphans (no confident parent) are
    kept as roots, flagged, not force-parented.
  - **Provenance everywhere.** Every parent edge records *how* it was found
    (``via``: lexical-head / hearst / llm / cluster), and coined parents record
    their justification (children + which signals agreed).
  - **Grounding by kind.** An extracted concept is grounded by its mention; a
    coined parent is grounded by its children; a relationship by its predicate and
    a sane domain/range; an axiom by the reasoner finding no contradiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


# How a parent edge was proposed. Ordered loosely by authority: a lexical head
# ("whole number" is-a "number") or a stated Hearst is-a is stronger than an LLM
# guess; clustering coins a parent where a family has no name.
VIA_LEXICAL = "lexical_head"   # compound-head subclass: "X Y" is-a "Y"
VIA_HEARST = "hearst"          # stated is-a in the text ("Y such as X")
VIA_LLM = "llm"                # LLM-proposed parent (broad, low authority)
VIA_CLUSTER = "cluster"        # grouped by embedding into a coined family

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG.sub("-", text.casefold()).strip("-")


@dataclass
class ParentEdge:
    """One ``subClassOf`` link from a child class to a parent class.

    ``via`` records the signal that proposed it; ``flagged`` marks an edge that
    Stage 4 should look at (a disagreement between signals, or a multi-parent
    node). Nothing flagged is dropped: the uncertainty is carried forward.
    """

    parent: str                 # id of the parent class (concept:* or coined:*)
    via: list[str] = field(default_factory=list)   # subset of VIA_*
    confidence: float = 0.0
    flagged: bool = False


@dataclass
class ClassNode:
    """An extracted concept, now placed in the taxonomy."""

    id: str
    label: str
    alt_labels: list[str] = field(default_factory=list)
    mention_count: int = 0
    parents: list[ParentEdge] = field(default_factory=list)
    coined: bool = False
    orphan: bool = False        # no confident parent; a flagged root for Stage 4
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class CoinedParent:
    """A parent class coined over a family that the vocabulary cannot name.

    The last resort, used only when no existing class will accept the children. A
    coined parent is a *hypothesis*: it must earn its place in Stage 4 by its
    children, so it records exactly which children and which signals justify it.
    """

    id: str
    label: str
    children: list[str] = field(default_factory=list)
    justification: dict = field(default_factory=dict)  # n_children, agreement[...]
    coined: bool = True
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relationship:
    """A relation, now given a domain and range lifted up the taxonomy.

    Domain/range are assigned *after* the taxonomy exists, by lifting each
    relation's observed arguments to the most general class that still covers them
    (their lowest common ancestor). Thin evidence is kept and flagged, not
    discarded.
    """

    id: str
    label: str
    domain: str | None = None
    range: str | None = None
    domain_source: str = "none"     # "lifted" | "single" | "none"
    range_source: str = "none"
    observed_subjects: list[str] = field(default_factory=list)
    observed_objects: list[str] = field(default_factory=list)
    evidence_count: int = 0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Axiom:
    """A logical axiom emitted in Stage 3 and checked by the reasoner.

    Currently disjointness among the children of a coined family (the clearest
    sibling case). ``relaxed`` records an axiom the reasoner had to drop to
    restore consistency, kept for the audit trail rather than deleted.
    """

    type: str                   # e.g. "disjointWith"
    classes: list[str] = field(default_factory=list)
    source: str = ""            # why it was emitted
    flagged: bool = False
    relaxed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
