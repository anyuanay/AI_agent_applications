"""Salience scoring for vocabulary elements.

After gating removes particulars, the vocabulary is on-topic but still long: a
domain's real ontology is its *salient* concepts, not every common noun an author
happened to use once. This module scores how central each concept is to the
domain, so the long tail can be parked and the core kept.

Four signals, each grounded in the artifacts already produced:

  - **scope match** (strongest): the concept is one the author themselves marked
    important, i.e. it appears in the Stage 0 scope (defined terms, key concepts,
    or topics). A defined term is salient by construction.
  - **frequency**: how often it is mentioned (a recall/centrality signal).
  - **spread**: across how much of the document it appears (a concept used
    throughout is more central than one confined to a single passage).
  - **grounding**: backed by a hallucination-free extractor (not LLM-only).

The score combines them; the orchestrator decides keep-vs-park from the score plus
simple rules (always keep a scope-matched or frequent concept).
"""

from __future__ import annotations

import math
import re

_WS = re.compile(r"\s+")
_PLURAL = re.compile(r"(?:es|s)$")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().casefold())


def _singular(s: str) -> str:
    return _PLURAL.sub("", s)


def build_scope_terms(scope: dict) -> set[str]:
    """Normalized set of author-marked CONCEPT terms (+ singular forms).

    Uses ``key_terms`` (defined terms and the LLM's general key concepts), which
    are noun concepts. Deliberately excludes ``topics``: section titles and
    learning objectives are procedural ("Divide Integers", "Add and Subtract
    Fractions"), so matching them would keep tasks/skills as if they were classes.
    """
    terms: set[str] = set()
    for t in scope.get("key_terms", []) or []:
        n = _norm(t)
        if n:
            terms.add(n)
            terms.add(_singular(n))
    return terms


def scope_match(concept: dict, scope_terms: set[str]) -> bool:
    """True if any of the concept's surfaces matches an author-marked term."""
    surfaces = set(concept.get("alt_labels") or [])
    surfaces.add(concept.get("representative", ""))
    surfaces.add(concept.get("label", ""))
    for s in surfaces:
        n = _norm(s)
        if n and (n in scope_terms or _singular(n) in scope_terms):
            return True
    return False


def spread(concept: dict, doc_len: int, buckets: int = 20) -> float:
    """Fraction of document segments the concept's occurrences touch, in [0, 1]."""
    occ = concept.get("occurrences") or []
    if not occ or doc_len <= 0:
        return 0.0
    seg = max(1, doc_len // buckets)
    hit = {min(buckets - 1, int(o["start"]) // seg) for o in occ}
    # Normalize by how many buckets this many mentions could possibly cover.
    return len(hit) / min(buckets, max(1, len(occ)))


def grounded(concept: dict) -> bool:
    return not concept.get("llm_only", False)


def score(concept: dict, max_log_mc: float, scope_terms: set[str],
          doc_len: int) -> tuple[float, dict]:
    """Weighted salience in [0, 1], plus its components for transparency."""
    sm = 1.0 if scope_match(concept, scope_terms) else 0.0
    freq = math.log1p(concept.get("mention_count", 0)) / max(max_log_mc, 1e-9)
    spr = spread(concept, doc_len)
    grd = 1.0 if grounded(concept) else 0.0
    s = 0.35 * sm + 0.30 * freq + 0.20 * spr + 0.15 * grd
    return s, {"scope_match": sm, "freq": round(freq, 3),
               "spread": round(spr, 3), "grounded": grd}
