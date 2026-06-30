"""Name a group of candidates: pick a representative, coin a canonical label.

The last of Stage 2's three steps. A group of synonymous candidates becomes one
:class:`VocabularyElement` with:

  - a single **canonical label** in ontology casing: ``PascalCase`` for a concept
    (``IncidentCommander``, ``LeastCommonMultiple``), ``camelCase`` for a relation
    (``isDivisibleBy``, ``dispatchedTo``);
  - every surface string in the group kept as an **alt_label**, so the link back
    to the text is never lost;
  - **pooled provenance**: the union of extractor sources and NER labels, and the
    merged literal occurrences (capped for file size, with the true total kept in
    ``mention_count``).

Choosing the representative is what makes an abbreviation defer to its expansion:
among the group's surfaces we prefer a grounded, spelled-out, frequently-seen,
longer form, so ``incident commander`` (not ``IC``) names the concept.
"""

from __future__ import annotations

import re

from element import (
    VocabularyElement, KIND_RELATION, SPACY_SOURCES, SOURCE_LLM,
)

_TOK = re.compile(r"[A-Za-z0-9]+")
# Articles/possessives we never want starting a label (Stage 1 mostly trims
# these, but LLM surfaces can still carry them: "the situation").
_LEADING_STOP = {"a", "an", "the", "this", "that", "these", "those",
                 "its", "their", "his", "her", "our", "your", "some", "any"}

_OCC_CAP = 50  # pooled occurrences kept per element; mention_count stays exact


def _looks_acronym(surface: str) -> bool:
    """A short all-caps token block such as ``IC``, ``LCM``, ``PPE``."""
    letters = surface.replace(".", "").strip()
    return bool(letters) and letters.isupper() and len(letters) <= 6


def _is_grounded(cand: dict) -> bool:
    return bool(cand.get("literal_span"))


def pick_representative(candidates: list[dict]) -> dict:
    """Pick the candidate whose surface should base the label.

    Preference order, highest first:
      1. grounded by a literal span (not an implicit LLM proposal),
      2. not an acronym (spelled-out forms make better labels),
      3. more mentions (the form the corpus actually favors),
      4. longer surface (more specific), then alphabetical for determinism.
    """
    def score(c: dict):
        surface = c.get("canonical") or c.get("key") or ""
        return (
            1 if _is_grounded(c) else 0,
            0 if _looks_acronym(surface) else 1,
            int(c.get("mention_count") or 0),
            len(surface),
            surface,
        )

    return max(candidates, key=score)


def _tokens(surface: str) -> list[str]:
    toks = _TOK.findall(surface)
    while toks and toks[0].casefold() in _LEADING_STOP:
        toks.pop(0)
    return toks


def _cap(tok: str) -> str:
    """Capitalize one token, preserving short all-caps acronyms (``LCM``)."""
    if tok.isupper() and tok.isalpha() and len(tok) <= 6:
        return tok
    return tok[:1].upper() + tok[1:].lower()


def canonical_label(surface: str, kind: str) -> str:
    """PascalCase for concepts, camelCase for relations.

    ``"least common multiple"`` -> ``LeastCommonMultiple``;
    ``"is divisible by"`` -> ``isDivisibleBy``; ``"dispatched to"`` ->
    ``dispatchedTo``. Falls back to a slug of the surface if tokenization empties
    it (pure symbols).
    """
    toks = _tokens(surface)
    if not toks:
        toks = _TOK.findall(surface)  # keep leading stop-word if it is all we have
    if not toks:
        return surface.strip() or "Unnamed"
    if kind == KIND_RELATION:
        first = toks[0] if (toks[0].isupper() and len(toks[0]) <= 6) else toks[0].lower()
        return first + "".join(_cap(t) for t in toks[1:])
    return "".join(_cap(t) for t in toks)


def build_element(candidates: list[dict], kind: str) -> VocabularyElement:
    """Merge a group of Stage 1 candidates into one named vocabulary element."""
    rep = pick_representative(candidates)
    rep_surface = rep.get("canonical") or rep.get("key") or ""
    label = canonical_label(rep_surface, kind)

    alt: set[str] = set()
    members: list[str] = []
    sources: set[str] = set()
    ner_labels: set[str] = set()
    occ_by_span: dict[tuple[int, int], dict] = {}
    mention_count = 0

    for c in candidates:
        if c.get("key"):
            members.append(c["key"])
        for v in (c.get("variants") or []):
            alt.add(v)
        if c.get("canonical"):
            alt.add(c["canonical"])
        sources.update(c.get("sources") or [])
        ner_labels.update(c.get("ner_labels") or [])
        mention_count += int(c.get("mention_count") or 0)
        for o in (c.get("occurrences") or []):
            occ_by_span[(int(o["start"]), int(o["end"]))] = {
                "start": int(o["start"]), "end": int(o["end"]),
                "text": o.get("text", ""),
            }

    occurrences = sorted(occ_by_span.values(), key=lambda o: o["start"])[:_OCC_CAP]
    literal_span = any(_is_grounded(c) for c in candidates)
    llm_only = SOURCE_LLM in sources and not (sources & SPACY_SOURCES)

    return VocabularyElement(
        label=label,
        kind=kind,
        representative=rep_surface,
        alt_labels=sorted(alt),
        members=sorted(set(members)),
        sources=sorted(sources),
        ner_labels=sorted(ner_labels),
        occurrences=occurrences,
        mention_count=mention_count,
        literal_span=literal_span,
        llm_only=llm_only,
    )
