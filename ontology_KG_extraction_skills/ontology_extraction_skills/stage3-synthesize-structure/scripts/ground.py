"""Ground the LLM's synthesized ontology back in the candidate vocabulary.

Synthesis is only trustworthy if every element traces to evidence. This module
maps each label the LLM produced back to a Stage 2 candidate, so the synthesized
class inherits real provenance (mention count, alternate labels, grounding). A
label that matches no candidate and was not flagged coined is an LLM
*introduction*: it is kept (recall-first) but flagged ``llm_introduced`` so Stage 4
must corpus-check it before admitting it.

Matching is on a normalized key (casefold, alphanumerics only) over each
candidate's label, representative, and alternate labels, so ``Whole Number`` and
``WholeNumber`` line up.
"""

from __future__ import annotations

import re

_KEY = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _KEY.sub("", (s or "").casefold())


def build_index(concepts: list[dict]) -> dict[str, dict]:
    """Normalized surface -> candidate concept (first occurrence wins)."""
    index: dict[str, dict] = {}
    for c in concepts:
        surfaces = set(c.get("alt_labels") or [])
        surfaces.add(c.get("representative", ""))
        surfaces.add(c.get("label", ""))
        for s in surfaces:
            k = norm(s)
            if k and k not in index:
                index[k] = c
    return index


def resolve(label: str, index: dict[str, dict]) -> dict | None:
    """Return the candidate a label grounds to, or None."""
    return index.get(norm(label))
