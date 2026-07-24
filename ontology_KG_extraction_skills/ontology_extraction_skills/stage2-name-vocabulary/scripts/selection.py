"""Select, minimally: drop only the obvious non-concepts.

The middle step of Stage 2, and the one most at risk of violating recall-first.
The rule from the method is explicit: drop only the obvious non-concepts
(sentence fragments, attribute values masquerading as terms, one-off noise that
appears once and nowhere else). Almost everything plausible survives, on purpose.
This is **not** the precision gate; Stage 4 review is. Selecting hard here would
throw away recall the whole pipeline is built to protect.

So this module is deliberately conservative. It flags an element only when it is
near-certainly not an ontology element, and even then nothing is deleted: a
rejected element is returned with its reason and written to a ``dropped`` list in
the output, as feedback, not a silent deletion. Stage 4 can overrule it.
"""

from __future__ import annotations

import re

from element import VocabularyElement, KIND_CONCEPT

# Document scaffolding that surfaces as candidates but names no domain concept.
# Kept short and generic on purpose: these are structural words of *documents*,
# not of the subject matter.
_BOILERPLATE = {
    "page", "step", "figure", "fig", "table", "example", "exercise", "chapter",
    "section", "note", "answer", "solution", "problem", "part", "item", "row",
    "column", "appendix", "outline", "objective", "introduction", "summary",
}

_FRAGMENT_MAX_TOKENS = 7   # a "concept" longer than this is almost surely a fragment
_NUMERIC = re.compile(r"^[\W\d_]+$")   # no letters at all: a value, not a concept
_TOK = re.compile(r"[A-Za-z0-9]+")


def rejection_reason(el: VocabularyElement) -> str | None:
    """Return why ``el`` is not an ontology element, or None to keep it.

    Conservative by design: only near-certain non-concepts are rejected, so the
    overwhelming majority of plausible elements pass through untouched.
    """
    surface = (el.representative or el.label or "").strip()
    toks = _TOK.findall(surface)

    if not surface or not toks:
        return "empty or symbol-only surface"

    # Attribute values masquerading as terms: pure numbers / punctuation.
    if _NUMERIC.match(surface):
        return "numeric or symbolic value, not a concept"

    # Document scaffolding (concepts only; a relation named 'step' is unlikely).
    if el.kind == KIND_CONCEPT and len(toks) == 1 and toks[0].casefold() in _BOILERPLATE:
        return "document boilerplate, not a domain concept"

    # Sentence fragments: an over-long multi-word span the chunker over-captured.
    if el.kind == KIND_CONCEPT and len(toks) > _FRAGMENT_MAX_TOKENS:
        return f"sentence fragment ({len(toks)} tokens)"

    # One-off noise: appears once, in exactly one place, with no extractor
    # agreement, and was never grounded by a literal span. This is the narrow
    # "appears once and nowhere else" case; anything seen twice survives.
    if (el.mention_count <= 1 and not el.literal_span
            and len(el.sources) <= 1 and el.llm_only):
        return "one-off ungrounded LLM proposal (appears nowhere in text)"

    return None


def partition(elements: list[VocabularyElement]):
    """Split elements into (kept, dropped); dropped pairs the element with a reason."""
    kept: list[VocabularyElement] = []
    dropped: list[tuple[VocabularyElement, str]] = []
    for el in elements:
        reason = rejection_reason(el)
        if reason:
            dropped.append((el, reason))
        else:
            kept.append(el)
    return kept, dropped
