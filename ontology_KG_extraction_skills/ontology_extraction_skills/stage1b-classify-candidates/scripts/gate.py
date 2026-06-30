"""Deterministic type/instance/non-concept gate.

An ontology holds **universals** (classes like ``Fraction``, ``Equation``), not
**particulars** (a specific value ``$\\frac{1}{5}$``, a worked number ``144``, a
person ``Marissa``) and not document debris (LaTeX fragments, section titles). The
bottom-up surfacing stage cannot tell these apart, so this gate sorts every
candidate into one of three bins before naming:

  - **class**       -> goes on to Stage 2 (the ontology vocabulary)
  - **individual**  -> a particular; routed to an A-Box instances file (kept for a
                       later knowledge-graph pass, not put in the ontology)
  - **non_concept** -> markup/boilerplate debris; routed to feedback, not deleted

The deterministic signal here is **morphology**, not spaCy NER labels: on this
corpus NER is far too noisy to trust (it tags ``equation`` as MONEY and
``fraction`` as PERSON), so using it to route would throw away core concepts.
Morphology (digits, LaTeX/markup symbols, document-structure words) is reliable.
Named individuals that have no digits (``Marissa``, ``Tuesday``) are left for the
optional LLM tiebreak, which is far more reliable than NER for the type/instance
call.
"""

from __future__ import annotations

import re

CLASS = "class"
INDIVIDUAL = "individual"
NON_CONCEPT = "non_concept"

# LaTeX / markup / HTML debris: a backslash, math delimiters, or entity chars.
_DEBRIS = re.compile(r"[\\${}^_|<>&]")
_DIGIT = re.compile(r"\d")
_ALPHA = re.compile(r"[A-Za-z]")

# Document-structure words: a candidate built around one of these (with a number)
# is a section/figure label, not a domain concept.
_DOCWORDS = {"chapter", "page", "section", "figure", "fig", "table", "example",
             "exercise", "exercises", "step", "solution", "appendix", "part",
             "problem", "unit"}   # 'unit' only when numbered (e.g. "unit 3")

_TOK = re.compile(r"[A-Za-z]+")


def _tokens(s: str) -> list[str]:
    return [t.casefold() for t in _TOK.findall(s)]


def classify_concept(cand: dict) -> tuple[str, str]:
    """Return (bin, reason) for a concept candidate."""
    s = cand.get("canonical") or cand.get("key") or ""
    if not _ALPHA.search(s):
        return NON_CONCEPT, "no letters (pure symbol/number)"
    if _DEBRIS.search(s):
        return NON_CONCEPT, "LaTeX/markup debris"
    toks = _tokens(s)
    has_digit = bool(_DIGIT.search(s))
    if has_digit and (_DOCWORDS & set(toks)):
        return NON_CONCEPT, "numbered document-structure label"
    if has_digit:
        return INDIVIDUAL, "contains a numeric value (a particular)"
    # No digit, no debris: a common-noun phrase. Treat as a class; named
    # individuals without digits are caught by the optional LLM tiebreak.
    return CLASS, "common-noun phrase"


def classify_relation(cand: dict) -> tuple[str, str]:
    """Relations are object properties, not instances; only debris is dropped."""
    s = cand.get("canonical") or cand.get("key") or ""
    if not _ALPHA.search(s):
        return NON_CONCEPT, "no letters"
    if _DEBRIS.search(s):
        return NON_CONCEPT, "LaTeX/markup debris"
    if _DIGIT.search(s):
        return NON_CONCEPT, "numeric (not a relation)"
    return CLASS, "predicate"


def classify(cand: dict) -> tuple[str, str]:
    if cand.get("kind") == "relation":
        return classify_relation(cand)
    return classify_concept(cand)
