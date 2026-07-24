"""Lexical / morphological normalization for dedup.

Dedup is deliberately shallow: casefold, lemmatize, and collapse string variants
so the *union* of the two extractors becomes a *set*. This is string-level dedup
only; meaning-level merging (e.g. ``LCM`` with ``least common multiple``) is left
to a later step.

A dedup *key* is the lemmatized, casefolded, whitespace-collapsed surface form:

    "Whole Numbers" -> "whole number"
    "fractions"     -> "fraction"
    "divides by"    -> "divide by"
"""

from __future__ import annotations

import re
import unicodedata

# Token-level punctuation we trim from the edges of a term before keying.
# Includes the markdown table pipe so "| Billions" keys cleanly to "billion".
_EDGE_PUNCT = " \t\n\r\"'`.,;:!?()[]{}<>|*#-"
_WS = re.compile(r"\s+")


def clean_surface(text: str) -> str:
    """Trim and collapse whitespace on a surface string, keeping its casing."""
    text = unicodedata.normalize("NFKC", text)
    text = _WS.sub(" ", text).strip()
    return text.strip(_EDGE_PUNCT).strip()


def key_from_tokens(tokens) -> str:
    """Build a dedup key from spaCy tokens, using in-context lemmas.

    ``tokens`` is any iterable of spaCy ``Token`` objects (a Span, a noun chunk,
    or a single token). Lemmas are taken in context, which is why we prefer this
    over re-parsing the surface string when we already have the parse.
    """
    parts = []
    for tok in tokens:
        if tok.is_space or tok.is_punct:
            continue
        lemma = (tok.lemma_ or tok.text).strip().casefold()
        if lemma:
            parts.append(lemma)
    return _WS.sub(" ", " ".join(parts)).strip()


def key_from_text(nlp, text: str) -> str:
    """Build a dedup key from a bare string by parsing it with ``nlp``.

    Used for LLM-proposed terms, which arrive as plain strings with no parse.
    """
    surface = clean_surface(text)
    if not surface:
        return ""
    doc = nlp(surface)
    key = key_from_tokens(doc)
    # Fall back to a casefolded surface if lemmatization emptied the string
    # (can happen for pure symbols / numerals).
    return key or surface.casefold()
