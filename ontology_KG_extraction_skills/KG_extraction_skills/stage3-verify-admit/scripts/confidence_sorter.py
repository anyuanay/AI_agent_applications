"""Sort mapped triples by confidence for greedy Stage 3 processing.

Four-level sort key (ascending = highest confidence first):
  1. Primary:   multi-sentence corroboration (len(sentence_indices) >= 2)
  2. Secondary: source agreement within a sentence
  3. Tertiary:  mapping_confidence from Stage 2
  4. Tiebreaker: sentence_index ascending (earlier in text first)
"""
from __future__ import annotations


def _sort_key(triple: dict) -> tuple:
    # Primary: cross-sentence corroboration
    sent_idxs = triple.get("sentence_indices") or [triple.get("sentence_index", 0)]
    primary = 0 if len(sent_idxs) >= 2 else 1

    # Secondary: source agreement
    sources = set(triple.get("sources", []))
    if "dep_parse" in sources and "llm" in sources:
        secondary = 0
    elif "dep_parse" in sources:
        secondary = 1
    elif "llm" in sources:
        secondary = 2
    else:
        secondary = 3

    # Tertiary: Stage 2 mapping confidence
    tertiary = 0 if triple.get("mapping_confidence") == "high" else 1

    # Tiebreaker: earlier sentence first
    tiebreak = triple.get("sentence_index", 0)

    return (primary, secondary, tertiary, tiebreak)


def sort_by_confidence(triples: list[dict]) -> list[dict]:
    """Return a new list sorted highest confidence first."""
    return sorted(triples, key=_sort_key)
