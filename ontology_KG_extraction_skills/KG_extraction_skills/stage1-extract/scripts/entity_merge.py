"""Entity mention merge: union spaCy and LLM RawMentions into EntityMentions.

Merge key: surface_form.strip().lower()

Resolution rules:
  - surface_form stored: first-seen case-preserved form for this key
  - sources: union of all source tags for this key
  - sentences: union of all sentence indices
  - mention_type: LLM classification wins when sources disagree.
    Specifically, if the LLM tagged a mention, its mention_type is used.
    The only exception: if LLM said "entity" and spaCy said "value" due
    to NER label (CARDINAL/QUANTITY/DATE), the LLM wins — spaCy commonly
    mislabels proper names and compound identifiers.
    If LLM never surfaced the mention, the spaCy tag stands.
"""
from __future__ import annotations

from models import EntityMention, RawMention


def merge(raw_mentions: list[RawMention]) -> list[EntityMention]:
    """Merge RawMentions into deduplicated EntityMentions.

    Args:
        raw_mentions: combined list from spaCy (NER + noun_chunk) and LLM passes.

    Returns:
        One EntityMention per unique normalized surface form.
    """
    # Groups keyed by normalized surface form
    # Value: {surface_form (first seen), sources, sentences, spacy_type, llm_type}
    groups: dict[str, dict] = {}

    for rm in raw_mentions:
        key = rm.surface_form.strip().lower()
        if not key:
            continue

        if key not in groups:
            groups[key] = {
                "surface_form": rm.surface_form,
                "sources": [],
                "sentences": [],
                "spacy_type": None,   # mention_type from ner / noun_chunk
                "llm_type":   None,   # mention_type from llm
            }

        g = groups[key]
        g["sources"].append(rm.source)
        g["sentences"].append(rm.sentence_index)

        if rm.source in ("ner", "noun_chunk"):
            if g["spacy_type"] is None:
                g["spacy_type"] = rm.mention_type
        elif rm.source == "llm":
            if g["llm_type"] is None:
                g["llm_type"] = rm.mention_type

    entity_mentions: list[EntityMention] = []
    for g in groups.values():
        # Resolve mention_type: LLM wins, falls back to spaCy
        if g["llm_type"] is not None:
            resolved_type = g["llm_type"]
        elif g["spacy_type"] is not None:
            resolved_type = g["spacy_type"]
        else:
            resolved_type = "entity"  # default

        entity_mentions.append(EntityMention(
            surface_form=g["surface_form"],
            mention_type=resolved_type,
            sources=g["sources"],
            sentences=g["sentences"],
        ))

    return entity_mentions
