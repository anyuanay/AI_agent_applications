"""Triple merge: normalize and union dep-parse and LLM triples into candidates.

Normalization:
  - subjects and objects: lowercase + whitespace-collapse (no lemmatization —
    lemmatization distorts named entities and multi-word identifiers)
  - predicates: lowercase + spaCy lemmatization (verbs/verb phrases are
    well-defined for lemmatization)

Merge key: (normalized_subject, normalized_predicate, normalized_object)

Negation rule: a merged triple is negated if EITHER source marks it negated.
Negated triples are excluded from _candidate_triples.json and written to
_negated_triples.json.

object_type is inherited from entity_mentions by normalized object lookup.
The field is omitted (None) when the object is not in entity_mentions.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from models import CandidateTriple, DepTriple, EntityMention, LLMTriple

_SOURCE_DEP = "dep_parse"
_SOURCE_LLM = "llm"


def _normalize_entity(text: str) -> str:
    """Normalize subject/object: lowercase, collapse whitespace."""
    return " ".join(text.strip().lower().split())


def _normalize_predicate(nlp: Any, text: str) -> str:
    """Normalize predicate: lowercase + spaCy lemmatize each token."""
    lowered = text.strip().lower()
    doc = nlp(lowered)
    return " ".join(tok.lemma_ for tok in doc)


def _build_mention_index(entity_mentions: list[EntityMention]) -> dict[str, str]:
    """Map normalized surface form -> mention_type."""
    index: dict[str, str] = {}
    for em in entity_mentions:
        key = em.surface_form.strip().lower()
        index[key] = em.mention_type
    return index


def merge(
    dep_triples: list[DepTriple],
    llm_triples: list[LLMTriple],
    entity_mentions: list[EntityMention],
    nlp: Any,
) -> tuple[list[CandidateTriple], list[CandidateTriple]]:
    """Merge dep-parse and LLM triples into unified candidate sets.

    Returns:
        candidate_triples — non-negated triples ready for Stage 2
        negated_triples   — triples that carry negated=true from any source
    """
    mention_index = _build_mention_index(entity_mentions)

    # Accumulator keyed by normalized (s, p, o)
    # Value: {subj, pred, obj, sources, sentence_indices, negated}
    groups: dict[tuple[str, str, str], dict] = {}

    def _add(subj: str, pred_raw: str, obj: str,
             source_tag: str, sent_idx: int, negated: bool) -> None:
        norm_s = _normalize_entity(subj)
        norm_p = _normalize_predicate(nlp, pred_raw)
        norm_o = _normalize_entity(obj)
        if not norm_s or not norm_p or not norm_o:
            return
        key = (norm_s, norm_p, norm_o)

        if key not in groups:
            groups[key] = {
                "subject":  subj,
                "predicate": pred_raw,
                "object":   obj,
                "sources":  [],
                "sentence_indices": [],
                "negated":  False,
            }

        g = groups[key]
        g["sources"].append(source_tag)
        g["sentence_indices"].append(sent_idx)
        # Negation is sticky: once set, stays set
        if negated:
            g["negated"] = True

    for dt in dep_triples:
        _add(dt.subject, dt.predicate, dt.object,
             _SOURCE_DEP, dt.sentence_index, dt.negated)

    for lt in llm_triples:
        _add(lt.subject, lt.predicate, lt.object,
             _SOURCE_LLM, lt.sentence_index, lt.negated)

    candidate_triples: list[CandidateTriple] = []
    negated_triples:   list[CandidateTriple] = []

    for g in groups.values():
        obj_key  = _normalize_entity(g["object"])
        obj_type: Optional[str] = mention_index.get(obj_key)

        all_indices = sorted(set(g["sentence_indices"]))
        ct = CandidateTriple(
            subject=g["subject"],
            predicate=g["predicate"],
            object=g["object"],
            object_type=obj_type,
            sentence_index=all_indices[0],
            sentence_indices=all_indices,
            sources=list(set(g["sources"])),
        )
        if g["negated"]:
            negated_triples.append(ct)
        else:
            candidate_triples.append(ct)

    return candidate_triples, negated_triples
