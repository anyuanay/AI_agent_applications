"""Shared data models for stage1-extract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sentence:
    sentence_index: int
    text: str

    def to_dict(self) -> dict:
        return {"sentence_index": self.sentence_index, "text": self.text}


@dataclass
class RawMention:
    """Intermediate mention record before the entity merge."""
    surface_form: str       # case-preserved, as extracted
    mention_type: str       # "entity" or "value"
    source: str             # "ner" | "noun_chunk" | "llm"
    sentence_index: int


@dataclass
class EntityMention:
    """Post-merge entity/value mention record."""
    surface_form: str       # canonical case-preserved first-seen form
    mention_type: str       # "entity" or "value" (resolved)
    sources: list[str]      # all sources that produced this mention
    sentences: list[int]    # all sentence indices where it appears

    def to_dict(self) -> dict:
        return {
            "surface_form": self.surface_form,
            "mention_type": self.mention_type,
            "sources": sorted(set(self.sources)),
            "sentences": sorted(set(self.sentences)),
        }


@dataclass
class DepTriple:
    """Raw triple from the spaCy dependency parse."""
    subject: str
    predicate: str
    object: str
    sentence_index: int
    negated: bool = False

    def to_dict(self) -> dict:
        d = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "sentence_index": self.sentence_index,
        }
        if self.negated:
            d["negated"] = True
        return d


@dataclass
class LLMTriple:
    """Raw triple returned by the LLM pass."""
    subject: str
    predicate: str
    object: str
    sentence_index: int
    negated: bool = False

    def to_dict(self) -> dict:
        d = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "sentence_index": self.sentence_index,
        }
        if self.negated:
            d["negated"] = True
        return d


@dataclass
class CandidateTriple:
    """Normalized, merged triple ready for Stage 2."""
    subject: str
    predicate: str
    object: str
    object_type: Optional[str]   # "entity" or "value"; None if not in entity_mentions
    sentence_index: int           # lowest sentence index (first occurrence)
    sentence_indices: list[int]
    sources: list[str]            # e.g. ["dep_parse", "llm"]

    def to_dict(self) -> dict:
        d = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "sentence_index": self.sentence_index,
            "sentence_indices": sorted(set(self.sentence_indices)),
            "sources": sorted(set(self.sources)),
        }
        if self.object_type is not None:
            d["object_type"] = self.object_type
        return d
