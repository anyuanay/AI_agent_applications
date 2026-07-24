"""Vocabulary element data model.

Stage 2 turns the noisy *bag of candidate mentions* from Stage 1 into a clean,
flat **vocabulary of named elements**. A :class:`VocabularyElement` is one named
thing (a concept or a relation) that many surface candidates collapsed into.

This is the *term to concept* transition: where a Stage 1 candidate was a single
deduplicated surface string, a Stage 2 element is a *meaning* that may gather
several candidates under one canonical label (``IC`` and ``incident commander``
become one ``IncidentCommander``).

An element knows:

  - its canonical ``label`` (``IncidentCommander`` / ``isDivisibleBy``),
  - its KIND: ``concept`` or ``relation`` (carried straight through from Stage 1),
  - every ``alt_label`` (all surface lexicalizations that merged in), so the link
    back to the text is never lost,
  - the candidate ``members`` (their Stage 1 keys) it was built from,
  - pooled provenance: extractor ``sources``, ``ner_labels``, and literal
    ``occurrences`` with offsets into the original file,
  - whether any member is grounded by a literal span, and whether the whole
    element is ``llm_only`` (no hallucination-free extractor ever backed it).

There is still **no structure**: no parents, no edges, no domain/range. That is
Stage 3's job. An element is just a named meaning with its pooled provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


KIND_CONCEPT = "concept"
KIND_RELATION = "relation"

# spaCy (hallucination-free) source tags, mirrored from Stage 1. An element is
# ``llm_only`` when none of these ever backed any of its members.
SPACY_SOURCES = frozenset({"ner", "term", "adj", "verb"})
SOURCE_LLM = "llm"

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """A stable, file-safe slug for element ids: ``IncidentCommander`` -> ``incidentcommander``."""
    return _SLUG.sub("-", text.casefold()).strip("-")


@dataclass
class VocabularyElement:
    """One named concept or relation after meaning-level grouping and naming."""

    label: str                     # canonical label (PascalCase concept / camelCase relation)
    kind: str                      # KIND_CONCEPT or KIND_RELATION
    representative: str            # the candidate surface the label was coined from
    alt_labels: list[str] = field(default_factory=list)   # every lexicalization that merged in
    members: list[str] = field(default_factory=list)      # Stage 1 candidate keys merged here
    sources: list[str] = field(default_factory=list)      # pooled extractor tags
    ner_labels: list[str] = field(default_factory=list)   # pooled NER labels
    occurrences: list[dict] = field(default_factory=list)  # pooled literal spans (capped)
    mention_count: int = 0         # total literal mentions across all members
    literal_span: bool = False     # at least one member is copied verbatim from source
    llm_only: bool = False         # no hallucination-free extractor backed any member

    @property
    def id(self) -> str:
        """Stable identifier, e.g. ``concept:incidentcommander``."""
        return f"{self.kind}:{slugify(self.label)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d
