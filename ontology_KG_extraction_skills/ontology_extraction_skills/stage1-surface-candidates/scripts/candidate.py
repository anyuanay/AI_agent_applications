"""Candidate data model.

Every surfaced candidate is a *mention*: a term that could become an ontology
element, carrying enough provenance to answer the only question this step is
allowed to leave open: *where did this come from?*

A candidate knows only:

  - its surface string(s) and a canonical (lemmatized) key used for dedup,
  - its KIND: ``concept`` or ``relation`` (grounding-by-kind starts here),
  - which extractor(s) surfaced it (``ner``, ``term``, ``adj``, ``verb``, ``llm``),
  - every literal occurrence (char offsets into the *original* input file),
  - whether it is grounded by a literal span at all, and
  - whether it was *only* proposed by the LLM (``llm_only``) so a later review
    can give it a harder look.

These fields encode three design principles: recall-first (nothing is scored
away here), provenance everywhere (offsets + source tags on every record), and
grounding-by-kind (``kind`` is set now).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# Extractor source tags. ``ner`` / ``term`` / ``adj`` / ``verb`` come from the
# hallucination-free spaCy extractor (literal spans). ``llm`` comes from the
# language model and may or may not be a literal span.
SOURCE_NER = "ner"      # spaCy named entity
SOURCE_TERM = "term"    # spaCy noun-phrase term or gerund nominalization
SOURCE_ADJ = "adj"      # spaCy adjective quality (opt-in)
SOURCE_VERB = "verb"    # spaCy syntactic predicate (verb, verb phrase, copular)
SOURCE_LLM = "llm"      # language-model proposal

SPACY_SOURCES = frozenset({SOURCE_NER, SOURCE_TERM, SOURCE_ADJ, SOURCE_VERB})

KIND_CONCEPT = "concept"
KIND_RELATION = "relation"


@dataclass
class Occurrence:
    """One literal appearance of a candidate in the source file.

    ``start``/``end`` are character offsets into the *original* input file, so a
    span is always traceable back to the exact bytes it was copied from. A
    candidate with no occurrences (an LLM term not found verbatim in the text) is
    not grounded by a literal span and is flagged accordingly.
    """

    start: int
    end: int
    text: str  # the exact substring of the source at [start, end)


@dataclass
class Candidate:
    """A single mention after dedup.

    One canonical string, every surface variant that collapsed into it, every
    literal occurrence, and the union of the extractors that surfaced it.
    """

    canonical: str                 # human-facing surface form (most frequent variant)
    key: str                       # normalized dedup key (lemmatized, casefolded)
    kind: str                      # KIND_CONCEPT or KIND_RELATION
    sources: list[str] = field(default_factory=list)   # subset of SOURCE_*
    variants: list[str] = field(default_factory=list)  # all surface strings seen
    occurrences: list[Occurrence] = field(default_factory=list)
    ner_labels: list[str] = field(default_factory=list)  # e.g. ["ORG", "PERSON"]

    @property
    def mention_count(self) -> int:
        """Number of literal occurrences in the source (recall/strength signal)."""
        return len(self.occurrences)

    @property
    def literal_span(self) -> bool:
        """True iff this candidate was copied verbatim from the source at least once.

        A literal span cannot be a hallucination: the bytes are right there. LLM
        terms that are *not* literal spans are implicit candidates: kept (recall-
        first) but tagged for a harder look in a later review.
        """
        return len(self.occurrences) > 0

    @property
    def llm_only(self) -> bool:
        """True iff no spaCy (hallucination-free) extractor backed this candidate.

        ``llm_only`` tags whatever the LLM adds beyond what the spaCy extractor
        found; a later review uses it to decide what still needs grounding.
        """
        return SOURCE_LLM in self.sources and not (set(self.sources) & SPACY_SOURCES)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mention_count"] = self.mention_count
        d["literal_span"] = self.literal_span
        d["llm_only"] = self.llm_only
        return d


@dataclass
class RawCandidate:
    """A single un-deduplicated mention as emitted by one extractor.

    The orchestrator collects these from every extractor and hands them to the
    deduper, which collapses them into :class:`Candidate` records.
    """

    text: str
    kind: str
    source: str
    key: str                       # normalized form, filled by the extractor
    start: Optional[int] = None    # char offset into source, or None for non-literal
    end: Optional[int] = None
    ner_label: Optional[str] = None
