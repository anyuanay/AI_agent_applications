"""Dedup: turn the union of both extractors into a set.

The final step, intentionally shallow. Candidates are grouped by
``(kind, normalized-key)``; within a group everything is merged:

  - surface variants are unioned (``"Whole Numbers"`` / ``"whole numbers"``),
  - literal occurrences are pooled and de-duplicated by offset,
  - extractor sources are unioned (so a term found by both spaCy *and* the LLM
    is no longer ``llm_only``),
  - NER labels are unioned.

String-level only. Two different strings that *mean* the same thing
(``"LCM"`` vs ``"least common multiple"``) survive as separate candidates here;
meaning-level merging is a harder operation left to a later step.
"""

from __future__ import annotations

from collections import Counter

from candidate import Candidate, Occurrence, RawCandidate


def _merge_group(key: str, kind: str, raws: list[RawCandidate]) -> Candidate:
    sources: set[str] = set()
    variants: Counter = Counter()
    ner_labels: set[str] = set()
    occ_by_span: dict[tuple[int, int], Occurrence] = {}

    for r in raws:
        sources.add(r.source)
        if r.text:
            variants[r.text] += 1
        if r.ner_label:
            ner_labels.add(r.ner_label)
        if r.start is not None and r.end is not None:
            occ_by_span[(r.start, r.end)] = Occurrence(
                start=r.start, end=r.end, text=r.text
            )

    # Canonical surface = the most frequently seen variant; ties broken by the
    # longer (more specific) string, then alphabetically for determinism.
    if variants:
        canonical = sorted(
            variants.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0])
        )[0][0]
    else:
        canonical = key

    return Candidate(
        canonical=canonical,
        key=key,
        kind=kind,
        sources=sorted(sources),
        variants=sorted(variants),
        occurrences=sorted(occ_by_span.values(), key=lambda o: o.start),
        ner_labels=sorted(ner_labels),
    )


def dedup(raws: list[RawCandidate]) -> list[Candidate]:
    """Collapse raw mentions into a deduplicated candidate set.

    Candidates with an empty key (failed normalization) are dropped: a term we
    cannot key is a term we cannot track, and tracking is the whole point.
    """
    groups: dict[tuple[str, str], list[RawCandidate]] = {}
    for r in raws:
        if not r.key:
            continue
        groups.setdefault((r.kind, r.key), []).append(r)

    merged = [_merge_group(key, kind, raws) for (kind, key), raws in groups.items()]
    # Order output for readability: concepts before relations, strongest first.
    merged.sort(key=lambda c: (c.kind, -c.mention_count, c.canonical.casefold()))
    return merged
