# Refinement gate checklist

Run before handing a reviewed ontology to `serialize-owl`. Every box must be
checked or the reason recorded.

## Coverage and counts

- [ ] Accepted and rejected counts stated at the top (proposed, accepted, rejected).
- [ ] Every rejection carries a reason (not grounded, low confidence, inconsistent).
- [ ] Rejected classes routed to a human, not silently dropped or re-added.

## The Test gate

- [ ] Hallucination guard ran: every accepted class is grounded in the source corpus.
- [ ] The planted `UnicornEvacuationProtocol` (or its real-data equivalent) was caught.
- [ ] No dangling parents: every `parent` is an accepted class or a root.
- [ ] No subclass cycles.
- [ ] Reasoner consistency pass clean (or noted as deferred until disjointness exists).

## Provenance

- [ ] Every accepted class has a real `source` document, no `(none)`.
- [ ] `confidence` carried through for the `scima:extractionConfidence` annotation.

## Handoff

- [ ] Target ontology version decided (for example v0.7 over v0.5).
- [ ] Delta is additive: no existing class or axiom silently changed.
- [ ] Accepted set frozen and passed to `serialize-owl`.
