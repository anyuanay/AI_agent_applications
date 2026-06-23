---
name: refine-ontology
description: Review proposed ontology classes through the RITE loop (Refine,
  Inspect, Test, Extend), dropping classes the source corpus never justified and
  checking the survivors for logical consistency. Stage 4 of ontology extraction.
  Use to turn noisy proposals into a vetted hierarchy before serialization.
tools: [rite_review, consistency_check]
---
Procedure:
  1. Take the `ProposedClass` proposals from `induce-schema` (and the coherence
     flags from `cluster-classes`).
  2. rite_review() against the source corpus with the two gates in
     `references/validation_rules.md`:
       - Test: every meaningful word of the label must appear in the sources
         (the hallucination guard).
       - Inspect: the pipeline confidence must clear the floor.
  3. consistency_check() the accepted set with an OWL reasoner: no unsatisfiable
     classes, no domain/range or disjointness violations.
  4. Report accepted vs rejected. Rejected classes go to a human for follow-up,
     not silently back into the ontology.
  5. Use `checklist.md` as the gate before handing off to `serialize-owl`.

Guidance for this stage:
- This is where precision is recovered after the high-recall proposal stages. Be
  strict: a fluent, plausible class that the documents never mention is exactly
  what the Test gate exists to catch (for example `UnicornEvacuationProtocol`).
- The four phases: Refine the labels and parents, Inspect against confidence and
  the corpus, Test for grounding and consistency, Extend by re-prompting for what
  review revealed is missing. See `references/rite_loop.md`.
- Keep humans in the loop. The automated gates stand in for expert judgement; they
  do not replace it. Surface every rejection with its reason.

Run:
    python scripts/rite_review.py               # accept/reject proposals, show the guard
    python scripts/consistency_check.py         # reasoner pass over the accepted set
