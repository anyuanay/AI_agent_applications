---
name: stage4-review-ontology
description: The trust gate of ontology extraction. Reviews the Stage 3 structure by testing every element against its kind (concept by corpus grounding, coined parent by >=2 grounded children, relationship by predicate + sane domain/range, axiom by the reasoner), then auto-accepts the clear cases, auto-rejects unsupported hallucinations, and escalates the ambiguous for a human. Applies the RITE Refine edits (merge duplicate labels, demote under-justified coined parents, park orphans, and optionally name coined families with gemini-3.1-flash-lite), re-runs a lightweight reasoner, and commits the survivors as an admitted ontology plus a feedback set. Use as the fourth and final step of ontology extraction, after inducing structure, or when asked to "review the ontology", "run RITE review", "accept/reject/escalate extracted classes", or "admit the ontology". Reads a <stem>_structure.json plus its source text and writes <stem>_structure_reviewed.json (and optionally Turtle) next to it.
---

# Review the Ontology (Agentic RITE)

This skill performs the **review** step of ontology extraction: the trust gate.
Everything upstream optimized for recall; this stage buys back precision. A review
agent reads each element's dossier, tests it **by its kind** using real tools, and
routes it to one of three outcomes, with a human ratifying the escalations.

The single job of this step: **decide what to admit.** Accept the grounded and
consistent, reject the unsupported, escalate the ambiguous, and apply cheap fixes
first, all without ever silently dropping anything.

## When to use this skill

- You have run Stage 3 (synthesize structure) and have a `<stem>_structure.json`, plus
  the original source text.
- You want an admitted ontology (the elements that pass review) plus an auditable
  record of every decision and a feedback set for the next pass.
- The user asks to "review the ontology", "run RITE review", "accept/reject/
  escalate", or "admit the ontology".

## The autonomy boundary

The agent acts alone on the clear cases and escalates only the genuinely
ambiguous, so a human is never surprised by what it admitted:

- **Auto-accept** a high-confidence element grounded by its kind and consistent.
- **Auto-reject** a clear hallucination: an extracted concept with no corpus
  support at all. Rejections go to a feedback set, not deletion.
- **Escalate** the genuinely ambiguous (multiple inheritance, borderline
  grounding, a relationship with no resolvable domain/range) to a human.

## The four tools (it tests, it does not guess)

`scripts/tools.py` implements the deterministic tools the agent calls:

- **grounding-check** by kind (a concept by a recorded mention; a coined parent by
  its grounded children; a relationship by its predicate),
- **corpus search** to re-ground a zero-mention concept by finding a surface in
  the text,
- **reasoner**, the lightweight consistency check (cycles + disjointness), re-run
  after every edit and repairing by relaxing the most suspect axiom first.

The fourth tool, the **ontology editor**, is the set of Refine edits in the
orchestrator.

## RITE

- **Refine** (`scripts/review_ontology.py`). Cheap fixes first: **merge** concepts
  that share an exact label (the near-duplicates Stage 3's id-disambiguation left
  behind), **demote** a coined parent with fewer than two grounded children to a
  plain class, **park** orphans, and (with `--llm`) **name** the coined families
  Stage 3 left flagged `needs_naming`. Every edit is recorded, never silent.
- **Inspect.** Each element carries a dossier (provenance, confidence, coined-or-
  extracted, flags). Coined, low-confidence, and flagged items get the hard look.
- **Test, by kind** (`scripts/verdicts.py`):

  | Kind | Test it must pass |
  |---|---|
  | Extracted concept | A recorded mention, or a surface found by corpus search. |
  | Coined parent | At least two grounded children (else demote; zero, reject). |
  | Relationship | Predicate grounding plus a domain and range that survived. |
  | Axiom | The reasoner finds no contradiction (else relaxed, not admitted). |

- **Extend.** Survivors are committed. Orphans are parked under a domain top and
  flagged for re-parenting on a later pass, never discarded. Concepts surfaced but
  tied to nothing (no parent, no children, in no admitted relationship) are kept
  in the feedback set for the next pass, the same recall-first instinct.

## The deferred naming lands here

Stage 2 named extracted concepts deterministically and Stage 3 left coined parents
with placeholder names (flagged `needs_naming`). This is the stage where, under
review, an LLM (`scripts/llm_refine.py`, `gemini-3.1-flash-lite`) proposes a real
name for each accepted coined family from its children. It is a Refine edit:
human-approvable and recorded. Without `--llm`, coined parents keep their
placeholders and stay flagged for a human.

## How to run

```bash
cd scripts
python review_ontology.py <STRUCTURE_JSON> [options]
```

The result is written next to the structure as `<STRUCTURE_JSON_stem>_reviewed.json`.

| Option | Default | Meaning |
|---|---|---|
| `--source PATH` | structure's `input_file` | Original text, for corpus grounding. |
| `--llm` | off | Use `gemini-3.1-flash-lite` to name accepted coined families. |
| `--llm-model` | `gemini-3.1-flash-lite` | Gemini model id. |
| `--emit-ttl` | off | Also write the admitted ontology as Turtle (`<stem>_admitted.ttl`). |
| `--env PATH` | auto | Explicit `.env` holding `GOOGLE_API_KEY` (only with `--llm`). |
| `--output PATH` | auto | Override the output path. |

Examples:

```bash
# Deterministic review (no API calls)
python review_ontology.py INPUT_structure.json

# Full review with LLM-named coined families and a Turtle export
python review_ontology.py INPUT_structure.json --llm --emit-ttl
```

## Output schema

`<stem>_structure_reviewed.json`:

```jsonc
{
  "stage": "4-review-ontology",
  "structure_file": "..._structure.json", "vocabulary_file": "...", "input_file": "....md",
  "generated_at": "ISO-8601 UTC",
  "config": { "llm": false, "llm_model": null },
  "stats": { "admitted_classes": 0, "admitted_coined": 0, "admitted_relationships": 0,
             "admitted_axioms": 0, "merged": 0, "demoted": 0, "parked_orphans": 0,
             "unconnected": 0, "verdicts": { "accept": 0, "reject": 0, "escalate": 0 } },
  "autonomy": { "auto_accepted": 0, "auto_rejected": 0, "escalated": 0 },
  "consistency": { "reasoner": "lightweight (cycles + disjointness)", "consistent": true,
                   "cycles_remaining": 0, "disjointness_relaxed": 0 },
  "admitted": {
    "classes": [ { "id": "concept:wholenumber", "label": "WholeNumber",
                   "alt_labels": ["..."], "parents": ["concept:number"],
                   "coined": false, "parked": false } ],
    "coined_parents": [ { "id": "coined:responderunit", "label": "ResponderUnit",
                          "children": ["concept:hazmatteam", "..."] } ],
    "relationships": [ { "id": "relation:dispatchedto", "label": "dispatchedTo",
                         "domain": "coined:responderunit", "range": "concept:incident" } ],
    "axioms": [ { "type": "disjointWith", "classes": ["...", "..."] } ]
  },
  "edits": [ { "type": "merge", "targets": ["...", "..."], "detail": { "kept": "..." } },
             { "type": "demote", "targets": ["coined:..."], "detail": { "grounded_children": 1 } },
             { "type": "name_coined", "targets": ["coined:..."], "detail": { "old": "...", "new": "..." } } ],
  "feedback": {
    "rejected": [ { "id": "...", "kind": "...", "verdict": "reject", "reasons": ["..."] } ],
    "escalated": [ { "id": "...", "kind": "...", "verdict": "escalate", "reasons": ["..."] } ],
    "parked_orphans": ["concept:..."],
    "unconnected": ["concept:..."]
  },
  "decisions": [ { "id": "...", "kind": "...", "verdict": "accept", "grounded": true,
                   "reasons": ["..."], "parked": false } ]
}
```

## Files

```
stage4-review-ontology/
├── SKILL.md                 # this file
├── requirements.txt
├── scripts/
│   ├── review_ontology.py   # orchestrator + CLI (the RITE loop; run this)
│   ├── verdicts.py          # test-by-kind -> accept / reject / escalate
│   ├── tools.py             # grounding-check, corpus search, lightweight reasoner
│   ├── llm_refine.py        # optional gemini-3.1-flash-lite naming of coined families
│   ├── emit_ttl.py          # optional Turtle serialization of the admitted ontology
│   └── model.py             # Decision / Edit dataclasses
└── references/
    └── method.md            # the review method in detail
```

## Requirements

- Python 3.10+
- `rdflib` only when `--emit-ttl` is used
- `google-genai` + `GOOGLE_API_KEY` only when `--llm` is used
- Otherwise pure standard library: the deterministic review needs no API keys.

## Output and what comes next

The artifact is the admitted ontology (the elements that passed review), an
auditable decision log, the Refine edits, and a feedback set. This closes one
pass of the four-stage pipeline. The pipeline is a loop: the feedback set
(rejected, parked orphans, unconnected concepts) and the admitted classes seed the
next pass over more documents, so each pass starts from a richer schema. Escalated
items await human ratification before they join the admitted set.

## Notes and limitations

- **The agent is a bounded policy, not an open-ended LLM loop.** It auto-accepts
  the clear majority (as the article intends: a class with many mentions does not
  need a hard look), auto-rejects the unsupported, and escalates the ambiguous.
  The LLM is used only for the generative Refine step (naming coined families).
- **Recall-first to the end.** Nothing is deleted. Rejected, parked, and
  unconnected elements all live in the feedback set for the next pass.
- **Grounding is corpus-based.** A concept with zero recorded mentions is rescued
  only if a surface is found verbatim in the source; otherwise it is the one thing
  the agent rejects on its own (a hallucination).
- **The reasoner is lightweight.** It checks cycles and disjointness contradictions
  and relaxes the most suspect axiom first. For full OWL DL entailment, take the
  `--emit-ttl` output to an external reasoner.
- **Merging is by exact label.** It folds back the duplicate-label concepts Stage
  3 disambiguated; distinct senses that happen to share a label are merged too, so
  the merge edits are recorded for a human to split if needed.
```
