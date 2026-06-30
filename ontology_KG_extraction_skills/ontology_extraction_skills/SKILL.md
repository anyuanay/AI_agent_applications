---
name: ontology-extraction
description: End-to-end orchestrator for extracting an ontology from a source document. Runs the full seven-stage pipeline (Stage 0 scope, Stage 1 surface candidates, Stage 1b classify type vs instance, Stage 2 name the vocabulary, Stage 2b salience, Stage 3 synthesize structure, Stage 4 RITE review) by chaining the seven stage skills in order with the correct file handoffs, via scripts/run_pipeline.py. Use this when the user gives a source (a textbook, manual, spec, or any document) and wants to "extract / build / learn an ontology from it", "run the whole ontology pipeline", "go from a source to an ontology", or "get started with the extraction skills end to end". For control over a single step (just scoping, just surfacing, just review), use the individual stageN-* skill instead. Produces an admitted ontology (Turtle) plus an auditable per-decision record next to the source.
---

# Ontology Extraction (end to end)

This is the **entry point** for the ontology-extraction skill set. The work is
done by seven independent stage skills, one per directory (`stage0-scope` through
`stage4-review-ontology`). This skill orchestrates them: it knows the order, the
file each stage hands to the next, and the few cross-stage inputs, so you can go
from a source document to an admitted ontology in one move.

An ontology is a **conceptualization** of a domain (its general kinds and
relationships), not a dump of every noun and number in a text. The pipeline is
framed top-down by a scope, fed by high-recall surfacing, narrowed in deliberate
steps (sort out particulars, name, rank by salience), and only then committed to
structure, under review. Recall is maximized early; pruning is deferred and never
silent.

## When to use this skill

- The user has a source document and wants an ontology extracted from it end to
  end ("build an ontology from this manual", "run the whole pipeline").
- You want a single reproducible command that produces the admitted ontology plus
  the per-decision audit trail.

Use a single `stageN-*` skill instead when the user wants just one step (only
scope, only surface candidates, only the RITE review), or wants to inspect and
edit a stage's output before continuing.

## The seven stages and the file handoff

Each stage writes its artifact **next to the source document**. That colocation
is what lets Stages 1b, 2b and 3 auto-detect the Stage 0 scope as a sibling file,
so keep every artifact in the source's directory. For a source `SRC.md`:

| Stage | Skill | Reads | Writes |
|-------|-------|-------|--------|
| 0  | stage0-scope               | `SRC.md`                                   | `SRC_scope.json` |
| 1  | stage1-surface-candidates  | `SRC.md`                                   | `SRC_candidates.json` |
| 1b | stage1b-classify-candidates| `SRC_candidates.json`                      | `SRC_candidates_gated.json` |
| 2  | stage2-name-vocabulary     | `SRC_candidates_gated.json` (+ source)     | `SRC_candidates_gated_vocabulary.json` |
| 2b | stage2b-salience           | `..._vocabulary.json` (+ scope, auto)      | `..._vocabulary_salient.json` |
| 3  | stage3-synthesize-structure| `..._salient.json` (+ scope, auto)         | `..._salient_synth_structure.json` |
| 4  | stage4-review-ontology     | `..._synth_structure.json` (+ source)      | `..._reviewed.json` + `..._admitted.ttl` |

Two handoffs worth remembering: Stage 1b's **gated** file (not the raw
candidates) is what feeds Stage 2; and the Stage 0 scope is consumed by 1b, 2b
and 3, so Stage 0 must run before them even in a partial run.

## How to run it (the orchestrator)

`scripts/run_pipeline.py` chains all seven stages as separate subprocesses (so
each stage's sibling imports never collide) and computes the filenames above.

```bash
# Full run, LLM-assisted (the default).
python scripts/run_pipeline.py /path/to/SRC.md

# Every deterministic path (Stage 3 still needs an LLM and will say so).
python scripts/run_pipeline.py /path/to/SRC.md --no-llm

# A sub-range, e.g. re-run from classification through structure.
python scripts/run_pipeline.py /path/to/SRC.md --from 1b --to 3

# See the exact commands without running anything.
python scripts/run_pipeline.py /path/to/SRC.md --dry-run
```

Flags: `--from`/`--to` (stage ids `0 1 1b 2 2b 3 4`), `--no-llm`, `--no-ttl`,
`--env PATH`, `--keep-going`, `--dry-run`. The final admitted ontology is
`SRC_candidates_gated_vocabulary_salient_synth_structure_admitted.ttl`.

## How to run it (stage by stage)

When you want to inspect or edit between steps, invoke the stage skills (or their
scripts) directly, in order. Each writes next to its input:

```bash
python stage0-scope/scripts/extract_scope.py            SRC.md --llm
python stage1-surface-candidates/scripts/surface_candidates.py SRC.md
python stage1b-classify-candidates/scripts/classify_candidates.py SRC_candidates.json --llm
python stage2-name-vocabulary/scripts/name_vocabulary.py SRC_candidates_gated.json
python stage2b-salience/scripts/rank_salience.py        SRC_candidates_gated_vocabulary.json
python stage3-synthesize-structure/scripts/synthesize_structure.py SRC_candidates_gated_vocabulary_salient.json
python stage4-review-ontology/scripts/review_ontology.py SRC_candidates_gated_vocabulary_salient_synth_structure.json --llm --emit-ttl
```

## Setup

- **API keys.** Stages 0, 1b, 3 and 4 use Gemini; Stage 2 uses OpenAI embeddings.
  Copy `.env.example` to `.env` and fill in the keys. The orchestrator passes
  `--env` to every stage that accepts it; auto-detection looks for `.env` at this
  folder root first.
- **Dependencies.** Each stage has its own `requirements.txt` (spaCy + a model for
  Stage 1, the Gemini and OpenAI SDKs, an RDF/reasoner stack for Stages 3 to 4).
  Install them into one environment before running the full pipeline.
- **LLM-free smoke test.** Stages 0, 1, 1b, 2 (with `--no-embeddings`) and 2b run
  deterministically, so `--no-llm --to 2b` exercises the chain without any keys.

## What you get

- `SRC_..._admitted.ttl` — the admitted ontology in Turtle.
- `SRC_..._reviewed.json` — every element with its verdict (accept / reject /
  escalate) and the evidence behind it.
- Intermediate artifacts for each stage, so any step can be re-run or audited.

This pipeline learns the **ontology** (the T-Box). Populating an
ontology-compliant **knowledge graph** (the A-Box) from sources, given that
ontology, is the separate job covered by Article 5 of the companion series and
its `scima/kg_extraction.py`.
