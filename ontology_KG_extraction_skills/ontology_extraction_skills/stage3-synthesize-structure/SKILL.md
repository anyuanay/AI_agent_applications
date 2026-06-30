---
name: stage3-synthesize-structure
description: Stage 3 of ontology extraction. Has gemini-3.1-flash-lite synthesize a parsimonious, well-organized ontology (an is-a taxonomy with a few coined abstract parents, plus relations with proper domain and range) from the salient candidate vocabulary, scoped by the Stage 0 domain statement and competency questions, and fenced by grounding so every class traces back to a candidate (anything introduced is flagged). Outputs a structure (classes, coined_parents, relationships, axioms, consistency) that Stage 4 reviews. Use after salience ranking (Stage 2b) and before review (Stage 4), or when asked to "induce structure", "build the taxonomy / is-a hierarchy", "synthesize the ontology", or "assign domain and range". Reads a salient vocabulary JSON (and the Stage 0 scope) and writes <stem>_synth_structure.json next to it.
---

# Synthesize Structure (LLM, grounded)

This skill performs the **structuring** step of ontology extraction: wire the flat
salient vocabulary into a shape (an is-a taxonomy with a few coined abstract
parents, plus relations with proper domain and range). Rather than inducing the
taxonomy purely bottom-up, it has an LLM *synthesize* a clean ontology from the
salient candidates, then grounds the result back in the evidence. The output
(`classes`, `coined_parents`, `relationships`, `axioms`, `consistency`) is what
Stage 4 (review) ratifies.

The single job of this step: **commit to a parsimonious, well-formed structure**,
using the LLM's strength (coherent hierarchy, sensible domain/range) while fencing
its weakness (hallucination) with grounding and scope.

## When to use this skill

- You have a salient vocabulary (Stage 2b) and a Stage 0 scope, and you want a
  clean is-a hierarchy with proper domain/range on relations.
- The user asks to "induce structure", "build the taxonomy", "synthesize the
  ontology", or "assign domain and range".

Why synthesis (not a purely bottom-up induction): a single corpus leaves a
bottom-up taxonomy fragmented (many orphans, edges only where a head noun is also
a concept) and, lacking parsed triples, forces domain/range to be guessed by
co-occurrence (noisy). The LLM organizes the whole class set at once and assigns
domain/range from the class set directly, while grounding keeps it honest.

## Mechanism

### 1. LLM taxonomy (`scripts/llm_synthesize.py`)

Given the Stage 0 domain statement and competency questions and the salient
candidate concepts (with mention counts), `gemini-3.1-flash-lite` returns a list
of classes, each with one parent, choosing classes from the candidates, omitting
duplicates/near-synonyms/procedural terms, and coining a few abstract parents
(flagged) where a group shares an unnamed parent. The prompt favors **coverage**
of the domain's concepts organized into a clean hierarchy, not minimalism.

### 2. Ground (`scripts/ground.py`)

Every label the LLM produced is mapped back to a candidate by a normalized key, so
the synthesized class inherits real provenance (mention count, alternate labels).
A label that matches no candidate and is not flagged coined is an **LLM
introduction**: kept (recall-first) but flagged `llm_introduced`, so Stage 4 must
corpus-check it before admitting. Omitted candidates are recorded for the feedback
loop.

### 3. LLM relations

Given the synthesized class labels, the candidate relations, the competency
questions, and the scope's suggested relations, the LLM returns a comprehensive
set of relations, each with a `domain` and `range` that are class labels. These
are resolved to class ids; a relation whose ends do not resolve is dropped. This
is where domain/range finally come out clean.

### 4. Axioms + reasoner (`scripts/reason.py`)

Disjointness among coined-family siblings; the lightweight reasoner breaks any
cycle and relaxes contradictory disjointness, then reports consistency.

## How to run

```bash
cd scripts
python synthesize_structure.py <SALIENT_VOCAB_JSON> [options]
```

The result is written next to the input as `<stem>_synth_structure.json`. The
Stage 0 scope is auto-detected.

| Option | Default | Meaning |
|---|---|---|
| `--scope PATH` | auto-detect | Stage 0 scope JSON (domain statement, competency questions, relations). |
| `--llm-model` | `gemini-3.1-flash-lite` | Gemini model id. |
| `--max-concepts N` | `0` (all) | Cap candidate concepts sent to the LLM. |
| `--max-relations N` | `200` | Cap candidate relations sent to the LLM. |
| `--env PATH` | auto | Explicit `.env` holding `GOOGLE_API_KEY`. |
| `--output PATH` | auto | Override the output path. |

## Output

The structure schema (`classes`, `coined_parents`,
`relationships`, `axioms`, `consistency`, `stats`), plus an `omitted_candidates`
list (concepts the LLM did not include, kept as feedback). Stage 4 reads it
directly.

## Files

```
stage3-synthesize-structure/
├── SKILL.md
├── requirements.txt
├── scripts/
│   ├── synthesize_structure.py  # orchestrator + CLI (run this)
│   ├── llm_synthesize.py        # gemini taxonomy + relations calls
│   ├── ground.py                # map LLM labels back to candidates; flag introductions
│   ├── reason.py                # lightweight reasoner (cycles + disjointness)
│   └── model.py                 # ClassNode / CoinedParent / Relationship / Axiom
└── references/
    └── method.md
```

## Requirements

- Python 3.10+
- `google-genai` + `GOOGLE_API_KEY`

## Notes and limitations

- **Grounded, not free.** The LLM may only name classes from the candidates; any
  introduction is flagged for Stage 4 to corpus-check. This keeps the provenance
  invariant despite using generative synthesis.
- **Parsimony is a prompt dial.** The taxonomy prompt balances coverage against
  minimalism; tighten or loosen it for a smaller or richer ontology. Omitted
  candidates are recorded, never lost.
- **Some LLM variance.** Re-runs can differ slightly in which relations or coined
  parents appear; the grounding step keeps every run honest, and Stage 4 is the
  final gate.
- **Relations are the main win.** Domain/range are assigned by the LLM over the
  class set, which is far cleaner than the bottom-up co-occurrence heuristic.
