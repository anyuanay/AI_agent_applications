---
name: stage1-extract
description: Stage 1 of the ontology-compliant KG extraction pipeline. Surfaces all entity and value mentions from a source text (no ontology consulted), then produces raw and merged candidate triples. Uses a spaCy pass (NER + noun-chunk mentions, sentence segmentation, dependency-parse triples) combined with a Gemini LLM pass (batched entity/value surfacing + triple extraction), followed by entity mention merge and triple merge. Output is six JSON files written next to the source: _sentences.json, _entity_mentions.json, _dep_triples.json, _llm_triples.json, _candidate_triples.json, and _negated_triples.json. Use when given a source text and an ontology and asked to "extract KG triples", "populate an A-box", or "run the KG extraction pipeline".
---

# Stage 1 — Extract

This is the **first stage** of the ontology-compliant KG extraction pipeline
(`stage1-extract → stage2-map-ontology → stage3-verify-admit`). It takes a
raw source text and produces a normalized, merged set of candidate triples
ready for ontology mapping. The ontology is **not consulted here** — Stage 1
is fully open and text-driven.

## Goal

Surface all entity mentions and value mentions, generate raw triples from two
independent passes (spaCy dependency parse and Gemini LLM), then normalize and
merge everything into unified candidate sets. Recall-first: pull everything,
sort nothing.

## Two extraction passes

### spaCy pass
- **NER** — named entity extraction with entity-type to `mention_type` mapping:
  `PERSON`, `ORG`, `GPE`, `EVENT`, `FAC`, `NORP`, `LOC`, `PRODUCT`, `WORK_OF_ART` → `entity`;
  `CARDINAL`, `QUANTITY`, `DATE`, `TIME`, `MONEY`, `PERCENT` → `value`.
- **Noun-phrase chunks** — NPs not already covered by NER (`mention_type: entity`).
- **Sentence segmentation** — stored as `_sentences.json`, used by the LLM pass
  and by downstream stages.
- **Dependency parse** — `(nsubj, root-verb, dobj)` triples per sentence with
  negation detection (any `dep_ == "neg"` child on the root verb marks the triple
  `negated: true`). Stored as `_dep_triples.json`.

### Gemini LLM pass
Iterates over `_sentences.json` in batches of N (default 7). One API call per
batch extracts per-sentence:
1. Entity and value mentions with `mention_type` classification.
2. All `(subject, predicate, object)` triples. Negated triples are included
   with `negated: true` rather than silently omitted.

Stored as `_llm_triples.json`. LLM failures on a batch skip that batch gracefully.

### Entity mention merge
After both passes, deduplicate by normalized surface form (`lowercase`,
`whitespace-strip`). One record per unique form in `_entity_mentions.json`.

- `sources` collects all extractors: `ner`, `noun_chunk`, `llm`.
- `sentences` collects all sentence indices.
- `mention_type`: LLM wins when sources disagree (LLM overrides spaCy CARDINAL/
  QUANTITY/DATE mislabelings of proper names).

### Triple merge
Combine `_dep_triples.json` (source: `dep_parse`) and `_llm_triples.json`
(source: `llm`) into a unified set.

Normalization:
- Subjects/objects: `lowercase` + whitespace-collapse only (no lemmatization —
  lemmatization distorts named entities and multi-word identifiers).
- Predicates: `lowercase` + spaCy lemmatization.

Merge duplicates into one record with all source tags. Collect all sentence
indices; set `sentence_index` to the lowest (first occurrence).

Negation is sticky: a merged triple is `negated: true` if **either** source
marks it negated.

After merge:
- Non-negated triples → `_candidate_triples.json`
- Negated triples → `_negated_triples.json`

`object_type` is inherited from `_entity_mentions.json` by normalized object
lookup; it is omitted when the object is not in entity mentions (nullable).

## How to run

```bash
cd scripts
python extract.py SOURCE_FILE [options]
```

Output files are written next to the source (or to `--output-dir`).

| Option | Default | Meaning |
|---|---|---|
| `--no-llm` | off | spaCy pass only; no API key needed |
| `--model MODEL` | `gemini-2.0-flash-lite` | Gemini model id |
| `--batch-size N` | `7` | sentences per LLM batch |
| `--spacy-model MODEL` | `en_core_web_sm` | spaCy model (must be installed) |
| `--env PATH` | auto | explicit path to `.env` holding `GOOGLE_API_KEY` |
| `--output-dir DIR` | source dir | directory for output files |
| `--max-sentences N` | `0` (all) | cap LLM sentences (evenly sampled) |

Examples:

```bash
# Full run with Gemini
python extract.py /data/source.md

# spaCy only (no API key needed)
python extract.py /data/source.md --no-llm

# Larger Gemini model, batch size 10
python extract.py /data/source.md --model gemini-2.0-flash --batch-size 10

# Cap LLM pass to 200 sentences (evenly sampled)
python extract.py /data/source.md --max-sentences 200

# Explicit env file and output directory
python extract.py /data/source.md --env /path/to/.env --output-dir /data/output/
```

## Output files

All files use the source file stem as prefix (e.g., source `chapter1.md`
→ `chapter1_sentences.json`, etc.). Each file includes a metadata header
(`stage`, `input_file`, `generated_at`, `stats`).

| File | Contents |
|---|---|
| `{stem}_sentences.json` | ordered sentence records |
| `{stem}_entity_mentions.json` | merged entity/value mentions |
| `{stem}_dep_triples.json` | raw dep-parse triples (pre-merge) |
| `{stem}_llm_triples.json` | raw LLM triples (pre-merge) |
| `{stem}_candidate_triples.json` | normalized merged non-negated triples |
| `{stem}_negated_triples.json` | merged triples with negated=true |

### Record schemas

`_entity_mentions.json`:
```json
{ "surface_form": "Commander Diaz", "mention_type": "entity",
  "sources": ["ner", "llm"], "sentences": [3] }
```

`_dep_triples.json` / `_llm_triples.json`:
```json
{ "subject": "whole numbers", "predicate": "include",
  "object": "counting numbers", "sentence_index": 12 }
{ "subject": "Commander", "predicate": "command", "object": "Team B",
  "sentence_index": 5, "negated": true }
```

`_candidate_triples.json`:
```json
{
  "subject":          "whole numbers",
  "predicate":        "include",
  "object":           "counting numbers",
  "object_type":      "entity",
  "sentence_index":   12,
  "sentence_indices": [12, 15],
  "sources":          ["dep_parse", "llm"]
}
```

`_negated_triples.json`:
```json
{
  "subject":          "Commander",
  "predicate":        "command",
  "object":           "Team B",
  "negated":          true,
  "sentence_index":   5,
  "sentence_indices": [5],
  "sources":          ["dep_parse"]
}
```

## Files

```
stage1-extract/
├── SKILL.md                  # this file
├── requirements.txt
└── scripts/
    ├── extract.py            # orchestrator + CLI (run this)
    ├── models.py             # Sentence, RawMention, EntityMention,
    │                         #   DepTriple, LLMTriple, CandidateTriple
    ├── spacy_pass.py         # NER + noun chunks + dep parse + sentences
    ├── llm_pass.py           # Gemini batched entity/value + triple extraction
    ├── entity_merge.py       # entity mention merge
    └── triple_merge.py       # triple normalization + merge
```

## Requirements

- Python 3.10+
- `spacy>=3.7` with `en_core_web_sm` (`python -m spacy download en_core_web_sm`)
- `google-generativeai>=0.5` (for the LLM pass; not needed with `--no-llm`)
- `GOOGLE_API_KEY` in a `.env` file (for the LLM pass)

## What comes next

The six output files are consumed by `stage2-map-ontology`, which reads
`{stem}_candidate_triples.json` and `{stem}_sentences.json` alongside the
target ontology to map each surface triple onto ontology IRIs or typed literals.
`{stem}_negated_triples.json` is archived; it is not forwarded to Stage 2.
