---
name: stage2-map-ontology
description: Stage 2 of the ontology-compliant KG extraction pipeline. Maps each candidate triple (from Stage 1) to ontology IRIs or typed literals using OpenAI text-embedding-3-small for semantic matching and Gemini gemini-3.1-flash-lite as a LLM fallback. First time the ontology is consulted. Produces _ontology_index.json (consumed by Stage 3 directly) and _mapped_triples.json. Use after stage1-extract when asked to "map triples to the ontology", "link mentions to classes", or "run stage 2 of the KG pipeline".
---

# Stage 2 — Map Ontology

This is the **second stage** of the ontology-compliant KG extraction pipeline.
It reads `_candidate_triples.json` and `_sentences.json` from Stage 1, together
with a target OWL/Turtle ontology, and maps each surface triple onto ontology
IRIs or typed literals. Triples that cannot be fully mapped are dropped.

## First contact with the ontology

Stage 1 was fully open and text-driven. Stage 2 is the first time the ontology
is consulted. It pre-computes an ontology index (`_ontology_index.json`) that
Stage 3 reads directly — Stage 3 never re-parses the ontology file.

## Mapping pipeline per triple

For each candidate triple (subject, predicate, object):

### 1. Subject typing
Determine which ontology class the subject surface form represents an instance of.

1. Embed `"[surface] — [sentence]"` and compute cosine against all class label
   embeddings (OpenAI text-embedding-3-small). If top score ≥ threshold →
   assign class, mint individual IRI, `mapping_confidence: high`.
2. Else → Gemini LLM classifier (surface form + sentence + full class list) →
   `mapping_confidence: low`.
3. If LLM returns no class → triple is dropped.

### 2. Predicate mapping
Determine which ontology property the surface predicate corresponds to. Searches
direct labels AND inverse labels in parallel at every step.

1. Lemma-match + string-match against property labels and aliases (direct +
   inverse). Match → `mapping_confidence: high`.
2. Else → embedding cosine over property label table → if above threshold,
   `mapping_confidence: high`.
3. Else → Gemini LLM (surface predicate + sentence + full property list with
   inverse labels) → `mapping_confidence: low`.
4. If LLM cannot map → triple is dropped.

An **inverse match** (surface predicate matches the inverse property's labels)
sets an internal `inverted` flag and assigns `predicate_iri` to the forward
property. The subject and object are swapped after both ends are typed.

### 3. Object mapping
Determined by the **predicate kind** from the ontology (not Stage 1's
`object_type` tag, which is a consistency signal only):

- `object_property` → type the object as a class instance (same algorithm as
  subject typing, same confidence levels).
- `datatype_property` → parse the surface form as a typed literal:
  integer, decimal, date, or string. Unit strings (e.g. "47 psi") are split
  into numeric value (stored) and unit annotation (discarded). Always
  `mapping_confidence: high`.

If typing or parsing fails → triple is dropped.

### 4. Triple inversion (when predicate matched inverse labels)
Swap `(subject_iri, subject_type)` and `(object_iri, object_type)` so that
Stage 3's domain/range checks see the corrected positions. The `inverted` flag
is consumed here and not written to `_mapped_triples.json`.

### 5. Mapping confidence aggregation
`mapping_confidence` = min of the three component confidences. `high` only if
all three components scored `high`; `low` if any used LLM fallback.

### IRI minting
Individual IRIs follow the pattern `<namespace><ClassName>_<slug>`.

Slug: lowercase + replace non-alphanumeric runs with underscores + collapse.

A minted IRI registry tracks `(class_iri, slug) → surface_form` and:
- Same surface form → reuse existing IRI (repeated mention).
- Different surface form at same slug → append counter suffix `_2`, `_3`, …

## How to run

```bash
cd scripts
python map_ontology.py CANDIDATE_TRIPLES ONTOLOGY [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--sentences PATH` | inferred | `_sentences.json` path |
| `--output-dir DIR` | triples dir | where to write output files |
| `--entity-threshold F` | `0.50` | cosine threshold for entity typing |
| `--pred-threshold F` | `0.45` | cosine threshold for predicate mapping |
| `--no-llm` | off | skip LLM fallback |
| `--env PATH` | auto | `.env` with `OPENAI_API_KEY` + `GOOGLE_API_KEY` |
| `--spacy-model MODEL` | `en_core_web_sm` | spaCy model for predicate normalization |

Examples:

```bash
# Full run (embedding + LLM fallback)
python map_ontology.py /data/source_candidate_triples.json /data/ontology.ttl

# Embedding-only (no LLM calls, cheaper but lower recall)
python map_ontology.py /data/source_candidate_triples.json /data/ontology.ttl --no-llm

# Looser thresholds for smaller/noisier ontologies
python map_ontology.py /data/source_candidate_triples.json /data/ontology.ttl \
  --entity-threshold 0.40 --pred-threshold 0.35
```

## Output files

| File | Contents |
|---|---|
| `{stem}_ontology_index.json` | pre-computed class/property index for Stage 3 |
| `{stem}_mapped_triples.json` | successfully mapped triples with IRIs |
| `{stem}_unmapped_triples.json` | dropped triples with drop_reason annotation |

### Mapped triple schema (object_property)
```json
{
  "subject_iri":        "http://example.org/algebra-ch1-2#WholeNumbers_whole_numbers",
  "subject_type":       "http://example.org/algebra-ch1-2#WholeNumbers",
  "predicate_iri":      "http://example.org/algebra-ch1-2#hasSubconcept",
  "predicate_kind":     "object_property",
  "object_iri":         "http://example.org/algebra-ch1-2#CountingNumbers_counting_numbers",
  "object_type":        "http://example.org/algebra-ch1-2#CountingNumbers",
  "sentence_index":     12,
  "sentence_indices":   [12, 15],
  "sources":            ["dep_parse", "llm"],
  "mapping_confidence": "high"
}
```

### Mapped triple schema (datatype_property)
```json
{
  "subject_iri":        "http://example.org/algebra-ch1-2#PrimeNumber_prime_numbers",
  "subject_type":       "http://example.org/algebra-ch1-2#PrimeNumber",
  "predicate_iri":      "http://example.org/algebra-ch1-2#hasExample",
  "predicate_kind":     "datatype_property",
  "object_literal":     "2, 3, 5, 7, 11",
  "object_datatype":    "xsd:string",
  "sentence_index":     47,
  "sentence_indices":   [47],
  "sources":            ["llm"],
  "mapping_confidence": "low"
}
```

### Ontology index key schema
```json
{
  "namespace": "http://example.org/algebra-ch1-2#",
  "classes": [{"iri": "...", "label": "...", "aliases": [...], "superclasses": [...]}],
  "properties": [{"iri": "...", "label": "...", "kind": "object_property",
                   "domain": "...", "range": "...", "inverse_of": "...",
                   "inverse_labels": [...], "is_functional": false,
                   "max_cardinality": null}],
  "subclass_closure": {"<class_iri>": ["<ancestor_iri>", ...]},
  "disjointness_pairs": [["<class_iri_a>", "<class_iri_b>"]]
}
```

## Files

```
stage2-map-ontology/
├── SKILL.md                    # this file
├── requirements.txt
└── scripts/
    ├── map_ontology.py         # orchestrator + CLI (run this)
    ├── ontology_index.py       # OWL/Turtle → pre-computed JSON index
    ├── embedder.py             # OpenAI text-embedding-3-small with caching
    ├── iri_registry.py         # IRI minting + slug collision handling
    ├── llm_client.py           # Gemini gemini-3.1-flash-lite fallback wrapper
    ├── typer.py                # entity typing: embedding → LLM fallback
    ├── predicate_mapper.py     # predicate mapping: string-match → embedding → LLM
    └── object_mapper.py        # object routing: entity typing or literal parsing
```

## Requirements

- Python 3.10+
- `rdflib>=6.0` — OWL/Turtle parsing
- `openai>=1.0` — text-embedding-3-small (`OPENAI_API_KEY` in `.env`)
- `spacy>=3.7` with `en_core_web_sm` — predicate lemmatization
- `google-genai>=1.0` — Gemini fallback (`GOOGLE_API_KEY` in `.env`; not needed with `--no-llm`)

## What comes next

`_mapped_triples.json` and `_ontology_index.json` are the inputs to
`stage3-verify-admit`, which checks domain/range/disjointness/cardinality
constraints and emits the final admitted A-Box triples plus `rdf:type`
assertions for all admitted individuals.
