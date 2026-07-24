---
name: stage3-verify-admit
description: Stage 3 of the ontology-compliant KG extraction pipeline. Validates mapped triples from Stage 2 against the ontology index using five rule-based checks (domain, range, disjointness, cardinality, datatype repair). Emits admitted triples, rdf:type assertions, and rejected triples. Needs no LLM or embedding — all verification is logic-only against the pre-computed index. Use after stage2-map-ontology when asked to "verify triples", "run ontology checks", "admit A-Box triples", or "run stage 3 of the KG pipeline".
---

# Stage 3 — Verify & Admit

This is the **third and final stage** of the ontology-compliant KG extraction
pipeline. It reads `_mapped_triples.json` and `_ontology_index.json` (both
written by Stage 2) and produces the final A-Box.

Stage 3 is entirely rule-based. No LLM or embedding calls are made — all
checks operate against the pre-computed ontology index.

## Processing pipeline

### Pre-processing: sort by confidence

All mapped triples are sorted highest-confidence-first before any checks run.
This ensures that when two triples conflict (on disjointness or cardinality)
the higher-confidence triple is processed first and wins.

Four-level sort key (ascending = highest confidence first):

| Level | Signal | Values |
|-------|--------|--------|
| 1 | Cross-sentence corroboration (`sentence_indices`) | `≥2` sentences → 0; else → 1 |
| 2 | Source agreement | `dep_parse+llm` → 0; `dep_parse` → 1; `llm` → 2 |
| 3 | Stage 2 mapping confidence | `high` → 0; `low` → 1 |
| 4 | Sentence index (tiebreaker) | Earlier sentence first |

### Check 1 — Domain

Is the subject's assigned type a subclass (via transitive closure) of the
property's declared `rdfs:domain`? If no domain is declared, the check passes
unconditionally.

- Pass: continue  
- Fail: `reason: domain_violation`

### Check 2 — Range (object_property)

Is the object's assigned type a subclass of the declared `rdfs:range`?
Unconditional pass if no range declared.

- Pass: continue  
- Fail: `reason: range_violation`

### Check 3 — Range (datatype_property)

Is the literal's XSD datatype compatible with the declared range type?

- Compatible: continue
- Incompatible but repairable: cast the literal (e.g. `"47"` string →
  `xsd:integer 47`). On success: `verdict: repaired`.
- Cast fails: `reason: datatype_mismatch`

### Check 4 — Disjointness

Does assigning the subject's (or object's) type to an individual already
admitted with a disjoint type violate `owl:disjointWith` / `owl:AllDisjointClasses`?

Greedy one-pass: first-processed (higher-confidence) triple wins; conflicting
lower-confidence triple is rejected.

- Pass: continue  
- Fail: `reason: disjointness_violation`

### Check 5 — Cardinality

**5a — Functional / max-cardinality (subject side):** if the property is
`owl:FunctionalProperty` or carries `owl:maxCardinality N`, reject any triple
that would exceed the allowed count for `(subject_iri, predicate_iri)`.

**5b — Inverse-functional (object side):** if the property is
`owl:InverseFunctionalProperty`, reject any triple whose
`(predicate_iri, object_iri)` pair already has a different subject IRI in the
admitted set.

Lower-bound constraints (`owl:minCardinality`) are not checked — a partial
extraction cannot be required to satisfy minimum coverage.

- Pass: `verdict: admit`  
- Fail: `reason: cardinality_violation`

### rdf:type emission

After all triples are processed, deduplicated `rdf:type` triples are emitted
for every individual that appears as a subject or object in any admitted triple.

## How to run

```bash
cd scripts
python verify_admit.py MAPPED_TRIPLES [ONTOLOGY_INDEX] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `ONTOLOGY_INDEX` | inferred from mapped path | `_ontology_index.json` from Stage 2 |
| `--output-dir DIR` | same dir as mapped triples | where to write output files |

Examples:

```bash
# Standard run (ontology index auto-inferred)
python verify_admit.py /data/source_mapped_triples.json

# Explicit paths
python verify_admit.py /data/source_mapped_triples.json /data/source_ontology_index.json \
  --output-dir /data/output/
```

## Output files

| File | Contents |
|---|---|
| `{stem}_admitted_triples.json` | Property triples that passed all checks |
| `{stem}_type_assertions.json` | Deduplicated `rdf:type` triples for all admitted individuals |
| `{stem}_rejected_triples.json` | Triples dropped at any check with `reason` annotation |

### Admitted triple schema (object_property)
```json
{
  "subject_iri":        "http://example.org/algebra-ch1-2#WholeNumbers_whole_numbers",
  "subject_type":       "http://example.org/algebra-ch1-2#WholeNumbers",
  "predicate_iri":      "http://example.org/algebra-ch1-2#hasSubconcept",
  "predicate_kind":     "object_property",
  "sentence_index":     12,
  "sources":            ["dep_parse", "llm"],
  "mapping_confidence": "high",
  "verdict":            "admit",
  "object_iri":         "http://example.org/algebra-ch1-2#CountingNumbers_counting_numbers",
  "object_type":        "http://example.org/algebra-ch1-2#CountingNumbers"
}
```

### Admitted triple schema (datatype_property, repaired)
```json
{
  "subject_iri":        "http://example.org/algebra-ch1-2#PrimeNumber_prime_numbers",
  "subject_type":       "http://example.org/algebra-ch1-2#PrimeNumber",
  "predicate_iri":      "http://example.org/algebra-ch1-2#hasExponentValue",
  "predicate_kind":     "datatype_property",
  "sentence_index":     47,
  "sources":            ["dep_parse"],
  "mapping_confidence": "high",
  "verdict":            "repaired",
  "object_literal":     2,
  "object_datatype":    "xsd:integer"
}
```

### Type assertion schema
```json
{ "subject_iri": "http://example.org/.../WholeNumbers_whole_numbers",
  "predicate_iri": "rdf:type",
  "object_iri":    "http://example.org/.../WholeNumbers" }
```

### Rejected triple schema
```json
{
  "subject_iri":    "...",
  "predicate_iri":  "...",
  "object_iri":     "...",
  "verdict":        "reject",
  "reason":         "domain_violation"
}
```

## Files

```
stage3-verify-admit/
├── SKILL.md                    # this file
├── requirements.txt            # no external dependencies
└── scripts/
    ├── verify_admit.py         # orchestrator + CLI (run this)
    ├── confidence_sorter.py    # 4-level sort (highest confidence first)
    ├── checker.py              # 5 checks with running admitted-state
    └── datatype_repair.py      # XSD compatibility check + cast repair
```

## Requirements

- Python 3.10+
- No external packages (standard library only)

## What came before

`_mapped_triples.json` and `_ontology_index.json` are produced by
`stage2-map-ontology` using OpenAI `text-embedding-3-small` for semantic
matching and Gemini `gemini-3.1-flash-lite` as LLM fallback. Stage 3 reads
these files directly and does not call either service.
