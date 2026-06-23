# Ontology Extraction Skills

A collection of composable agent skills that extract and refine an OWL class
hierarchy from source documents. The pipeline mirrors Article 4 of the series
("Extracting Ontologies from Sources") and is backed by the runnable package in
`../../ontology_kg_for_agents/scima/`.

## The pipeline

```
sources ──► extract-concepts ──► induce-schema ──► cluster-classes ──► refine-ontology ──► serialize-owl ──► scima_owl_vX_Y.ttl
            (Hearst + NER)        (LLM schema)      (family grouping)   (RITE review)        (versioned Turtle)
```

Each stage is a self-contained skill with its own `SKILL.md`. They can be run
individually or chained end to end.

| Skill | Stage | What it does |
| ----- | ----- | ------------ |
| `extract-concepts` | 1 | Surface candidate classes and relations from raw text (Hearst patterns, NER). |
| `induce-schema` | 2 | Ask an LLM for a structured-output OWL class hierarchy. |
| `cluster-classes` | 3 | Group leaf classes into families and check cluster coherence. |
| `refine-ontology` | 4 | Run the RITE loop (Refine, Inspect, Test, Extend) with a hallucination guard and a reasoner consistency pass. |
| `serialize-owl` | 5 | Emit the reviewed classes as versioned Turtle with provenance. |

## Shared assets

`shared/` holds the cross-skill dataclasses (`ProposedClass`, `RiteResult`), a
sample source corpus, and reference notes used by more than one skill. Scripts
import from `shared/` and from the backing `scima` package rather than copying
logic.

## Conventions

- Every skill follows the `SKILL.md` frontmatter shape (`name`, `description`,
  `tools`) used elsewhere in this repo.
- Scripts add the repo root to `sys.path` and import the real functions from
  `scima.ontology_learning`, so the skills and the backing package never drift.
- Prose here and in every skill avoids em dashes, per repo style.
