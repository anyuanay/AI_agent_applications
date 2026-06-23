---
name: serialize-owl
description: Emit a reviewed set of proposed classes as a versioned Turtle/OWL
  delta with extractionConfidence and sourceDocument provenance on every class.
  Stage 5 of ontology extraction. Use to land an accepted hierarchy as a
  scima_owl_vX_Y.ttl file ready to load.
tools: [to_turtle]
---
Procedure:
  1. Take the accepted `ProposedClass` set from `refine-ontology`.
  2. to_turtle() to render each class as `owl:Class` with `rdfs:subClassOf`,
     `rdfs:label`, `scima:extractionConfidence`, and `scima:sourceDocument`.
  3. Write the result as a versioned delta (for example `scima_owl_v0_7.ttl`)
     alongside the existing ontologies in `../../ontology_kg_for_agents/ontologies/`.
  4. Load it once to confirm it parses, then register the version with the loader.

Guidance for this stage:
- Emit a delta, not a rewrite. The file holds only the new classes, properties,
  and axioms over the prior version. Keep it additive so existing SPARQL queries
  and agent plans do not break.
- Provenance is mandatory, not optional. Every learned class carries its
  confidence and source document so a later reader can audit where it came from.
- Match the repo's Turtle conventions: prefix scheme, decimal-typed confidence,
  one class per block. See `references/turtle_conventions.md`.
- Never use em dashes in labels or comments, per repo style.

Run:
    python scripts/to_turtle.py                 # print the Turtle delta to stdout
    python scripts/to_turtle.py out.ttl         # write the delta to a file
