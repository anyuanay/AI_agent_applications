---
name: extract-concepts
description: Surface candidate ontology classes and is-a relations from raw source
  text using Hearst lexico-syntactic patterns and NER. Stage 1 of ontology
  extraction. Use when you have documents and need raw concept candidates to feed
  schema induction, not a finished ontology.
tools: [hearst_hyponyms, extract_concepts]
---
Procedure:
  1. Read the source text (default corpus: `../shared/corpus/emergency_procedures.txt`).
  2. hearst_hyponyms() the text to surface (hypernym, hyponym) pairs from patterns
     like "X such as A, B and C" and "X, including A, B and C".
  3. extract_concepts() to fold those pairs into a deduplicated candidate-class list.
  4. Hand the candidates to `induce-schema` for structured class induction.

Guidance for this stage:
- This is deliberately high-recall and noisy. Do not filter aggressively here;
  later stages validate. A missed concept cannot be recovered downstream, a wrong
  one only costs a review rejection.
- Hearst patterns are precise but sparse: they only fire on explicit enumerations.
  Treat NER candidates as the wider net and Hearst pairs as the high-confidence core.
- Keep provenance from the first touch: record which document each candidate came
  from so `refine-ontology` can ground it and `serialize-owl` can annotate it.
- See `references/hearst_pattern_catalog.md` for the full pattern set and limits.

Run:
    python scripts/hearst_patterns.py            # print hypernym/hyponym pairs
    python scripts/ner_concepts.py               # print candidate concepts
