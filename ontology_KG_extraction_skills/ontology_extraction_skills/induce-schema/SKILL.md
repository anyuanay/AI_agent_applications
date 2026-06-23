---
name: induce-schema
description: Induce a candidate OWL class hierarchy from source text with an LLM
  using a structured-output (JSON schema) prompt. Stage 2 of ontology extraction.
  Use when you have source documents or concept candidates and need parented
  classes with confidence and provenance, not just raw terms.
tools: [extract_ontology_classes]
---
Procedure:
  1. Load the source text (default: `../shared/corpus/emergency_procedures.txt`).
  2. extract_ontology_classes() with the system prompt in
     `references/system_prompt.md`, which forces JSON output matching the schema.
  3. Parse the JSON into `ProposedClass` records (iri, parent, label, confidence, source).
  4. Pass the proposals to `cluster-classes` for a coherence check, then to
     `refine-ontology` for review.

Guidance for this stage:
- Always demand structured output. Prose responses are not parseable and break
  the pipeline. The prompt says "Return ONLY valid JSON"; strip code fences before
  json.loads (the script already does).
- The LLM will hallucinate. It invents fluent, plausible classes the source never
  supports. Do not trust confidence alone; grounding against the corpus happens in
  `refine-ontology`. The offline `StubLLMClient` plants one hallucination
  (`UnicornEvacuationProtocol`) on purpose so you can see the guard fire.
- Carry `source` provenance through every class. A class with `source: "(none)"`
  is a red flag for review.
- To run against a real model, swap `StubLLMClient` for a client whose
  `complete(system, user) -> str` returns the model's text. Default to the latest
  Claude model for the real run.
- See `references/induction_examples.md` for few-shot templates.

Run:
    python scripts/induce_classes.py            # induce classes (offline stub by default)
