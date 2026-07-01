# Generative AI Agents and Applications

Companion code for the **Generative AI Agents and Applications** series published on [Medium](https://medium.com/@anyuanay).

Each directory in this repository contains full, runnable source code discussed in the series. **Anyone is free to fork, adapt, and vibe-code this codebase to fit their own datasets, domains, and needs; nothing here is meant to be used as-is without adjustment.**

Each project is built up layer by layer rather than as a fresh codebase per article. The **literature review agent** is the recurring example for *Building Agents That Work*: a flat tool-use loop in the early parts, then memory and retrieval, planning and orchestration, runtime primitives, safety, observability, evaluation, and uncertainty. The **SCIMA** agent backs *Ontology and Knowledge Graphs for Intelligent Agents*: one ontology and knowledge graph that grows from v0.1 through v0.8, from OWL building blocks through knowledge graphs, context graphs, ontology extraction, and ontology-compliant knowledge-graph extraction, with two parallel skill sets alongside the backing package: a seven-stage ontology-extraction skill set (Article 4) and a three-stage KG-extraction skill set (Article 5). Each project's own README maps every module to the part or article that introduced it.

---

## Series Index

| Directory                                                            | Spans      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`lit_review_agent/`](./lit_review_agent/)                           | Parts 1–18 | An agent that searches Semantic Scholar, reads abstracts, and drafts a literature review. Grows across the series to include a vector store and property-graph memory, a plan/sub-agent orchestrator, skills and hooks, prompt-injection defenses, span-based tracing, model routing and cost rollups, an offline evaluation suite, progressive disclosure, a confidence gate, and a stacked outer-loop system (event-driven trigger, verification loop, and hill-climbing).                                                                                                                                                                                                                                                                                                             |
| [`ontology_kg_for_agents/`](./ontology_kg_for_agents/)               |            | **SCIMA**, the SmartCity Infrastructure Management Agent. One coherent ontology and knowledge graph that grows alongside the *Ontology and Knowledge Graphs for Intelligent Agents* series: OWL DL building blocks, versioned SCIMA-OWL files (v0.1 through v0.8), triple population, named graphs, SPARQL and geo queries, k-hop context graphs with relevance scoring and eviction, the seven-stage ontology-extraction pipeline with a RITE review guard (Article 4), and a three-stage Extract/Map/Verify pipeline that populates an ontology-compliant A-Box with a five-check compliance gate (domain, range, datatype repair, disjointness, cardinality) against SCIMA-OWL v0.8 (Article 5). Tests pin every ontology version to the series Growth Tracker.                       |
| [`ontology_KG_extraction_skills/`](./ontology_KG_extraction_skills/) |            | Two parallel sets of composable agent skills, one per article. `ontology_extraction_skills/` packages the Article 4 ontology-extraction pipeline as seven self-contained stage skills (`stage0-scope` through `stage4-review-ontology`), with a top-level `ontology-extraction` entry skill and `scripts/run_pipeline.py` that chain all stages end to end. `KG_extraction_skills/` packages the Article 5 KG-extraction pipeline as three stage skills: `stage1-extract` (spaCy NER + dep parse + Gemini LLM pass + entity and triple merge), `stage2-map-ontology` (OpenAI embedding typer + Gemini LLM fallback + IRI registry + predicate and object mapper), and `stage3-verify-admit` (five-check logic-only compliance gate, rdf:type emission). Each stage also runs standalone. |

More projects will be added as the series continues.

---

## Getting Started

Each project is self-contained. Navigate into the project directory and follow the steps below.

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) for the live literature review agent. Its evaluation suite, retrieval eval, graph traversals, memory, and hooks all run offline with no key. The SCIMA `ontology_kg_for_agents` package runs entirely offline, including its ontology-extraction and knowledge-graph-extraction pipelines (the LLM and embedding steps ship as deterministic stubs). The standalone extraction skill sets are the exception: `ontology_extraction_skills/` LLM-assisted stages call Gemini and OpenAI (the deterministic stages run without keys via `--no-llm --to 2b`); `KG_extraction_skills/` stages 1 and 2 call Gemini and OpenAI, while stage 3 is fully offline.

### Setup

```bash
cd <project-directory>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

### Running the literature review agent

```bash
# The flat single-agent loop (default goal: GNN papers, 2024)
python harness.py

# A custom goal
python harness.py "Find the five most-cited papers on RAG published in 2023-2024 and draft a literature review."

# Limit agent turns
python harness.py "..." --max-turns 10

# The orchestrated survey: plan -> parallel sub-agents -> synthesize -> reflect -> save
python orchestrator.py "Survey graph neural networks across application areas."
```

Output is saved to `lit_review_agent/output/`.

### Running the offline pieces (no API key)

```bash
python eval_suite.py       # the eval ladder: worker-B abstain, parser regression, confidence gate
python retrieval_eval.py   # the Recall@k retrieval slice
python uncertainty.py      # the confidence gate on its own
python loop.py             # Part 18: two morning runs of the four-loop stack (add --live for real makers)
```

### Running the SCIMA ontology / knowledge graph (no API key)

```bash
cd ontology_kg_for_agents

# Article 1: inspect the v0.1 ontology
python -m scima.ontology

# Article 2: populate the v0.2 knowledge graph and run the geo query
python -m scima.knowledge_graph --populate sample --query lights-near I-204

# Article 3: build a context graph and trace its evolution across agent turns
python -m scima.context_graph --build --focal Incident_I204 --goal resolve
python -m scima.context_graph --trace I-204

# Article 4: extract an ontology from sources through the seven-stage pipeline
python -m scima.ontology_extraction --corpus corpus/emergency_procedures.txt
python -m scima.ontology_extraction --emit   # machine copy under build/

# Article 5: extract an ontology-compliant KG from Incident Report I-204
python -m scima.kg_extraction
python -m scima.kg_extraction --corpus corpus/incident_report_I204.txt

# Run the tests that pin each article's ontology to its Growth Tracker
pytest
```

The Article 4 extraction pipeline is also packaged as a standalone seven-stage
skill set (`stage0-scope` through `stage4-review-ontology`) under
`ontology_KG_extraction_skills/ontology_extraction_skills/`, with an
`ontology-extraction` entry skill and `scripts/run_pipeline.py` that run all
seven stages on a source document end to end; see that directory's `SKILL.md`
for the orchestrator and per-stage usage.

The Article 5 KG-extraction pipeline is packaged as a three-stage skill set
(`stage1-extract`, `stage2-map-ontology`, `stage3-verify-admit`) under
`ontology_KG_extraction_skills/KG_extraction_skills/`; each stage skill runs
standalone and hands off JSON files to the next.

---

## Repository Structure

```
.
├── lit_review_agent/       # The recurring example, built across Parts 1-17
│   ├── harness.py          # Control loop, system prompt, tool registry (incl. ask_user), tracing + hooks + provenance, run_worker
│   ├── orchestrator.py     # Plan -> parallel sub-agents -> synthesize -> reflect -> save, with model routing
│   ├── loop.py             # Part 18: the four-loop stack (agent / verification / event-driven / hill-climbing)
│   ├── tools.py            # search / fetch / save / done / ask_user, structured errors, untrusted envelope, graph ingestion, parse_year_range
│   ├── tools_server.py     # The same tools as an optional MCP server
│   ├── memory.py           # PaperMemory (remember / recall) and trim_for_window
│   ├── vector_store.py     # A tiny persistent cosine store
│   ├── embeddings.py       # One embed function, same model both sides (a stand-in)
│   ├── textutil.py         # chunk and reorder_for_window (lost-in-the-middle)
│   ├── graph.py            # Paper/Author/Venue/Topic property graph, has_edge structural check
│   ├── graph_tools.py      # Typed traversals, query_graph passthrough, hybrid_recall
│   ├── hooks.py            # guard_file_writes, enforce_budget, require_approval, log_call (Allow / Block)
│   ├── skills.py           # Progressive-disclosure skill loader
│   ├── skills/             # systematic_review SKILL.md + checklist
│   ├── production.py       # Model routing, version pinning, cost rollup (the span tree is a bill)
│   ├── uncertainty.py      # Confidence gate, detection signals, provenance-tagged Claim, response_policy
│   ├── disclosure.py       # Progressive disclosure of a run at three altitudes (summary / plan / trace)
│   ├── tracing.py          # Span, Tracer, span(), and the Run artifact
│   ├── eval_suite.py       # Grading ladder, grade_run, LLM judge, parser regression, confidence-gate cases
│   ├── retrieval_eval.py   # Standalone Recall@k retrieval slice
│   ├── eval_fixtures/      # Recorded traces: the worker-B failure and its fix
│   ├── requirements.txt
│   ├── README.md           # Module-to-part map and notes on the stand-ins
│   └── output/             # Generated reviews (git-ignored)
├── ontology_kg_for_agents/ # SCIMA, built across the ontology & KG series
│   ├── scima/              # The SCIMA package (grows across the series)
│   │   ├── ontology.py             # Load + inspect versioned SCIMA-OWL files (v0.1 through v0.8)
│   │   ├── building_blocks.py      # Article 1: classes, individuals, properties, axioms
│   │   ├── knowledge_graph.py      # Article 2: populate triples, named graphs, SPARQL, geo query
│   │   ├── context_graph.py        # Article 3: k-hop projection, relevance scoring, turns, eviction
│   │   ├── ontology_extraction.py  # Article 4: seven-stage Scope/Surface/Sort/Name/Salience/Structure/Review + RITE
│   │   └── kg_extraction.py        # Article 5: three-stage Extract/Map/Verify pipeline, OntologyIndex builder, five-check compliance gate
│   ├── ontologies/         # Canonical SCIMA-OWL, one Turtle file per version (v0.1 through v0.8)
│   │   ├── scima_owl_v0_1.ttl      # Article 1: core class hierarchy
│   │   ├── scima_owl_v0_2.ttl      # Article 2: sensing vocabulary
│   │   ├── scima_owl_v0_5.ttl      # Article 3: agent, goal, context-graph vocabulary
│   │   ├── scima_owl_v0_6.ttl      # Article 4: emergency-response vocabulary (26 classes, 34 props, 15 axioms)
│   │   └── scima_owl_v0_8.ttl      # Article 5: KG extraction schema (30 classes, 41 props, 18 axioms)
│   ├── corpus/
│   │   ├── emergency_procedures.txt    # Article 4 source corpus
│   │   └── incident_report_I204.txt    # Article 5 source: 6-sentence incident narrative
│   ├── tests/              # Per-article tests pinned to the Growth Tracker
│   │   ├── test_article_01.py
│   │   ├── test_article_02.py
│   │   ├── test_article_03.py
│   │   ├── test_article_04.py
│   │   └── test_article_05.py      # v0.8 schema + three-stage pipeline + 45 behavioural checks
│   ├── requirements.txt
│   └── README.md           # Article-to-code map and design notes
├── ontology_KG_extraction_skills/      # Two parallel skill sets for Articles 4 and 5
│   ├── ontology_extraction_skills/     # Article 4: ontology-extraction pipeline as agent skills
│   │   ├── SKILL.md                    # ontology-extraction: the end-to-end orchestrator entry skill
│   │   ├── scripts/run_pipeline.py     # chains all seven stages with the correct file handoff
│   │   └── stage0-scope … stage4-review-ontology/  # seven self-contained stage skills
│   └── KG_extraction_skills/           # Article 5: KG-extraction pipeline as agent skills
│       ├── stage1-extract/             # spaCy NER + dep parse + Gemini LLM pass + entity/triple merge
│       ├── stage2-map-ontology/        # OpenAI embedding typer + Gemini LLM fallback + IRI registry + predicate/object mapper
│       └── stage3-verify-admit/        # five-check logic-only compliance gate + confidence sort + rdf:type emission
└── README.md
```

---

## License

MIT
