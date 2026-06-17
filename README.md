# Generative AI Agents and Applications

Companion code for the **Generative AI Agents and Applications** series published on [Medium](https://medium.com/@anyuanay).

Each directory in this repository contains full, runnable source code discussed in the series.

Each project is built up layer by layer rather than as a fresh codebase per article. The **literature review agent** is the recurring example for *Building Agents That Work*: a flat tool-use loop in the early parts, then memory and retrieval, planning and orchestration, runtime primitives, safety, observability, evaluation, and uncertainty. The **SCIMA** agent backs *Ontology and Knowledge Graphs for Intelligent Agents*: one ontology and knowledge graph that grows from v0.1 onward. Each project's own README maps every module to the part or article that introduced it.

---

## Series Index

| Directory | Spans | Description |
|-----------|-------|-------------|
| [`lit_review_agent/`](./lit_review_agent/) | Parts 1–17 | An agent that searches Semantic Scholar, reads abstracts, and drafts a literature review. Grows across the series to include a vector store and property-graph memory, a plan/sub-agent orchestrator, skills and hooks, prompt-injection defenses, span-based tracing, model routing and cost rollups, an offline evaluation suite, progressive disclosure, and a confidence gate. |
| [`ontology_kg_for_agents/`](./ontology_kg_for_agents/) | Articles 1–2+ | **SCIMA**, the SmartCity Infrastructure Management Agent. One coherent ontology and knowledge graph that grows alongside the *Ontology and Knowledge Graphs for Intelligent Agents* series: OWL DL building blocks, versioned SCIMA-OWL files, triple population, named graphs, SPARQL, and geo queries, with tests that keep the published code honest. |

More projects will be added as the series continues.

---

## Getting Started

Each project is self-contained. Navigate into the project directory and follow the steps below.

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) for the live literature review agent. Its evaluation suite, retrieval eval, graph traversals, memory, and hooks all run offline with no key. The SCIMA ontology and knowledge graph project runs entirely offline.

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
```

### Running the SCIMA ontology / knowledge graph (no API key)

```bash
cd ontology_kg_for_agents

# Inspect the v0.1 ontology
python -m scima.ontology

# Populate the v0.2 knowledge graph and run the geo query
python -m scima.knowledge_graph --populate sample --query lights-near I-204

# Run the tests that pin each article's ontology to its Growth Tracker
pytest
```

---

## Repository Structure

```
.
├── lit_review_agent/       # The recurring example, built across Parts 1-17
│   ├── harness.py          # Control loop, system prompt, tool registry (incl. ask_user), tracing + hooks + provenance, run_worker
│   ├── orchestrator.py     # Plan -> parallel sub-agents -> synthesize -> reflect -> save, with model routing
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
│   │   ├── ontology.py         # Load + inspect versioned SCIMA-OWL files
│   │   ├── building_blocks.py  # Article 1: classes, individuals, properties, axioms
│   │   └── knowledge_graph.py  # Article 2: populate triples, named graphs, SPARQL, geo query
│   ├── ontologies/         # Canonical SCIMA-OWL, one Turtle file per version
│   ├── tests/              # Per-article tests pinned to the Growth Tracker
│   ├── requirements.txt
│   └── README.md           # Article-to-code map and design notes
└── README.md
```

---

## License

MIT
