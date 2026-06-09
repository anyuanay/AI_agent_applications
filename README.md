# Generative AI Agents and Applications

Companion code for the **Generative AI Agents and Applications** series published on [Medium](https://medium.com/@anyuanay).

Each directory in this repository contains full, runnable source code discussed in the series.

The **literature review agent** is the recurring example. Rather than a fresh project per article, it is built up layer by layer: a flat tool-use loop in the early parts, then memory and retrieval, planning and orchestration, runtime primitives, safety, observability, and evaluation. The single `lit_review_agent/` codebase reflects all of that, and its own [README](./lit_review_agent/README.md) maps every module to the part that introduced it.

---

## Series Index

| Directory | Spans | Description |
|-----------|-------|-------------|
| [`lit_review_agent/`](./lit_review_agent/) | Parts 1–12 | An agent that searches Semantic Scholar, reads abstracts, and drafts a literature review. Grows across the series to include a vector store and property-graph memory, a plan/sub-agent orchestrator, skills and hooks, prompt-injection defenses, span-based tracing, and an offline evaluation suite. |

More projects will be added as the series continues.

---

## Getting Started

Each project is self-contained. Navigate into the project directory and follow the steps below.

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) for the live agent. The evaluation suite, retrieval eval, graph traversals, memory, and hooks all run offline with no key.

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
python eval_suite.py       # the failure-driven eval case, red on a failing trace, green on the fix
python retrieval_eval.py   # the Recall@k retrieval slice
```

---

## Repository Structure

```
.
├── lit_review_agent/       # The recurring example, built across Parts 1-12
│   ├── harness.py          # Control loop, system prompt, tool registry, tracing + hooks + provenance, run_worker
│   ├── orchestrator.py     # Plan -> parallel sub-agents -> synthesize -> reflect -> save
│   ├── tools.py            # search / fetch / save / done, structured errors, untrusted envelope, graph ingestion
│   ├── tools_server.py     # The same tools as an optional MCP server
│   ├── memory.py           # PaperMemory (remember / recall) and trim_for_window
│   ├── vector_store.py     # A tiny persistent cosine store
│   ├── embeddings.py       # One embed function, same model both sides (a stand-in)
│   ├── textutil.py         # chunk and reorder_for_window (lost-in-the-middle)
│   ├── graph.py            # Paper/Author/Venue/Topic property graph
│   ├── graph_tools.py      # Typed traversals, query_graph passthrough, hybrid_recall
│   ├── hooks.py            # guard_file_writes, enforce_budget, log_call (Allow / Block)
│   ├── skills.py           # Progressive-disclosure skill loader
│   ├── skills/             # systematic_review SKILL.md + checklist
│   ├── tracing.py          # Span, Tracer, span(), and the Run artifact
│   ├── eval_suite.py       # Grading ladder, grade_run, LLM judge, replayable EvalCase
│   ├── retrieval_eval.py   # Standalone Recall@k retrieval slice
│   ├── eval_fixtures/      # Recorded traces: the worker-B failure and its fix
│   ├── requirements.txt
│   ├── README.md           # Module-to-part map and notes on the stand-ins
│   └── output/             # Generated reviews (git-ignored)
└── README.md
```

---

## License

MIT
