# Literature review agent

The recurring example from the *Building Agents That Work* series, built up
layer by layer across the articles. It is a model in a control loop (Part 1)
that searches Semantic Scholar, reads abstracts, and drafts a literature review,
with the engineering disciplines from each part wired in.

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # needed for the live agent
```

The live agent and orchestrator call the Anthropic API and the Semantic Scholar
API. The evaluation suite, retrieval eval, graph traversals, memory, and hooks
all run **offline** with no key.

## Run it

```bash
# the flat single-agent loop (Parts 1-6, 10, 11)
python harness.py "Find the three most-cited papers on graph neural networks
                   published in 2024 and draft a literature review section."

# the orchestrated survey: plan -> parallel sub-agents -> synthesize -> reflect
python orchestrator.py "Survey graph neural networks across application areas."

# evaluation, fully offline: the worker-B case red on the failing trace, green on the fix
python eval_suite.py

# the Part 6 retrieval slice, offline
python retrieval_eval.py
```

## How the modules map to the series

| Module | Part | What it is |
| --- | --- | --- |
| `harness.py` | 1, 4, 7, 10, 11 | the control loop, system prompt, tool registry, tracing + hooks + provenance wiring, `run_worker` sub-agent |
| `tools.py` | 5, 6, 10 | `search_papers` / `fetch_paper` / `save_to_file` / `done`; structured rate-limit errors; the untrusted-document envelope; graph ingestion |
| `tools_server.py` | 5 | the same four tools as an optional MCP server (expose once, use everywhere) |
| `embeddings.py` | 6 | one embed function, same model both sides (a dependency-free stand-in) |
| `vector_store.py` | 6 | a tiny persistent cosine store |
| `textutil.py` | 2, 6 | `chunk` and `reorder_for_window` (lost-in-the-middle made operational) |
| `memory.py` | 6 | `PaperMemory` (`remember` / `recall`) and `trim_for_window` |
| `graph.py` | 6, 14 | the Paper/Author/Venue/Topic property graph and CITES/WROTE/PUBLISHED_IN/ON_TOPIC edges |
| `graph_tools.py` | 5, 6 | typed traversals (preferred) vs the `query_graph` passthrough; `hybrid_recall` GraphRAG |
| `hooks.py` | 8, 10, 1 | `guard_file_writes`, `enforce_budget`, `log_call`; the Allow/Block contract |
| `skills.py` + `skills/` | 8 | progressive-disclosure skill loader and the `systematic_review` SKILL.md |
| `orchestrator.py` | 7, 8, 9 | the four layers assembled as a hierarchy (Part 9's verdict: not a peer mesh) |
| `tracing.py` | 11 | `Span`, `Tracer`, the `span()` context manager, and the `Run` artifact downstream reads |
| `eval_suite.py` | 12 | the ladder of graders, `grade_run` (cheapest-first), the LLM judge, and the replayable worker-B `EvalCase` |
| `retrieval_eval.py` | 6, 12 | the standalone Recall@k retrieval slice |
| `eval_fixtures/` | 11, 12 | recorded traces: the worker-B failure frozen into a case, and its fix |

## Notes on the stand-ins

A few pieces are deliberately self-contained so the codebase runs without heavy
infrastructure, while keeping the *interface* the articles describe:

- **Embeddings** are a deterministic hashed bag-of-words. Swap `embed` for a
  real model; nothing above it changes. The principle that matters is *one model
  both sides*.
- **The graph** is in-memory. In production this is Neo4j; `query_graph` is where
  a Cypher passthrough lands, which is why it returns a structured "unsupported"
  result here rather than pretending. The typed traversals are fully real.
- **The MCP server** is optional (`pip install mcp`).
- **Evaluation** grades recorded traces offline; the LLM judge is skipped when no
  API key is present, so the deterministic rungs still run in CI.
