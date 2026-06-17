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
# the flat single-agent loop (Parts 1-6, 10, 11, 16, 17)
python harness.py "Find the three most-cited papers on graph neural networks
                   published in 2024 and draft a literature review section."

# the orchestrated survey: plan -> parallel sub-agents -> synthesize -> reflect,
# now with model routing (Part 13) and progressive-disclosure output (Part 16)
python orchestrator.py "Survey graph neural networks across application areas."

# evaluation, fully offline: the worker-B abstain case (Parts 11-12), the
# wrong-year parser regression (Part 15), and the confidence gate (Part 17)
python eval_suite.py

# the Part 6 retrieval slice, offline
python retrieval_eval.py

# the confidence gate on its own (Part 17), offline
python uncertainty.py
```

## How the modules map to the series

| Module | Part | What it is |
| --- | --- | --- |
| `harness.py` | 1, 4, 7, 10, 11, 16, 17 | the control loop, system prompt + capability scope, tool registry (incl. `ask_user`), tracing + hooks (incl. the approval gate) + provenance wiring, `run_worker` sub-agent |
| `tools.py` | 5, 6, 10, 15, 17 | `search_papers` / `fetch_paper` / `save_to_file` / `done` / `ask_user`; structured rate-limit errors; the untrusted-document envelope; graph ingestion; `parse_year_range` (the Part 15 holistic fix) |
| `tools_server.py` | 5 | the same four tools as an optional MCP server (expose once, use everywhere) |
| `embeddings.py` | 6 | one embed function, same model both sides (a dependency-free stand-in) |
| `vector_store.py` | 6 | a tiny persistent cosine store |
| `textutil.py` | 2, 6 | `chunk` and `reorder_for_window` (lost-in-the-middle made operational) |
| `memory.py` | 6 | `PaperMemory` (`remember` / `recall`) and `trim_for_window` |
| `graph.py` | 6, 14, 17 | the Paper/Author/Venue/Topic property graph; CITES/WROTE/PUBLISHED_IN/ON_TOPIC edges; `has_edge` (the structural uncertainty detector) |
| `graph_tools.py` | 5, 6 | typed traversals (preferred) vs the `query_graph` passthrough; `hybrid_recall` GraphRAG |
| `hooks.py` | 8, 10, 1, 16 | `guard_file_writes`, `enforce_budget`, `require_approval` (human-in-the-loop gate), `log_call`; the Allow/Block contract |
| `skills.py` + `skills/` | 8 | progressive-disclosure skill loader and the `systematic_review` SKILL.md |
| `orchestrator.py` | 7, 8, 9, 13, 16 | the four layers assembled as a hierarchy (Part 9's verdict: not a peer mesh); model routing + progressive-disclosure output |
| `production.py` | 13 | model routing (`route`), version pinning (`pin_versions`), and the cost rollup (`cost_rollup`, the span tree is a bill) |
| `uncertainty.py` | 16, 17 | the `assert_citation` confidence gate, detection signals ranked by trust, provenance-tagged `Claim`, and the resolve/ask/hedge/abstain `response_policy` |
| `disclosure.py` | 16 | progressive disclosure of a run at three altitudes (summary / plan / trace) |
| `tracing.py` | 11 | `Span`, `Tracer`, the `span()` context manager, and the `Run` artifact downstream reads |
| `eval_suite.py` | 12, 15, 17 | the ladder of graders, `grade_run` (cheapest-first), the LLM judge, the replayable worker-B `EvalCase`, the Part 15 `parser_regression`, and the Part 17 `confidence_gate_cases` |
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
- **Model routing** (`production.route`) names a model per stage; swapping the
  three model ids is the only change needed to retune the cost/quality tradeoff.
  The cost rollup uses approximate per-1k prices to make the routing win concrete.
- **The approval gate** (`hooks.require_approval`) defaults to an auto-approver so
  unattended runs and CI proceed; pass `console_approver` (or a UI callback) for
  an interactive, human-in-the-loop run. It fires only on consequential actions
  (writes outside `output/`, overwrites, send/delete-class tools).
- **`ask_user`** uses a console responder when a TTY is present and otherwise
  abstains, so the uncertainty path is exercised without blocking automation.
