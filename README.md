# Generative AI Agents and Applications

Companion code for the **Generative AI Agents and Applications** series published on [Medium](https://medium.com/@anyuanay).

Each directory in this repository corresponds to a story in the series and contains the full, runnable source code discussed in the article.

---

## Series Index

| Directory | Article | Description |
|-----------|---------|-------------|
| [`lit_review_agent/`](./lit_review_agent/) | *Building a Literature Review Agent with Claude and Tool Use* | An agentic loop that searches Semantic Scholar, fetches paper abstracts, and drafts a Markdown literature review — driven by Claude's tool-use capability. |

More projects will be added as the series continues.

---

## Getting Started

Each project is self-contained. Navigate into the project directory and follow the steps below.

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
cd <project-directory>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

### Running the literature review agent

```bash
# Default goal (GNN papers, 2024)
python harness.py

# Custom goal
python harness.py "Find the five most-cited papers on RAG published in 2023-2024 and draft a literature review."

# Limit agent turns
python harness.py "..." --max-turns 10
```

Output is saved to `lit_review_agent/output/`.

---

## Repository Structure

```
.
├── lit_review_agent/       # Article 1 — Literature Review Agent
│   ├── harness.py          # Agent loop (prompt → tool dispatch → loop)
│   ├── tools.py            # Tool implementations (search, fetch, save, done)
│   ├── requirements.txt
│   └── output/             # Generated reviews (git-ignored)
└── README.md
```

---

## License

MIT
