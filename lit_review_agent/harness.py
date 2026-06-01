"""
Literature review agent — harness.

Runs the agent loop: post prompt → parse tool calls → dispatch → feed results back.
Terminates when the model calls `done` or max_turns is reached.

Usage:
    python harness.py "Find the three most-cited papers on graph neural networks
                       published in 2024 and draft a literature review."
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

from tools import done, fetch_paper, save_to_file, search_papers

# ---------------------------------------------------------------------------
# System prompt — the policy layer that shapes every turn of the agent's behavior.
# Keep it short and specific. Vague instructions produce vague behavior.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a research assistant. Your job is to find relevant academic papers,
read their abstracts, and synthesize a concise literature review section.

Rules:
- Always cite findings with the paper's title, authors, and year.
- If a search returns no useful results, refine your query and try again.
- Prefer papers with high citation counts when looking for influential work.
- Save the final draft to a .md file, then call `done` to finish.
"""

# ---------------------------------------------------------------------------
# Tool schemas — what the model knows it can call.
# Description quality directly determines how reliably the model calls each tool.
# See Part 5 of the series for the full story on why this matters.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "search_papers",
        "description": (
            "Search Semantic Scholar for academic papers. "
            "Use specific queries rather than broad topics. "
            "Set sort_by='citations' when you want the most influential work. "
            "Pass year as '2024' or '2023-2024' to filter by publication year."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific.",
                },
                "year": {
                    "type": "string",
                    "description": "Year or year range, e.g. '2024' or '2023-2024'. Omit for no filter.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "citations"],
                    "description": "Sort order. Use 'citations' for most-cited papers.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_paper",
        "description": (
            "Fetch the full abstract and metadata for a specific paper "
            "using its Semantic Scholar paper ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Semantic Scholar paper ID from a search result.",
                },
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "save_to_file",
        "description": "Write the final literature review to a file in the output/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename, e.g. 'lit_review.md'.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content.",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that the task is complete. "
            "Call this only after the draft has been saved to a file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
]

TOOL_DISPATCH = {
    "search_papers": search_papers,
    "fetch_paper": fetch_paper,
    "save_to_file": save_to_file,
    "done": done,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(goal: str, max_turns: int = 20) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": goal}]

    print(f"Goal: {goal}\n{'─' * 60}")

    for turn in range(max_turns):
        print(f"\n[Turn {turn + 1}]")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Print any text the model emits before or between tool calls
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                print(f"Model: {block.text.strip()}")

        # Append assistant turn to conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Natural end-of-turn (no tool calls)
        if response.stop_reason == "end_turn":
            print("\nAgent stopped (end_turn).")
            break

        # Dispatch tool calls and collect results
        if response.stop_reason == "tool_use":
            tool_results = []
            finished = False

            for block in response.content:
                if block.type != "tool_use":
                    continue

                args_display = json.dumps(block.input)
                print(f"  → {block.name}({args_display})")

                result = TOOL_DISPATCH[block.name](**block.input)

                # Truncate long results for console readability only
                display = result if len(result) < 400 else result[:400] + "…"
                print(f"  ← {display}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

                if block.name == "done":
                    finished = True

            messages.append({"role": "user", "content": tool_results})

            if finished:
                print("\nTask complete.")
                return

    else:
        print(f"\nReached max_turns ({max_turns}) without a `done` call.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Literature review agent")
    parser.add_argument(
        "goal",
        nargs="?",
        default=(
            "Find the three most-cited papers on graph neural networks "
            "published in 2024 and draft a literature review section. "
            "Save the review to 'gnn_lit_review.md'."
        ),
        help="Goal for the agent (quoted string)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Hard cap on agent turns (default: 20)",
    )
    args = parser.parse_args()
    run_agent(args.goal, max_turns=args.max_turns)
