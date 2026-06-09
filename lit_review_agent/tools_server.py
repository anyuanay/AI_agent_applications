"""
The lit-review tools as an MCP server (Part 5, Section 6).

Optional. This is the "expose once, use everywhere" shape: the same four tools,
moved from functions embedded in the harness to a standalone MCP server any
agent that speaks the protocol can call. The harness gets shorter; the tools
become independently versioned and testable.

Run:  python tools_server.py        (starts the stdio server)
Needs the optional `mcp` package:  pip install mcp
"""

from __future__ import annotations

import tools  # the same implementations the harness uses

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The MCP server is optional. Install it with: pip install mcp"
    ) from e

mcp = FastMCP("lit-review-tools")


@mcp.tool()
def search_papers(query: str, year: str | None = None, sort_by: str = "relevance") -> str:
    """Search Semantic Scholar for academic papers.

    Use specific queries rather than broad topics.
    Set sort_by='citations' when you want the most influential work.
    Pass year as '2024' or '2023-2024' to filter by publication year.
    """
    return tools.search_papers(query, year=year, sort_by=sort_by)


@mcp.tool()
def fetch_paper(paper_id: str) -> str:
    """Fetch full metadata and abstract for a paper by Semantic Scholar ID."""
    return tools.fetch_paper(paper_id)


@mcp.tool()
def save_to_file(filename: str, content: str) -> str:
    """Write the final literature review to the output/ directory."""
    return tools.save_to_file(filename, content)


@mcp.tool()
def done(summary: str) -> str:
    """Signal that the task is complete. Call only after saving the draft."""
    return tools.done(summary)


if __name__ == "__main__":
    mcp.run()  # starts the stdio server
