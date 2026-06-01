import json
import httpx
from pathlib import Path

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,authors,year,citationCount,abstract,venue"
OUTPUT_DIR = Path(__file__).parent / "output"


def search_papers(query: str, year: str | None = None, sort_by: str = "relevance") -> str:
    """Search Semantic Scholar. Returns top 10 results as JSON."""
    params = {
        "query": query,
        "limit": 25,
        "fields": PAPER_FIELDS,
    }
    if year:
        params["year"] = year  # e.g. "2024" or "2023-2025"

    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params=params, timeout=20
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})

    # Sort client-side so the tool contract is reliable regardless of API behavior
    if sort_by == "citations":
        papers = sorted(
            papers, key=lambda p: p.get("citationCount") or 0, reverse=True
        )

    results = []
    for p in papers[:10]:
        results.append(
            {
                "id": p["paperId"],
                "title": p.get("title"),
                "authors": [a["name"] for a in (p.get("authors") or [])[:3]],
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract_snippet": (p.get("abstract") or "")[:300],
            }
        )

    return json.dumps({"count": len(results), "papers": results}, indent=2)


def fetch_paper(paper_id: str) -> str:
    """Fetch full metadata and abstract for one paper by Semantic Scholar ID."""
    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
            params={"fields": PAPER_FIELDS},
            timeout=20,
        )
        resp.raise_for_status()
        p = resp.json()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})

    return json.dumps(
        {
            "id": p.get("paperId"),
            "title": p.get("title"),
            "authors": [a["name"] for a in (p.get("authors") or [])],
            "year": p.get("year"),
            "venue": p.get("venue"),
            "citations": p.get("citationCount", 0),
            "abstract": p.get("abstract"),
        },
        indent=2,
    )


def save_to_file(filename: str, content: str) -> str:
    """Write content to a file inside the output/ directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    return f"Saved {len(content)} characters to {path}"


def done(summary: str) -> str:
    """Signal that the task is complete."""
    return f"DONE: {summary}"
