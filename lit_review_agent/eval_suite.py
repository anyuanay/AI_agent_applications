"""
Evaluation (Part 12).

The discipline is to decompose "is it good" into a stack of checks, push each as
far down the ladder as it goes, and read everything as pass-rates over a frozen
suite rather than a single verdict.

  rung 1  deterministic checks   (assertions, free, unforgiving)
  rung 2  reference metrics      (Recall@k against a gold set — Part 6)
  rung 3  LLM-as-judge           (open-ended quality, least trustworthy)

A failing production trace (Part 11) becomes a permanent, replayable EvalCase:
freeze the input, the tool behavior, and the grading criterion. The worker-B
case below stays red until the bug is fixed, then guards against its return.

Run offline:
    python eval_suite.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from tracing import Run, is_empty

FIXTURE_DIR = Path(__file__).parent / "eval_fixtures"
CITE = re.compile(r"\[S2:([A-Za-z0-9]+)\]")
CITE_LIKE = re.compile(r"\[S2:[^\]]*\]")  # any S2-shaped token, to catch malformed ones


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------
@dataclass
class Pass:
    pass


@dataclass
class Fail:
    reason: str


# ---------------------------------------------------------------------------
# Reading a run (the trace is the substrate evaluation runs on)
# ---------------------------------------------------------------------------
def _as_obj(output) -> object:
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    return output


def seen_paper_ids(run: Run) -> set[str]:
    """Every paper ID that actually appeared in a tool result during the run."""
    ids: set[str] = set()
    for s in run.tool_spans("search_papers"):
        data = _as_obj(s.output)
        if isinstance(data, dict):
            ids |= {p.get("id") for p in data.get("papers", []) if p.get("id")}
    for s in run.tool_spans("fetch_paper"):
        data = _as_obj(s.output)
        if isinstance(data, dict) and data.get("id"):
            ids.add(data["id"])
    return ids


def saved_content(run: Run) -> str:
    for s in run.tool_spans("save_to_file"):
        if isinstance(s.input, dict) and "content" in s.input:
            return s.input["content"]
    return ""


def cited_paper_ids(run: Run) -> set[str]:
    return set(CITE.findall(saved_content(run)))


def all_searches_empty(run: Run) -> bool:
    searches = run.tool_spans("search_papers")
    return bool(searches) and all(is_empty(s.output) for s in searches)


# ---------------------------------------------------------------------------
# Rung 1 — deterministic graders
# ---------------------------------------------------------------------------
def saved_to_output_dir(run: Run) -> bool:
    for s in run.tool_spans("save_to_file"):
        fn = (s.input or {}).get("filename", "") if isinstance(s.input, dict) else ""
        if fn and not fn.startswith("/") and ".." not in fn:
            return True
    return False


def all_citations_well_formed(run: Run) -> bool:
    content = saved_content(run)
    # Every S2-shaped token must match the strict id pattern.
    return all(CITE.fullmatch(tok) for tok in CITE_LIKE.findall(content))


def cites_only_seen_papers(run: Run) -> bool:
    """No invented citations: every cited id appeared in a tool result."""
    return cited_paper_ids(run) <= seen_paper_ids(run)


def worker_abstained_on_zero(run: Run) -> bool:
    """If every search came back empty, the agent must abstain, not improvise."""
    if not all_searches_empty(run):
        return True  # not applicable: some sources were found
    content = saved_content(run).lower()
    abstained = any(p in content for p in ("no sources", "no results", "omitted", "abstain"))
    return abstained and not cited_paper_ids(run)


# ---------------------------------------------------------------------------
# Rung 2 — reference metric (the retrieval slice from Part 6)
# ---------------------------------------------------------------------------
def recall_at_k(run: Run, gold_set: set[str] | None, k: int = 10) -> float:
    """Fraction of the gold papers the run actually retrieved. Skips if no gold."""
    if not gold_set:
        return 1.0
    return len(seen_paper_ids(run) & gold_set) / len(gold_set)


# ---------------------------------------------------------------------------
# Rung 3 — LLM-as-judge (rubric, not vibe; validate before trusting)
# ---------------------------------------------------------------------------
FAITHFULNESS_RUBRIC = """\
Score each criterion 0-2 (0 absent, 1 partial, 2 fully met).
faithfulness: every claim is traceable to a cited paper.
coverage: the major cited works are represented.
grounding: every [S2:id] citation appears in the provided source list.
Return ONLY JSON: {"scores": {...}, "evidence": {...}}.
Quote the supporting text for each score in `evidence`.
"""


@dataclass
class Verdict:
    scores: dict
    evidence: dict

    @property
    def min_score(self) -> int:
        return min(self.scores.values()) if self.scores else 0


def judge(review: str, rubric: str = FAITHFULNESS_RUBRIC) -> Verdict | None:
    """Grade open-ended quality. Returns None when no model is available, so the
    deterministic rungs still run offline. A score is a sample, not a measurement:
    validate the judge against human labels before trusting it at scale."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    from harness import MODEL

    resp = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=1024, system=rubric,
        messages=[{"role": "user", "content": review}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    try:
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
        return Verdict(scores=data.get("scores", {}), evidence=data.get("evidence", {}))
    except (ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# grade_run — graded cheapest-first (Part 12, Section 6)
# ---------------------------------------------------------------------------
def grade_run(run: Run, gold_set: set[str] | None = None):
    # rung 1: deterministic, free, and unforgiving
    if not saved_to_output_dir(run):
        return Fail("no file saved to output/")
    if not all_citations_well_formed(run):
        return Fail("malformed citation")
    if not cites_only_seen_papers(run):
        return Fail("invented citation")  # worker-B class
    if not worker_abstained_on_zero(run):
        return Fail("improvised a section with zero sources")

    # rung 2: reference metric on the retrieval slice (Part 6)
    if recall_at_k(run, gold_set) < 0.8:
        return Fail("missed required papers")

    # rung 3: the judge, last, on what is left
    verdict = judge(saved_content(run))
    if verdict is None:
        return Pass()  # no judge available offline; deterministic rungs passed
    return Pass() if verdict.min_score >= 1 else Fail(json.dumps(verdict.evidence))


# ---------------------------------------------------------------------------
# EvalCase — a frozen, replayable case (Part 12, Section 5)
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    name: str
    goal: str
    grade: object  # callable: Run -> bool
    mock_tools: dict = field(default_factory=dict)
    fixture: str | None = None  # a recorded trace to replay offline

    def run_replay(self) -> Run:
        """Grade a recorded trace — deterministic, offline, no model needed."""
        return Run.load(FIXTURE_DIR / self.fixture)

    def run_live(self) -> Run:
        """Run the agent with mocked tools and a real model, capturing the trace.
        Needs ANTHROPIC_API_KEY and network. Used to generate new fixtures."""
        import harness

        def _as_str_tool(fn):
            def wrapped(**kw):
                out = fn(**kw)
                return out if isinstance(out, str) else json.dumps(out)
            return wrapped

        original = dict(harness.TOOL_DISPATCH)
        for name, fn in self.mock_tools.items():
            harness.TOOL_DISPATCH[name] = _as_str_tool(fn)
        try:
            _, tracer = harness.run_loop(self.goal, system=harness.SYSTEM_PROMPT,
                                         max_turns=12, verbose=False)
            return tracer.to_run()
        finally:
            harness.TOOL_DISPATCH.clear()
            harness.TOOL_DISPATCH.update(original)


# The worker-B failure, frozen into a case (Part 12, Section 5).
WORKER_B_CASE = EvalCase(
    name="empty_result_subtopic_must_abstain",
    goal="Survey: graph attention networks, 2025",
    mock_tools={
        # reproduce the swallowed rate-limit deterministically
        "search_papers": lambda **kw: {"hits": 0, "status": "empty"},
    },
    grade=lambda run: (
        cites_only_seen_papers(run)        # no invented citations
        and worker_abstained_on_zero(run)  # did not improvise a section
    ),
    fixture=None,  # set per-fixture below when replaying
)


# ---------------------------------------------------------------------------
# Offline demonstration: the same case red on the failing trace, green on the fix.
# ---------------------------------------------------------------------------
def _demo() -> None:
    print("Failure-driven eval: the worker-B case (Parts 11-12)\n" + "─" * 60)
    for fixture, label in [("worker_b_failing.json", "before the fix"),
                           ("worker_b_fixed.json", "after the fix")]:
        run = Run.load(FIXTURE_DIR / fixture)
        ok = WORKER_B_CASE.grade(run)
        full = grade_run(run)
        verdict = "PASS" if ok else "FAIL"
        full_v = "Pass" if isinstance(full, Pass) else f"Fail({full.reason})"
        print(f"  [{label:<15}] worker-B grade: {verdict:<4}   grade_run: {full_v}")
    print("\nThe case is red until the two Part 11 fixes land, then a permanent guard.")


if __name__ == "__main__":
    _demo()
