"""
Progressive disclosure of agent actions (Part 16, Section 4).

The trace (Part 11) is the full record. A person does not want all of it, and
burying the one thing that matters under forty routine tool calls is lost-in-
the-middle (Parts 2, 6) aimed at human attention. So show the agent's work at
the altitude the user needs, with depth on request:

  level "summary" : one line, the default view.
  level "plan"    : the plan with the current step marked (the Part 7 todo is
                    the natural middle tier).
  level "trace"   : the full per-worker span waterfall, one click down.

This is the same instinct as Part 8 skills (only what is needed enters context),
mirrored at the human surface.
"""

from __future__ import annotations

from tracing import Run


def _worker_spans(run: Run):
    return [s for s in run.spans if s.name == "worker"]


def summary_line(run: Run) -> str:
    """The default altitude: one line a researcher can read at a glance."""
    workers = _worker_spans(run)
    papers = sum(1 for s in run.spans if s.name == "tool:fetch_paper")
    searches = sum(1 for s in run.spans if s.name == "tool:search_papers")
    synthesizing = any(s.name == "synthesize" for s in run.spans)
    state = "synthesizing now" if synthesizing else "gathering sources"
    n = len(workers) or searches
    return f"Surveying {n} subtopics · {papers} papers retrieved · {state}"


def plan_view(run: Run) -> list[str]:
    """The middle altitude: the plan, each step with its status."""
    lines: list[str] = []
    for i, s in enumerate(_worker_spans(run), 1):
        mark = {"ok": "[x]", "empty": "[!]", "error": "[x]"}.get(s.status, "[ ]")
        label = (s.input or {}).get("subgoal", f"subtopic {i}") if isinstance(s.input, dict) else f"subtopic {i}"
        note = "0 papers" if s.status == "empty" else ""
        lines.append(f"  {mark} {label} {note}".rstrip())
    return lines or ["  [ ] (no plan recorded)"]


def trace_view(run: Run) -> list[str]:
    """The deepest altitude: the full span waterfall, on demand."""
    lines = []
    for s in run.spans:
        flag = "" if s.status == "ok" else f"  <{s.status}>"
        lines.append(f"  {s.name:<24} {s.ms:>6}ms{flag}")
    return lines


def render(run: Run, level: str = "summary") -> str:
    """Render a run at one of three altitudes. Default is the one-line summary;
    `plan` and `trace` are the drill-downs."""
    if level == "summary":
        return summary_line(run)
    if level == "plan":
        return summary_line(run) + "\n" + "\n".join(plan_view(run))
    if level == "trace":
        return summary_line(run) + "\n" + "\n".join(trace_view(run))
    raise ValueError(f"unknown level: {level!r} (use summary | plan | trace)")


if __name__ == "__main__":
    from pathlib import Path

    from tracing import TRACE_DIR

    paths = sorted(Path(TRACE_DIR).glob("run_*.json")) if TRACE_DIR.exists() else []
    if not paths:
        print("No traces yet. Run orchestrator.py first.")
    else:
        run = Run.load(paths[-1])
        for level in ("summary", "plan", "trace"):
            print(f"\n=== level: {level} ===")
            print(render(run, level))
