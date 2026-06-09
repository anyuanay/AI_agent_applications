"""
Orchestrator — the four layers assembled (Parts 7 and 8).

    plan  ->  parallel sub-agent workers  ->  synthesize  ->  reflect  ->  save

Each worker (harness.run_worker) is a context firewall: its own window, a
narrower role, fewer tools, and only a summary crosses back. The synthesis step
loads the `systematic_review` skill on demand. The save passes through the same
guard_file_writes hook the harness installs.

This is the hierarchy from Part 8, not a peer mesh — Part 9's verdict was that
the lit-review job has one owner, one clock, and one window-able orchestrator,
so peers would only add coordination tax.

Usage:
    python orchestrator.py "Survey graph neural networks across application areas."
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

from harness import MODEL, run_worker
from skills import active_skill, skill
from tools import save_to_file
from tracing import Tracer, span

_client = anthropic.Anthropic()


def _ask(system: str, user: str, max_tokens: int = 2048) -> str:
    """One-shot model call, no tools — used for plan/synthesize/reflect."""
    resp = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def make_plan(goal: str) -> list[dict]:
    """Decompose the goal into subfield sub-goals plus a synthesis step (Part 7 §4)."""
    raw = _ask(
        "You are a planning assistant. Decompose a survey goal into 2-4 distinct "
        "subfields, then a final synthesis step. Return ONLY JSON of the form "
        '[{"subgoal": "...", "kind": "subfield"}, {"subgoal": "synthesize", "kind": "synthesis"}].',
        goal,
    )
    try:
        plan = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        # Fall back to a single subfield so the pipeline still runs.
        plan = [{"subgoal": goal, "kind": "subfield"},
                {"subgoal": "synthesize", "kind": "synthesis"}]
    return plan


def run_in_parallel(fn, items):
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(items)))) as pool:
        return list(pool.map(fn, items))


def synthesize(summaries: list[str]) -> str:
    """Merge worker summaries into one review (Part 7 §5 sequential merge)."""
    procedure = active_skill()
    system = "You synthesize per-subtopic summaries into one coherent literature review."
    if procedure:  # the loaded skill's procedure, only present inside the with-block
        system += "\n\nFollow this procedure:\n" + procedure
    body = "\n\n".join(f"## Subtopic summary {i+1}\n{s}" for i, s in enumerate(summaries))
    return _ask(system, "Synthesize these into one review, preserving every [S2:id] tag:\n\n" + body)


def reflect_and_revise(draft: str, summaries: list[str], max_passes: int = 2) -> str:
    """Bounded generate-critique-revise (Part 7 §3). Critique is grounded in the
    source summaries, not in the model's unaided opinion."""
    review = draft
    sources = "\n\n".join(summaries)
    for _ in range(max_passes):
        critique = _ask(
            "You are a critical reviewer. List concrete, verifiable problems with the "
            "draft, grounded ONLY in the provided sources: unsupported claims, missing "
            "[S2:id] tags, citations not present in the sources. If none, reply 'OK'.",
            f"SOURCES:\n{sources}\n\nDRAFT:\n{review}",
        )
        if critique.strip().upper().startswith("OK"):
            break
        review = _ask(
            "Revise the draft to fix exactly the listed problems. Do not add new "
            "citations that are not in the sources. Return the full revised review.",
            f"CRITIQUE:\n{critique}\n\nDRAFT:\n{review}",
        )
    return review


def run_survey(goal: str, filename: str = "survey.md") -> str:
    """The four layers, with runtime primitives (Part 8 Figure 4)."""
    tracer = Tracer(goal=goal)
    print(f"Survey goal: {goal}\n{'─' * 60}")

    with span(tracer, "plan", parent=tracer.run_id, input={"goal": goal}) as ps:
        plan = make_plan(goal)
        ps.output = plan
    subfields = [s for s in plan if s["kind"] == "subfield"]
    print(f"Plan: {len(subfields)} subfields")

    # SUB-AGENTS: each worker is a context firewall (Part 7's promise, built).
    def _work(s):
        with span(tracer, "worker", parent=tracer.run_id, input=s) as ws:
            out = run_worker(s["subgoal"], tracer=tracer, parent=ws.span_id)
            ws.output = out
            ws.status = "empty" if not out.strip() else "ok"
            return out

    summaries = run_in_parallel(_work, subfields)

    # SKILL: the formal-review procedure loads only now, on demand.
    with skill("systematic_review"):
        with span(tracer, "synthesize", parent=tracer.run_id) as ss:
            draft = synthesize(summaries)
            ss.output = draft[:200]
        with span(tracer, "reflect", parent=tracer.run_id) as rs:
            review = reflect_and_revise(draft, summaries, max_passes=2)
            rs.output = review[:200]

    # HOOK: guard_file_writes guards this in the full harness path.
    with span(tracer, "tool:save_to_file", parent=tracer.run_id,
              input={"filename": filename}) as fs:
        fs.output = save_to_file(filename, review)
    tracer.save()
    print(f"\nSaved survey to output/{filename}")
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrated literature survey")
    parser.add_argument("goal", nargs="?",
                        default="Survey graph neural networks across application areas.")
    parser.add_argument("--filename", default="survey.md")
    args = parser.parse_args()
    run_survey(args.goal, filename=args.filename)
