"""
Loop engineering — the outer loops (Part 18).

Parts 1-17 engineered the inside of one control loop: one agent, one run, one
task, started when a human types and ended when it runs out of turns. Part 18
zooms out one altitude. You stop prompting the agent turn by turn and design the
loops that prompt it; the leverage moves from prompt quality to loop
architecture. Nothing here is new machinery. It is the series' own primitives
reused at the outer-loop altitude:

  - sub-agents (Parts 7/8)        -> run_subagent: the maker and the verifier
  - the confidence gate (Part 17) -> the verifier's per-citation check
  - the property graph (Part 14)  -> the ground truth the verifier checks against
  - connectors / MCP (Parts 5/8)  -> the digest and the human inbox (stand-ins)
  - the state file (Part 6)       -> external memory so a run skips finished work
  - traces (Part 11)              -> the improvement loop's input
  - the learning loop (Part 15)   -> tune_triage, at the operated-system altitude

This module is the `loop.py` from the Part 18 article, made runnable. The
`morning_run` body reads like the article's sketch line for line:

    candidates = arxiv.new_today(topics=tracked_topics())   # trigger
    for paper in candidates:
        if state.already_surveyed(paper): continue          # state file
        with worktree(paper):                               # isolated, parallel-safe
            draft  = run_subagent("summarize", paper=paper) # the maker
            review = run_subagent("verify", draft=draft)    # the checker
            if review.all_claims_grounded:
                connectors.post_digest(draft)
            else:
                connectors.to_human_inbox(draft, review.flags)
        state.record(paper)
    tune_triage(traces.since("yesterday"))                  # improvement

Verification is load-bearing because stacking compounds whatever the inner loop
produces, including its errors (Part 2's determinism-is-a-fiction). An
unattended loop that grades its own work compounds confident-wrong output at
machine speed, so the verifier checks against ground truth, and the digest is
gated on that check — never on the maker saying it is done.

Usage:
    python loop.py            # two mornings, fully offline (stub maker)
    python loop.py --live     # use the live run_subagent makers (needs an API key)
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from graph import GRAPH
from tracing import TRACE_DIR, Run, Span, Tracer, span
from uncertainty import STATE, assert_citation

OUTPUT_DIR = Path(__file__).parent / "output"
STATE_PATH = OUTPUT_DIR / "loop_state.json"
LOOP_GOAL = "morning lit-review loop"

_CITE = re.compile(r"\[S2:([^\]]+)\]")  # the [S2:<id>] tag the system prompt mandates
USE_LIVE_MAKER = False                   # flipped by morning_run(live=True)

# The triage rule the trigger runs from. tune_triage reaches back in and edits
# `drop_topics` from production feedback (Part 15's learning loop, outer altitude).
DEFAULT_STATE: dict = {
    "seen": [],
    "triage": {
        "min_year": 2023,
        "tracked_topics": ["graph neural networks", "retrieval"],
        "drop_topics": [],
    },
}

# ---------------------------------------------------------------------------
# The feed the trigger reads (offline stand-in). In production this is an arXiv
# query, an RSS poll, or a webhook; here it is a fixed list so the loop's
# architecture runs without a network. Each paper carries the Semantic Scholar
# shape graph.add_paper ingests. `_planted` is a citation the maker will invent
# but the graph cannot back — the verifier must catch it (the article's planted
# hallucination made literal).
# ---------------------------------------------------------------------------
FEED: list[dict] = [
    {"id": "P1", "title": "Graph Transformers at Scale", "year": 2024,
     "topics": ["graph neural networks"], "references": ["R1", "R2"]},
    {"id": "P2", "title": "Retrieval-Augmented GNNs", "year": 2024,
     "topics": ["graph neural networks", "retrieval"], "references": ["R2", "R3"],
     "_planted": "R9"},  # P2 will cite R9, which is not in its references
    {"id": "P3", "title": "A Survey of Graph Methods", "year": 2019,
     "topics": ["graph neural networks"], "references": ["R1"]},  # stale: < min_year
    {"id": "P4", "title": "Quantum Knitting Patterns", "year": 2024,
     "topics": ["textiles"], "references": ["R7"]},  # off-topic
]


# ---------------------------------------------------------------------------
# Drafts and reviews — the objects that cross between maker and checker.
# ---------------------------------------------------------------------------
@dataclass
class Draft:
    paper_id: str
    title: str
    text: str


@dataclass
class Review:
    all_claims_grounded: bool
    flags: list[str] = field(default_factory=list)     # citations the graph cannot back
    verified: list[str] = field(default_factory=list)  # citations confirmed against the graph


# ---------------------------------------------------------------------------
# The state file (Part 6): external memory between runs, so the trigger does not
# re-survey papers it already handled and the tuned triage rule persists.
# ---------------------------------------------------------------------------
class StateFile:
    def __init__(self, path: Path = STATE_PATH) -> None:
        self.path = path
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            data = json.loads(json.dumps(DEFAULT_STATE))  # deep copy of the default
        self.seen: list[str] = list(data["seen"])
        self.triage: dict = data["triage"]

    def reset(self) -> None:
        self.seen = []
        self.triage = json.loads(json.dumps(DEFAULT_STATE["triage"]))
        self._save()

    def already_surveyed(self, paper: dict) -> bool:
        return paper["id"] in self.seen

    def record(self, paper: dict) -> None:
        if paper["id"] not in self.seen:
            self.seen.append(paper["id"])
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(
            json.dumps({"seen": self.seen, "triage": self.triage}, indent=2),
            encoding="utf-8",
        )


state = StateFile()


def tracked_topics() -> list[str]:
    """The topics the trigger watches: tracked minus whatever tune_triage dropped.
    Dropping a topic here is how the improvement loop sharpens tomorrow's feed."""
    t = state.triage
    return [topic for topic in t["tracked_topics"] if topic not in t["drop_topics"]]


# ---------------------------------------------------------------------------
# The trigger (event-driven loop). `arxiv.new_today` is the heartbeat: an event,
# not a prompt. Offline it reads a queue; in production an arXiv/RSS/webhook poll.
# ---------------------------------------------------------------------------
class _Arxiv:
    def __init__(self) -> None:
        self.queue: list[dict] = list(FEED)

    def new_today(self, topics: list[str]) -> list[dict]:
        """Today's papers on the watched topics, fresh enough to be worth a look."""
        wanted = set(topics)
        floor = state.triage["min_year"]
        return [
            p for p in self.queue
            if (set(p.get("topics", [])) & wanted) and (p.get("year") or 0) >= floor
        ]


arxiv = _Arxiv()


@contextmanager
def worktree(paper: dict):
    """Isolation stand-in (Part 7 parallelism / the worktree primitive). In
    production a git worktree per paper so concurrent makers never collide; here
    a no-op boundary that keeps the sketch's shape honest."""
    yield


# ---------------------------------------------------------------------------
# Sub-agents (the agent loop and the verification loop). run_subagent dispatches
# by role: "summarize" is the maker, "verify" is the separate checker. The
# checker grounds itself in the Part 14 graph through the Part 17 gate, never in
# the maker's own narration.
# ---------------------------------------------------------------------------
def run_subagent(role: str, *, paper: dict | None = None, draft: Draft | None = None):
    if role == "summarize":
        return _summarize(paper)
    if role == "verify":
        return _verify(draft)
    raise ValueError(f"unknown sub-agent role: {role!r}")


def _summarize(paper: dict) -> Draft:
    """The maker. Offline it emits a deterministic draft whose [S2:id] citations
    are the paper's real references plus, where planted, one invented id. Live it
    is a real sub-agent (harness.run_worker), a context firewall per paper."""
    if USE_LIVE_MAKER:
        from harness import run_worker

        text = run_worker(
            f"Summarize the key contribution of '{paper['title']}' "
            f"(Semantic Scholar id {paper['id']}) in two sentences, citing sources [S2:<id>]."
        )
    else:
        cited = list(paper.get("references", []))
        if paper.get("_planted"):
            cited.append(paper["_planted"])
        tags = " ".join(f"[S2:{c}]" for c in cited)
        text = (f"{paper['title']} ({paper['year']}) advances the field, building on "
                f"prior work {tags}. [S2:{paper['id']}]")
    return Draft(paper_id=paper["id"], title=paper["title"], text=text)


def _verify(draft: Draft) -> Review:
    """The checker. Every [S2:id] citation is checked against the graph with the
    Part 17 confidence gate. A citation the graph cannot back is flagged, not
    asserted; the draft is grounded only if nothing is flagged."""
    verified, flagged = [], []
    for cited in _CITE.findall(draft.text):
        if cited == draft.paper_id:
            continue  # the self-tag is the paper labelling itself, not a citation
        decision = assert_citation(draft.paper_id, cited)  # Part 17 gate over the Part 14 graph
        (verified if decision["action"] == STATE else flagged).append(cited)
    return Review(all_claims_grounded=not flagged, flags=flagged, verified=verified)


# ---------------------------------------------------------------------------
# Connectors (Parts 5/8, stand-ins). Verified drafts go to the digest; anything
# with a flagged citation is routed to a human inbox (Part 17's unverified ->
# escalate). In production these are MCP calls to a channel and an issue tracker.
# ---------------------------------------------------------------------------
class _Connectors:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.digest: list[Draft] = []
        self.inbox: list[tuple[Draft, list[str]]] = []

    def post_digest(self, draft: Draft) -> None:
        self.digest.append(draft)

    def to_human_inbox(self, draft: Draft, flags: list[str]) -> None:
        self.inbox.append((draft, flags))

    def write(self, day: str) -> tuple[Path, Path]:
        """Flush the run's connector activity to files (the stand-in for posting)."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        dpath = OUTPUT_DIR / f"digest_{day}.md"
        ipath = OUTPUT_DIR / f"inbox_{day}.md"
        dlines = [f"# Verified digest — {day}", ""]
        dlines += [f"- **{d.title}** ({d.paper_id}): citations checked against the graph."
                   for d in self.digest]
        ilines = [f"# Needs review — {day}", ""]
        ilines += [f"- **{d.title}** ({d.paper_id}): unverified citations {flags} — "
                   "do not publish as fact." for d, flags in self.inbox]
        dpath.write_text("\n".join(dlines) + "\n", encoding="utf-8")
        ipath.write_text("\n".join(ilines) + "\n", encoding="utf-8")
        return dpath, ipath


connectors = _Connectors()


# ---------------------------------------------------------------------------
# Traces (Part 11) are the improvement loop's input. `traces.since` returns the
# loop's prior runs; the researcher's kept/deleted feedback is recorded onto
# those traces, so the signal tune_triage learns from lives in the traces.
# ---------------------------------------------------------------------------
class _Traces:
    def since(self, _when: str) -> list[Run]:
        """Prior loop runs. The `_when` window ('yesterday') is honoured loosely:
        the stand-in returns every recorded loop run rather than filtering by date."""
        if not TRACE_DIR.exists():
            return []
        runs = []
        for path in sorted(TRACE_DIR.glob("run_*.json")):
            try:
                run = Run.load(path)
            except (json.JSONDecodeError, OSError):
                continue
            if run.goal == LOOP_GOAL:
                runs.append(run)
        return runs


traces = _Traces()


def record_feedback(run_id: str, verdicts: dict[str, str]) -> None:
    """The researcher keeps some digests and deletes others. Record that verdict
    onto the run's trace, where the improvement loop will read it back."""
    path = TRACE_DIR / f"run_{run_id}.json"
    run = Run.load(path)
    for paper_id, verdict in verdicts.items():
        run.spans.append(Span(name=f"feedback:{paper_id}", parent=run.run_id, output=verdict))
    path.write_text(run.to_json(), encoding="utf-8")


def tune_triage(runs: list[Run]) -> str:
    """The improvement loop (Loop 4). Join each run's maker spans (paper -> topics)
    with its feedback spans (kept / deleted), find a tracked topic that was only
    ever deleted, and drop it from triage so tomorrow's selection is sharper."""
    topics_by_paper: dict[str, set] = {}
    verdict_by_paper: dict[str, str] = {}
    for run in runs:
        for s in run.spans:
            if s.name.startswith("maker:") and isinstance(s.input, dict):
                topics_by_paper[s.input.get("id")] = set(s.input.get("topics", []))
            elif s.name.startswith("feedback:"):
                verdict_by_paper[s.name.split(":", 1)[1]] = s.output

    tracked = set(state.triage["tracked_topics"])
    kept, deleted = set(), set()
    for paper_id, verdict in verdict_by_paper.items():
        topics = topics_by_paper.get(paper_id, set())
        if verdict == "kept":
            kept |= topics
        elif verdict == "deleted":
            deleted |= topics

    # A tracked topic deleted and never kept is a systematic miss.
    to_drop = sorted(((deleted - kept) & tracked) - set(state.triage["drop_topics"]))
    if not to_drop:
        return "no change (no tracked topic was systematically deleted)"
    state.triage["drop_topics"] = sorted(set(state.triage["drop_topics"]) | set(to_drop))
    state._save()
    return f"dropped {to_drop} from triage (deleted, never kept)"


# ---------------------------------------------------------------------------
# The morning run — the four loops wired into one operated system, body matching
# the article's sketch. The human is off the crank; what remains is owning what
# "relevant" and "verified" mean.
# ---------------------------------------------------------------------------
def morning_run(*, live: bool = False) -> dict:
    global USE_LIVE_MAKER
    USE_LIVE_MAKER = live
    connectors.reset()
    day = date.today().isoformat()
    tracer = Tracer(goal=LOOP_GOAL)
    processed: list[str] = []

    with span(tracer, "trigger", parent=tracer.run_id) as ts:
        candidates = arxiv.new_today(topics=tracked_topics())   # trigger: an event, not a prompt
        ts.output = {"candidates": [p["id"] for p in candidates]}

    for paper in candidates:
        if state.already_surveyed(paper):                       # state file: skip done work
            continue
        with worktree(paper):                                   # isolated, parallel-safe
            with span(tracer, f"maker:{paper['id']}", parent=tracer.run_id, input=paper) as ms:
                draft = run_subagent("summarize", paper=paper)  # the maker
                ms.output = draft.text
            with span(tracer, f"verify:{paper['id']}", parent=tracer.run_id) as vs:
                review = run_subagent("verify", draft=draft)    # the checker: separate role
                vs.output = {"grounded": review.all_claims_grounded, "flags": review.flags}
                vs.status = "ok" if review.all_claims_grounded else "error"
            if review.all_claims_grounded:                      # checked against the Part 14 graph
                connectors.post_digest(draft)                   # act in the real world
            else:
                connectors.to_human_inbox(draft, review.flags)  # ask / abstain (Part 17)
        state.record(paper)
        processed.append(paper["id"])

    with span(tracer, "connectors", parent=tracer.run_id) as cs:
        dpath, ipath = connectors.write(day)
        cs.output = {"digest": len(connectors.digest), "inbox": len(connectors.inbox)}

    with span(tracer, "tune_triage", parent=tracer.run_id) as hs:
        note = tune_triage(traces.since("yesterday"))           # improvement: fold results back in
        hs.output = note

    tracer.save()  # saved after tune_triage, so this run is not its own input
    return {
        "run_id": tracer.run_id,
        "candidates": [p["id"] for p in candidates],
        "processed": processed,
        "digest": [d.paper_id for d in connectors.digest],
        "inbox": [d.paper_id for d, _ in connectors.inbox],
        "tune": note,
        "digest_path": str(dpath),
        "inbox_path": str(ipath),
    }


# ---------------------------------------------------------------------------
# Offline demo: two mornings. Day one catches the planted citation and queues it
# for review; the researcher keeps the GNN digest and deletes the retrieval one;
# day two's improvement pass reads that feedback and sharpens triage.
# ---------------------------------------------------------------------------
def _seed_world() -> None:
    for paper in FEED:
        GRAPH.add_paper(paper)  # CITES edges come free from each paper's references


def _clear_loop_artifacts() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    if TRACE_DIR.exists():
        for path in TRACE_DIR.glob("run_*.json"):
            try:
                stale = Run.load(path).goal == LOOP_GOAL
            except (json.JSONDecodeError, OSError):
                stale = False
            if stale:
                path.unlink()
    for path in OUTPUT_DIR.glob("digest_*.md"):
        path.unlink()
    for path in OUTPUT_DIR.glob("inbox_*.md"):
        path.unlink()


def _demo(live: bool = False) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    _clear_loop_artifacts()
    state.reset()
    arxiv.queue = list(FEED)
    _seed_world()

    print("Loop engineering — two morning runs (Part 18)\n" + "─" * 60)

    day1 = morning_run(live=live)
    print("Morning 1")
    print(f"  trigger    : {day1['candidates']}  (stale/off-topic papers never surface)")
    print(f"  digest     : {day1['digest']}  -> {day1['digest_path']}")
    print(f"  inbox      : {day1['inbox']}  (verifier flagged a citation the graph cannot back)")
    print(f"  improvement: {day1['tune']}")

    # The researcher reviews: keeps the GNN digest, deletes the retrieval one.
    record_feedback(day1["run_id"], {"P1": "kept", "P2": "deleted"})
    before = tracked_topics()
    print("\n[researcher keeps P1's digest, deletes P2's]\n")

    day2 = morning_run(live=live)
    after = tracked_topics()
    print("Morning 2")
    print(f"  trigger    : {day2['candidates']}  processed {day2['processed']}  "
          "(already-surveyed papers skipped)")
    print(f"  improvement: {day2['tune']}")
    print(f"  watched topics: {before}  ->  {after}  (tomorrow's selection is sharper)")

    print("\nThe verifier caught the planted citation; the triage rule tuned itself "
          "from feedback. The human owns 'relevant' and 'verified', not the crank.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop engineering (Part 18)")
    parser.add_argument("--live", action="store_true",
                        help="Use live run_subagent makers (needs an API key).")
    args = parser.parse_args()
    _demo(live=args.live)
