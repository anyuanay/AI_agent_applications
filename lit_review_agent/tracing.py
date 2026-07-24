"""
Observability — spans and traces (Part 11).

The unit of visibility is the trace (one run) made of spans (one unit of work),
nested into the run's causal tree. A span captures identity + parent (which
builds the tree), verbatim input/output (the bug is in the exact text), timing,
token/cost, and a status — including the sneaky `empty` (succeeded but zero
results) that does not throw.

Instrumentation lives here, in the harness layer, not in each tool: it is a
guarantee made once, not a courtesy each tool has to remember.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRACE_DIR = Path(__file__).parent / "traces"


def now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Span:
    """One unit of work: a model call, a tool call, a retrieval, a sub-agent."""

    name: str
    parent: str | None
    span_id: str = field(default_factory=_new_id)
    start: float = field(default_factory=now)
    end: float | None = None
    status: str = "ok"  # ok | empty | error
    error: str | None = None
    input: object = None
    output: object = None
    tokens: int = 0

    @property
    def ms(self) -> int:
        if self.end is None:
            return 0
        return int((self.end - self.start) * 1000)


class Tracer:
    """Accumulates the spans of a single run into one trace."""

    def __init__(self, run_id: str | None = None, goal: str = "") -> None:
        self.run_id = run_id or _new_id()
        self.goal = goal
        self.spans: list[Span] = []

    def record(self, span: Span) -> None:
        self.spans.append(span)

    def to_run(self) -> "Run":
        return Run(run_id=self.run_id, goal=self.goal, spans=list(self.spans))

    def save(self, directory: Path = TRACE_DIR) -> Path:
        directory.mkdir(exist_ok=True)
        path = directory / f"run_{self.run_id}.json"
        path.write_text(self.to_run().to_json(), encoding="utf-8")
        return path


def is_empty(output: object) -> bool:
    """Detect the succeeded-but-zero-results case that does not raise."""
    if output is None:
        return True
    text = output if isinstance(output, str) else json.dumps(output)
    try:
        data = json.loads(text) if isinstance(text, str) else output
    except (json.JSONDecodeError, TypeError):
        return text.strip() == ""
    if isinstance(data, dict):
        if data.get("status") == "empty":
            return True
        if data.get("hits") == 0 or data.get("count") == 0:
            return True
        if "papers" in data and not data["papers"]:
            return True
    return False


@contextmanager
def span(tracer: Tracer, name: str, parent: str | None, **attrs):
    """Emit a span around an instrumented call. Same shape as Part 11.

        with span(tracer, "tool:" + call.name, parent=run_id,
                  input=call.args) as s:
            s.output = TOOL_DISPATCH[call.name](**call.args)
            s.status = "empty" if is_empty(s.output) else "ok"
    """
    s = Span(name=name, parent=parent, input=attrs.get("input"))
    try:
        yield s
    except Exception as e:  # noqa: BLE001 — record then re-raise
        s.status = "error"
        s.error = repr(e)
        raise
    finally:
        s.end = now()
        tracer.record(s)  # ship to the trace store


# ---------------------------------------------------------------------------
# Run — a recorded trace, with the accessors graders and dashboards read.
# This is the artifact everything downstream (Part 12 evaluation) consumes.
# ---------------------------------------------------------------------------


@dataclass
class Run:
    run_id: str
    goal: str
    spans: list[Span]

    # ----- serialization -----
    def to_json(self) -> str:
        return json.dumps(
            {"run_id": self.run_id, "goal": self.goal,
             "spans": [asdict(s) for s in self.spans]},
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Run":
        data = json.loads(text)
        spans = [Span(**s) for s in data["spans"]]
        return cls(run_id=data["run_id"], goal=data.get("goal", ""), spans=spans)

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ----- accessors used by the trace-reading checklist and the eval graders -----
    def tool_spans(self, tool: str | None = None) -> list[Span]:
        out = [s for s in self.spans if s.name.startswith("tool:")]
        if tool:
            out = [s for s in out if s.name == f"tool:{tool}"]
        return out

    def outputs_for(self, tool: str) -> list[object]:
        return [s.output for s in self.tool_spans(tool)]

    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.spans)

    def has_anomaly(self) -> bool:
        return any(s.status in ("empty", "error") for s in self.spans)
