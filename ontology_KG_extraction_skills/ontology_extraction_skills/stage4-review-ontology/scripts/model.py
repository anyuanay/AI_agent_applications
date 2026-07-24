"""Data model for Stage 4 (agentic RITE review).

Stage 4 is the **trust gate**. Everything upstream optimized for recall; this
stage buys back precision. A review agent reads each element's dossier, tests it
*by its kind* using real tools, and routes it to one of three outcomes, with a
human ratifying the escalations:

  - **accept**   a high-confidence element, grounded by its kind and consistent.
  - **reject**   a clear hallucination: an extracted concept with no corpus
                 support at all. Rejections go to a feedback set, not deletion.
  - **escalate** the genuinely ambiguous (disagreement flags, low confidence,
                 borderline grounding, multiple inheritance) to a human.

Along the way the agent performs the four RITE verbs: **Refine** (cheap fixes:
merge near-duplicates, demote an unjustified coined parent, park an orphan, rename
/ name a coined family), **Inspect** (read the dossier), **Test** (by kind), and
**Extend** (commit survivors; park orphans; keep the rest as feedback for the next
pass). The invariants still hold: nothing is silently dropped, every decision
records its reasons (provenance), and the test applied depends on the kind
(grounding by kind).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


ACCEPT = "accept"
REJECT = "reject"
ESCALATE = "escalate"


@dataclass
class Decision:
    """The agent's verdict on one element, with the reasons that produced it."""

    id: str
    kind: str                       # concept | coined_parent | relationship | axiom
    verdict: str                    # ACCEPT | REJECT | ESCALATE
    grounded: bool = False
    reasons: list[str] = field(default_factory=list)
    parked: bool = False            # admitted but parked under a domain top (orphan)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edit:
    """A Refine action the agent applied, kept for the audit trail.

    Refine edits are the feedback loop in action: a demote/merge is Stage 2's job
    re-run, a re-parent is Stage 3's. They are recorded as human-approvable edits,
    never silent rewrites.
    """

    type: str                       # merge | demote | park | name_coined
    targets: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
