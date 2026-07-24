"""
Uncertainty and provenance (Parts 16 and 17).

Part 17's thesis: a neural agent acts with the same confidence whether it knows
the answer or is inventing it. The fix is a vocabulary for doubt — separate the
uncertainty you can reduce (epistemic) from the one you cannot (aleatoric) — and
a response policy that follows from it: resolve, ask, hedge, or abstain.

This module is the practical core, the part that is buildable today:

  - detection      : signals ranked by trust. The structural/graph check is the
                     only one grounded in something other than the model's own
                     self-report, so it is the one the gate leans on.
  - assert_citation: the confidence gate before a factual claim, backed by the
                     Part 14 graph as the verification surface. If "X cites Y" is
                     not an edge in the graph, the agent does not state it as
                     fact — it resolves it, asks, or flags it unverified.
  - provenance     : Part 16's user-facing face of the same idea. Every surfaced
                     claim carries where it came from; generation is marked as
                     generation, so a person can calibrate trust (the Part 16
                     over-/under-reliance failure).

The response-policy mapping (Section 3) made explicit:

    EPISTEMIC, agent can reduce      -> RESOLVE  (retrieve / call a tool / plan)
    EPISTEMIC, only the user can      -> ASK      (a clarifying question)
    ALEATORIC, bounded                -> HEDGE     (range + confidence + a floor)
    ALEATORIC, unbounded / high-stakes-> ABSTAIN   (decline or escalate)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from graph import GRAPH

# The four terminal actions of the response policy (Part 17, Section 3).
RESOLVE, ASK, HEDGE, ABSTAIN, STATE = "resolve", "ask", "hedge", "abstain", "state"


def response_policy(*, reducible: bool, by_agent: bool, bounded: bool, high_stakes: bool) -> str:
    """Map an uncertainty's shape to the action it calls for (the Figure 2 matrix)."""
    if reducible:
        return RESOLVE if by_agent else ASK           # epistemic
    if bounded and not high_stakes:
        return HEDGE                                   # aleatoric, manageable
    return ABSTAIN                                      # aleatoric, unbounded/high-stakes


# ---------------------------------------------------------------------------
# Detection — signals, ranked by trust (Part 17, Section 2 / Figure 1).
# The honest note: the three "self-report" signals can be confidently wrong;
# the structural check is the only one validated against ground truth.
# ---------------------------------------------------------------------------
def verbalized_confidence(_claim: str) -> float | None:
    """'How sure are you?' -> a number. Convenient and ungrounded: the number is
    itself a generated token, so a confident hallucination rates itself high.
    Stubbed; never the basis for a consequential action on its own."""
    return None


def self_consistency(samples: list[str]) -> float:
    """Sample N times, measure agreement. More reliable than logprobs, costs N x
    tokens, and mostly catches aleatoric spread — when every sample agrees on the
    same falsehood it reports high agreement anyway."""
    if not samples:
        return 0.0
    most_common = Counter(s.strip() for s in samples).most_common(1)[0][1]
    return most_common / len(samples)


def structural_check(from_id: str, rel: str, to_id: str) -> bool:
    """The trustworthy detector: does the claimed edge exist in the graph?
    Not 'the model feels sure' but 'this assertion checks against ground truth'."""
    return GRAPH.has_edge(from_id, rel, to_id)


# ---------------------------------------------------------------------------
# Provenance (Part 16, Section 5): point at an external artifact, not at the
# model's own narration. A verified claim is stronger than an explanation.
# ---------------------------------------------------------------------------
@dataclass
class Claim:
    text: str
    source: str          # "graph", "tool:semantic_scholar", or "model" (generation)
    verified: bool

    def render(self) -> str:
        if self.verified:
            return f"{self.text}  [source: {self.source}, verified]"
        if self.source == "model":
            return f"{self.text}  [unverified: model generation, not a source]"
        return f"{self.text}  [unverified: no supporting {self.source} record]"


# ---------------------------------------------------------------------------
# assert_citation — the confidence gate (Part 17, Section 5).
# Mirrors the harness.py sketch in the article: verify, else resolve, else ask,
# else flag. The agent never asserts an unverified citation as fact.
# ---------------------------------------------------------------------------
def assert_citation(
    from_id: str,
    to_id: str,
    *,
    fetcher=None,   # callable(from_id, to_id) -> bool: can we go check? (epistemic)
    asker=None,     # callable(question) -> str: resolve ambiguity with the user
    ambiguous: bool = False,
) -> dict:
    """Decide how to handle the claim 'from_id cites to_id'.

    Returns a structured decision: which branch of the response policy fired, the
    text to surface, and whether it is verified. The branches are the article's:
    structurally verified -> state; checkable -> resolve; ambiguous -> ask; else
    -> flag unverified (never assert as fact)."""
    if structural_check(from_id, "CITES", to_id):
        claim = Claim(f"{from_id} cites {to_id}", source="graph", verified=True)
        return {"action": STATE, "claim": claim, "text": claim.render()}

    if fetcher and fetcher(from_id, to_id):          # epistemic: reduce it by acting
        GRAPH.add_edge("CITES", f"Paper:{from_id}", f"Paper:{to_id}")
        claim = Claim(f"{from_id} cites {to_id}", source="tool:semantic_scholar", verified=True)
        return {"action": RESOLVE, "claim": claim, "text": claim.render()}

    if ambiguous and asker:                          # epistemic: only the user knows
        answer = asker(f"Which paper do you mean by '{from_id}' / '{to_id}'?")
        return {"action": ASK, "answer": answer, "text": f"(asked user: {answer})"}

    claim = Claim(f"{from_id} may cite {to_id}", source="graph", verified=False)
    return {"action": ABSTAIN, "claim": claim, "text": claim.render()}


# ---------------------------------------------------------------------------
# Offline demo: the gate verifies a real edge and flags a missing one.
# ---------------------------------------------------------------------------
def _demo() -> None:
    GRAPH.add_paper({"id": "A", "title": "Seed", "references": ["B"]})  # A -CITES-> B
    GRAPH.add_paper({"id": "C", "title": "Other"})

    print("Confidence gate (Part 17)\n" + "─" * 60)
    for frm, to, note in [("A", "B", "real CITES edge"), ("A", "C", "no edge")]:
        d = assert_citation(frm, to)
        print(f"  {frm} -> {to:<3} ({note:<16}) action={d['action']:<8} {d['text']}")
    print("\nVerified claims state the source; unverified claims are flagged, not asserted.")


if __name__ == "__main__":
    _demo()
