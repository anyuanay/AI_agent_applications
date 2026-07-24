"""Test by kind: the heart of Stage 4.

Grounding-by-kind pays off here. The test an element must pass depends on what it
is (the article's table):

  | Kind             | Test it must pass                                          |
  |------------------|------------------------------------------------------------|
  | Extracted concept| Corpus grounding: a recorded mention or a surface in text. |
  | Coined parent    | >= 2 grounded children (+ signal agreement). 1 child -> demote.|
  | Relationship     | Predicate grounding + a sane domain/range.                 |
  | Axiom            | The reasoner finds no contradiction.                       |

Each test returns a verdict (accept / reject / escalate). The autonomy boundary
falls straight out of these: the agent acts alone on the clear cases (accept the
grounded-and-consistent, reject the unsupported) and escalates only the genuinely
ambiguous (multiple inheritance, borderline grounding, missing domain/range).
"""

from __future__ import annotations

from model import Decision, ACCEPT, REJECT, ESCALATE
from tools import Corpus, concept_grounded

# Coined-parent dispositions returned alongside a Decision.
COINED_ACCEPT = "accept"
COINED_DEMOTE = "demote"     # exactly one grounded child: demote to a plain class
COINED_REJECT = "reject"     # no grounded children


def test_concept(node: dict, corpus: Corpus) -> Decision:
    """Test an extracted concept: it must be grounded in the corpus."""
    grounded, how = concept_grounded(node, corpus)
    flags = node.get("flags", [])
    d = Decision(id=node["id"], kind="concept", verdict=ACCEPT, grounded=grounded)

    if not grounded:
        d.verdict = REJECT
        d.reasons.append("no corpus support (auto-rejected hallucination)")
        return d

    if "multi_parent" in flags:
        d.verdict = ESCALATE
        d.reasons.append("multiple inheritance; confirm the extra parent(s)")
        return d

    if how == "corpus_search":
        # Grounded only by a surface match, not a recorded mention: borderline.
        d.verdict = ESCALATE
        d.reasons.append("borderline grounding (no recorded mention; found by corpus search)")
        return d

    if node.get("orphan"):
        d.reasons.append("grounded; orphan parked under a domain top for re-parenting")
        d.parked = True
        return d  # ACCEPT + parked

    d.reasons.append("grounded by recorded mention")
    return d


def test_coined(cp: dict, grounded_children: int, total_children: int) -> tuple[Decision, str]:
    """Test a coined parent: it must have >= 2 grounded children.

    One grounded child -> demote (the coined node disappears, the child stands on
    its own). None -> reject. Returns (decision, disposition).
    """
    d = Decision(id=cp["id"], kind="coined_parent", verdict=ACCEPT,
                 grounded=grounded_children >= 2)
    agreement = (cp.get("justification") or {}).get("agreement", [])

    if grounded_children >= 2:
        d.reasons.append(f"{grounded_children} grounded children; "
                         f"agreement={agreement}")
        return d, COINED_ACCEPT
    if grounded_children == 1:
        d.verdict = REJECT
        d.reasons.append("only one grounded child; demoted to a plain class")
        return d, COINED_DEMOTE
    d.verdict = REJECT
    d.reasons.append("no grounded children; coined hypothesis not justified")
    return d, COINED_REJECT


def test_relationship(rel: dict) -> Decision:
    """Test a relationship: predicate grounding plus a sane domain and range."""
    d = Decision(id=rel["id"], kind="relationship", verdict=ACCEPT, grounded=True)
    has_dom = bool(rel.get("domain"))
    has_rng = bool(rel.get("range"))
    if has_dom and has_rng:
        d.reasons.append(f"predicate grounded; domain+range resolved "
                         f"(dom={rel.get('domain_source')}, rng={rel.get('range_source')})")
        return d
    d.verdict = ESCALATE
    missing = [n for n, ok in (("domain", has_dom), ("range", has_rng)) if not ok]
    d.reasons.append(f"predicate grounded but {', '.join(missing)} unresolved; "
                     f"needs a human to assign")
    return d


def test_axiom(ax: dict) -> Decision:
    """Test an axiom: it must not have been relaxed by the reasoner."""
    aid = "+".join(ax.get("classes", []))
    d = Decision(id=f"axiom:{ax.get('type')}:{aid}", kind="axiom",
                 verdict=ACCEPT, grounded=True)
    if ax.get("relaxed"):
        d.verdict = REJECT
        d.reasons.append("relaxed by the reasoner (contradicted the hierarchy)")
        return d
    d.reasons.append("passed the reasoner")
    return d
