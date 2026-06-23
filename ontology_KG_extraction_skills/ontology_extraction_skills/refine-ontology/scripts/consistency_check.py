"""Stage 4b: logical consistency check over the accepted classes.

After RITE accepts a set of classes, an OWL reasoner confirms the hierarchy is
logically sound: no unsatisfiable classes, no class that is both A and disjoint
with A, no domain/range contradictions. In a full setup this calls a DL reasoner
(HermiT, ELK, or Pellet via owlready2). Here we run lightweight structural checks
that catch the common errors with no external reasoner, and we note where the
real reasoner would plug in.

Usage:
    python scripts/consistency_check.py [path/to/corpus.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SKILLS_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "ontology_kg_for_agents"), str(_SKILLS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scima.ontology_learning import (  # noqa: E402
    StubLLMClient,
    extract_ontology_classes,
    rite_review,
)

DEFAULT_CORPUS = _SKILLS_ROOT / "shared" / "corpus" / "emergency_procedures.txt"
ROOTS = {"", "scima:owl:Thing"}


def structural_checks(accepted: list) -> list[str]:
    """Cheap checks that approximate a reasoner's consistency pass.

    Real DL reasoning (HermiT/ELK via owlready2) replaces this once the ontology
    has disjointness and property restrictions. These catch the structural errors
    that show up first: dangling parents and subclass cycles.
    """
    violations: list[str] = []
    iris = {c.iri for c in accepted}
    parent_of = {c.iri: c.parent for c in accepted}

    # 1. Every non-root parent must be a class we actually accepted.
    for c in accepted:
        if c.parent not in ROOTS and c.parent not in iris:
            violations.append(f"dangling parent: {c.iri} -> {c.parent} (not in accepted set)")

    # 2. No cycles in the subclass chain.
    for start in iris:
        seen, cur = set(), start
        while cur in parent_of and parent_of[cur] not in ROOTS:
            cur = parent_of[cur]
            if cur in seen:
                violations.append(f"subclass cycle reaching {cur}")
                break
            seen.add(cur)

    return violations


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    corpus = corpus_path.read_text(encoding="utf-8")
    proposals = extract_ontology_classes(corpus, llm_client=StubLLMClient())
    accepted = rite_review(proposals, corpus=corpus).accepted

    violations = structural_checks(accepted)
    print(f"Consistency check over {len(accepted)} accepted classes:")
    if not violations:
        print("  no structural violations found.")
        print("  (plug in HermiT/ELK via owlready2 for full DL consistency once")
        print("   disjointness and property restrictions are present.)")
        return 0
    for v in violations:
        print(f"  VIOLATION: {v}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
