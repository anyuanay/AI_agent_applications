"""Stage 3: clustering-based class induction.

Wraps ``scima.ontology_learning.cluster_into_families``. Groups leaf protocols
under the most similar family by label-token overlap, then reports where the
clustering agrees or disagrees with the LLM-proposed parent. Agreement is the
coherence signal; disagreement is a flag for review.

Usage:
    python scripts/cluster_families.py [path/to/corpus.txt]
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
    cluster_into_families,
    extract_ontology_classes,
)

DEFAULT_CORPUS = _SKILLS_ROOT / "shared" / "corpus" / "emergency_procedures.txt"
ROOT_PARENT = "scima:EmergencyProtocol"


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    text = corpus_path.read_text(encoding="utf-8")
    proposals = extract_ontology_classes(text, llm_client=StubLLMClient())

    families = [p for p in proposals if p.parent == ROOT_PARENT]
    family_iris = {f.iri for f in families}
    leaves = [p for p in proposals if p.parent in family_iris]

    assignment = cluster_into_families(leaves, families)
    proposed_parent = {leaf.iri: leaf.parent for leaf in leaves}

    print(f"Clustered {len(leaves)} leaves under {len(families)} families.\n")
    agree = disagree = 0
    for family_iri, members in assignment.items():
        print(f"{family_iri}:")
        for leaf_iri in members:
            ok = proposed_parent.get(leaf_iri) == family_iri
            agree += ok
            disagree += not ok
            mark = "ok" if ok else "FLAG (proposed parent " + proposed_parent[leaf_iri] + ")"
            print(f"    {leaf_iri:42s} {mark}")
    print(f"\nCoherence: {agree} agree, {disagree} flagged for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
