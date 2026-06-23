"""Stage 4: the RITE review loop and hallucination guard.

Wraps ``scima.ontology_learning.rite_review``. Two automated checks stand in for
the expert's judgement:

  * Test:    every meaningful word of a class label must occur in the source
             corpus (the grounding / hallucination guard).
  * Inspect: the pipeline's own confidence must clear a floor.

Anything failing either check is rejected for human follow-up. With the offline
stub, the planted ``scima:UnicornEvacuationProtocol`` is the class that gets
caught.

Usage:
    python scripts/rite_review.py [path/to/corpus.txt] [min_confidence]
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


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    min_conf = float(argv[2]) if len(argv) > 2 else 0.5
    corpus = corpus_path.read_text(encoding="utf-8")

    proposals = extract_ontology_classes(corpus, llm_client=StubLLMClient())
    result = rite_review(proposals, corpus=corpus, min_confidence=min_conf)

    total = len(result.accepted) + len(result.rejected)
    print(f"RITE review of {total} proposals (min_confidence={min_conf}):")
    print(f"  accepted: {len(result.accepted)}")
    print(f"  rejected: {len(result.rejected)}")
    for r in result.rejected:
        print(f"    rejected (not grounded in sources): {r.local_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
