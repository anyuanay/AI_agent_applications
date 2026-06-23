"""Stage 2: LLM schema induction.

Wraps ``scima.ontology_learning.extract_ontology_classes``: a structured-output
prompt turns source text into candidate OWL classes with parents, confidence,
and provenance. Runs offline by default via the deterministic ``StubLLMClient``;
pass a real client (for example ``anthropic.Anthropic`` wrapped to expose
``complete(system, user) -> str``) to induce for real.

Usage:
    python scripts/induce_classes.py [path/to/corpus.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SKILLS_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "ontology_kg_for_agents"), str(_SKILLS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scima.ontology_learning import StubLLMClient, extract_ontology_classes  # noqa: E402

DEFAULT_CORPUS = _SKILLS_ROOT / "shared" / "corpus" / "emergency_procedures.txt"


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    text = corpus_path.read_text(encoding="utf-8")
    proposals = extract_ontology_classes(text, llm_client=StubLLMClient())
    print(f"LLM induced {len(proposals)} candidate classes from {corpus_path.name}:")
    for c in proposals:
        parent = c.parent or "(root)"
        print(f"  {c.iri:40s} <: {parent:28s} conf={c.confidence:.2f}  src={c.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
