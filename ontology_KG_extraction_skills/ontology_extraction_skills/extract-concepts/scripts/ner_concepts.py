"""Stage 1b: NER-style candidate concept extraction.

Folds the Hearst hypernym/hyponym pairs into a deduplicated set of candidate
concept labels. In a production pipeline this is where a named-entity recognizer
(spaCy, a fine-tuned tagger, or an LLM tagger) widens the net beyond explicit
enumerations. Here we derive candidates deterministically from the Hearst pairs
so the skill runs offline with no model download.

Usage:
    python scripts/ner_concepts.py [path/to/corpus.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SKILLS_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "ontology_kg_for_agents"), str(_SKILLS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scima.ontology_learning import hearst_hyponyms  # noqa: E402

DEFAULT_CORPUS = _SKILLS_ROOT / "shared" / "corpus" / "emergency_procedures.txt"


def candidate_concepts(text: str) -> list[str]:
    """Deduplicated, order-preserving list of candidate concept labels."""
    seen: set[str] = set()
    concepts: list[str] = []
    for hyper, hypo in hearst_hyponyms(text):
        for term in (hyper, hypo):
            if term not in seen:
                seen.add(term)
                concepts.append(term)
    return concepts


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    text = corpus_path.read_text(encoding="utf-8")
    concepts = candidate_concepts(text)
    print(f"Found {len(concepts)} candidate concepts in {corpus_path.name}:")
    for c in concepts:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
