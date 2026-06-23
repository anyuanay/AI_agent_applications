"""Stage 1a: Hearst-pattern hyponym extraction.

A thin, runnable wrapper over ``scima.ontology_learning.hearst_hyponyms`` so the
skill stays in lockstep with the backing package. Surfaces (hypernym, hyponym)
pairs from lexico-syntactic contexts such as "X such as A, B and C".

Usage:
    python scripts/hearst_patterns.py [path/to/corpus.txt]
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


def main(argv: list[str]) -> int:
    corpus_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    text = corpus_path.read_text(encoding="utf-8")
    pairs = hearst_hyponyms(text)
    print(f"Extracted {len(pairs)} hypernym/hyponym pairs from {corpus_path.name}:")
    for hyper, hypo in pairs:
        print(f"  {hypo}  is-a  {hyper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
