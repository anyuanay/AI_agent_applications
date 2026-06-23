"""Stage 5: serialize accepted classes to a Turtle/OWL delta.

Renders the RITE-accepted `ProposedClass` set as Turtle, one `owl:Class` block
per class, with `scima:extractionConfidence` and `scima:sourceDocument`
provenance. Produces a delta over the prior ontology version, not a full rewrite.

This stage runs the whole offline pipeline (induce -> review) so it is runnable
end to end, then emits the accepted hierarchy.

Usage:
    python scripts/to_turtle.py [out.ttl] [path/to/corpus.txt]
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

HEADER = """@prefix scima: <http://scima.city/ontology#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .

# Learned ontology delta. One owl:Class per accepted class, with provenance.
"""


def class_to_turtle(c) -> str:
    lines = [f"scima:{c.local_name} a owl:Class ;"]
    if c.parent and c.parent not in ("", "scima:owl:Thing"):
        lines.append(f"    rdfs:subClassOf {c.parent} ;")
    lines.append(f'    rdfs:label "{c.label}" ;')
    lines.append(f'    scima:extractionConfidence "{c.confidence:.2f}"^^xsd:decimal ;')
    src = c.source or "(none)"
    lines.append(f'    scima:sourceDocument "{src}" .')
    return "\n".join(lines)


def render(accepted) -> str:
    blocks = [class_to_turtle(c) for c in accepted]
    return HEADER + "\n" + "\n\n".join(blocks) + "\n"


def main(argv: list[str]) -> int:
    out_path = Path(argv[1]) if len(argv) > 1 else None
    corpus_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_CORPUS
    corpus = corpus_path.read_text(encoding="utf-8")

    proposals = extract_ontology_classes(corpus, llm_client=StubLLMClient())
    accepted = rite_review(proposals, corpus=corpus).accepted
    turtle = render(accepted)

    if out_path is not None:
        out_path.write_text(turtle, encoding="utf-8")
        print(f"Wrote {len(accepted)} classes to {out_path}")
    else:
        print(turtle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
