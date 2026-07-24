#!/usr/bin/env python3
"""Orchestrator: extract scope (Stage 0 of ontology extraction).

Frame the ontology before extracting it. The bottom-up pipeline (surface, name,
structure, review) has no notion of what the document is *about* or at what level
of abstraction the ontology should sit, so it drowns in worked-example instances.
Stage 0 supplies the top-down spine: a domain statement, the topics, the seed
concepts the author themselves defined, competency questions that bound scope, and
an explicit out-of-scope note that names the instance/fact kinds the later stages
must keep OUT of the ontology.

Input  : the original source document (text or markdown).
Output : ``<input_stem>_scope.json`` next to the input.

Mechanism:
  - mine the document structure deterministically (TOC, objectives, definitions,
    bold terms) -> a high-precision skeleton, then
  - (optional, --llm, gemini-3.1-flash-lite) summarize the skeleton into a domain
    statement, cleaned topics, general key concepts/relations, competency
    questions, and an out-of-scope note.

The scope artifact is consumed downstream: Stage 1b (type/instance gate) uses the
domain statement and out-of-scope note; Stage 2b (salience) boosts the seed
concepts; a synthesis step is scoped by the topics and competency questions.

Usage:
    python extract_scope.py INPUT_FILE [--llm] [--output PATH] [--env PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import structure  # noqa: E402


def _merge_unique(*lists) -> list[str]:
    seen, out = set(), []
    for lst in lists:
        for x in lst or []:
            k = x.casefold()
            if k not in seen:
                seen.add(k); out.append(x)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: Extract scope")
    ap.add_argument("input_file", type=Path, help="source document")
    ap.add_argument("--llm", action="store_true",
                    help="use gemini-3.1-flash-lite to synthesize the scope")
    ap.add_argument("--no-content", action="store_true",
                    help="do not emit the exercise/example-stripped content file")
    ap.add_argument("--llm-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--env", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.input_file.is_file():
        ap.error(f"input file not found: {args.input_file}")

    source = args.input_file.read_text(encoding="utf-8", errors="replace")
    out_path = args.output or args.input_file.with_name(
        f"{args.input_file.stem}_scope.json")

    print(f"Stage 0 Scope :: {args.input_file.name}  ({len(source):,} chars)")
    struct = structure.parse(source)
    print(f"  structure: {len(struct['topics'])} topics, "
          f"{len(struct['objectives'])} objectives, "
          f"{len(struct['defined_terms'])} defined terms")

    # Content filtering: drop exercise / example / activity sections so their
    # instances never reach concept surfacing. Stage 1 should run on this file.
    content_file = None
    content_stats = {}
    if not args.no_content:
        content, dropped = structure.content_regions(source)
        content_file = args.input_file.with_name(f"{args.input_file.stem}_content.md")
        content_file.write_text(content, encoding="utf-8")
        content_stats = {
            "sections_dropped": len(dropped),
            "chars_total": len(source),
            "chars_kept": len(content),
            "chars_dropped": len(source) - len(content),
            "pct_kept": round(100 * len(content) / max(len(source), 1), 1),
        }
        print(f"  content: dropped {len(dropped)} exercise/example section(s); "
              f"kept {content_stats['pct_kept']}% of text "
              f"-> {content_file.name}")

    llm_out: dict = {}
    if args.llm:
        import llm_scope
        try:
            key = llm_scope.load_api_key(args.env)
            scoper = llm_scope.LLMScope(key, model=args.llm_model)
            print(f"  summarizing skeleton with {args.llm_model} ...")
            llm_out = scoper.summarize(structure.skeleton_text(struct))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] LLM scoping unavailable ({str(exc)[:120]})")
            llm_out = {}

    # Seed concepts: the author's defined terms, plus the LLM's general concepts.
    key_terms = _merge_unique(struct["defined_terms"], llm_out.get("key_concepts"))
    topics = _merge_unique(struct["topics"], llm_out.get("topics"))

    out = {
        "stage": "0-scope",
        "input_file": str(args.input_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"llm": bool(llm_out), "llm_model": args.llm_model if llm_out else None},
        "domain_statement": llm_out.get("domain_statement", ""),
        "topics": topics,
        "key_terms": key_terms,
        "relations": llm_out.get("relations", []),
        "competency_questions": llm_out.get("competency_questions", []),
        "out_of_scope": llm_out.get("out_of_scope", []),
        "content_file": str(content_file) if content_file else None,
        "structure": struct,
        "stats": {
            "topics": len(topics), "key_terms": len(key_terms),
            "objectives": len(struct["objectives"]),
            "defined_terms": len(struct["defined_terms"]),
            "competency_questions": len(llm_out.get("competency_questions", [])),
            **content_stats,
        },
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Stage 0 done")
    if out["domain_statement"]:
        print(f"    domain : {out['domain_statement'][:90]}")
    print(f"    topics : {len(topics)}  key_terms: {len(key_terms)}  "
          f"competency_qs: {len(out['competency_questions'])}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
