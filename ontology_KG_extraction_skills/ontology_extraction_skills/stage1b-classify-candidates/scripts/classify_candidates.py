#!/usr/bin/env python3
"""Orchestrator: classify candidates by kind (Stage 1b of ontology extraction).

The type/instance gate. It sits between Stage 1 (surface) and Stage 2 (name) and
sorts every surfaced candidate into class / individual / non_concept, so only
universals flow into the ontology while particulars are preserved for a later
knowledge-graph pass and debris is parked as feedback.

Input  : the Stage 1 candidates JSON (``<stem>_candidates.json``), and optionally
         the Stage 0 scope (``<source>_scope.json``) for the LLM tiebreak.
Output : ``<candidates_stem>_gated.json`` (its ``candidates`` key holds the class
         candidates, so Stage 2 consumes it directly), plus the routed-out
         individuals and non-concepts in the same file.

Mechanism:
  - deterministic morphology gate (digits / LaTeX / numbered doc labels), then
  - optional LLM tiebreak (--llm, gemini-3.1-flash-lite) over the class survivors
    to catch named individuals NER cannot (Marissa, Tuesday), using the scope's
    domain statement and out-of-scope note.

Usage:
    python classify_candidates.py CANDIDATES_JSON [--llm] [--scope PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate  # noqa: E402


def _find_scope(candidates_file: Path, payload: dict) -> Path | None:
    """Best-effort: locate the Stage 0 scope file for the same source."""
    src = payload.get("input_file", "")
    if src:
        stem = Path(src).stem
        # content stem is "<orig>_content"; scope is "<orig>_scope.json"
        orig = stem[:-len("_content")] if stem.endswith("_content") else stem
        cand = Path(src).with_name(f"{orig}_scope.json")
        if cand.is_file():
            return cand
    guess = candidates_file.parent
    for p in guess.glob("*_scope.json"):
        return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1b: Classify candidates by kind")
    ap.add_argument("candidates_file", type=Path,
                    help="Stage 1 candidates JSON")
    ap.add_argument("--scope", type=Path, default=None,
                    help="Stage 0 scope JSON (default: auto-detect)")
    ap.add_argument("--llm", action="store_true",
                    help="LLM tiebreak over class survivors (gemini-3.1-flash-lite)")
    ap.add_argument("--llm-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--env", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.candidates_file.is_file():
        ap.error(f"candidates file not found: {args.candidates_file}")

    payload = json.loads(args.candidates_file.read_text(encoding="utf-8"))
    cands = payload.get("candidates", [])
    out_path = args.output or args.candidates_file.with_name(
        f"{args.candidates_file.stem}_gated.json")

    print(f"Stage 1b Classify :: {args.candidates_file.name}  ({len(cands)} candidates)")

    classes: list[dict] = []
    individuals: list[dict] = []
    non_concepts: list[dict] = []
    reasons: Counter = Counter()

    # --- deterministic gate ---------------------------------------------
    for c in cands:
        binname, reason = gate.classify(c)
        rec = dict(c)
        if binname == gate.CLASS:
            classes.append(rec)
        elif binname == gate.INDIVIDUAL:
            rec["gate_reason"] = reason
            individuals.append(rec)
        else:
            rec["gate_reason"] = reason
            non_concepts.append(rec)
        reasons[reason] += 1
    print(f"  deterministic: {len(classes)} class, {len(individuals)} individual, "
          f"{len(non_concepts)} non-concept")

    # --- optional LLM tiebreak over class concepts ----------------------
    scope_path = args.scope or _find_scope(args.candidates_file, payload)
    scope = {}
    if scope_path and Path(scope_path).is_file():
        scope = json.loads(Path(scope_path).read_text(encoding="utf-8"))

    llm_moved = 0
    if args.llm:
        import llm_classify
        try:
            key = llm_classify.load_api_key(args.env)
            clf = llm_classify.LLMClassifier(
                key, model=args.llm_model,
                domain=scope.get("domain_statement", ""),
                out_of_scope=scope.get("out_of_scope", []))
            # Only re-check concept classes (relations stay object properties).
            concept_classes = [c for c in classes if c.get("kind") == "concept"]
            labels = clf.classify([c["canonical"] for c in concept_classes])
            kept: list[dict] = [c for c in classes if c.get("kind") != "concept"]
            for c in concept_classes:
                lab = labels.get(c["canonical"], "class")
                if lab == "individual":
                    c["gate_reason"] = "LLM: named individual"; individuals.append(c); llm_moved += 1
                elif lab == "non_concept":
                    c["gate_reason"] = "LLM: not a domain term"; non_concepts.append(c); llm_moved += 1
                else:
                    kept.append(c)
            classes = kept
            print(f"  LLM tiebreak: moved {llm_moved} class -> individual/non-concept")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] LLM tiebreak skipped ({str(exc)[:100]})")

    n_rel = sum(1 for c in classes if c.get("kind") == "relation")
    n_con = sum(1 for c in classes if c.get("kind") == "concept")
    stats = {
        "in_candidates": len(cands),
        "classes": len(classes), "class_concepts": n_con, "class_relations": n_rel,
        "individuals": len(individuals), "non_concepts": len(non_concepts),
        "llm_moved": llm_moved,
        "by_reason": dict(reasons),
    }

    out = {
        "stage": "1b-classify-candidates",
        "candidates_file": str(args.candidates_file),
        "input_file": payload.get("input_file"),
        "scope_file": str(scope_path) if scope_path else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"llm": bool(args.llm and llm_moved >= 0 and scope is not None),
                   "llm_model": args.llm_model if args.llm else None},
        "stats": stats,
        "candidates": classes,          # class candidates -> Stage 2 reads this key
        "individuals": individuals,      # A-Box particulars, kept for a later KG pass
        "non_concepts": non_concepts,    # debris, parked as feedback
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Stage 1b done")
    print(f"    classes    : {len(classes)} ({n_con} concept, {n_rel} relation)")
    print(f"    individuals: {len(individuals)}  non-concepts: {len(non_concepts)}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
