#!/usr/bin/env python3
"""Orchestrator: rank by salience (Stage 2b of ontology extraction).

Keep the salient core of the vocabulary and park the long tail. A domain's
ontology is its central concepts, not every common noun; this stage scores each
concept's centrality (scope match, frequency, spread, grounding) and splits the
vocabulary into a kept set and a parked set. Nothing is deleted: parked concepts
are feedback for a later pass.

Input  : the Stage 2 vocabulary JSON (``<stem>_vocabulary.json``) and the Stage 0
         scope (``<source>_scope.json``, auto-detected) for the scope-match signal.
Output : ``<vocabulary_stem>_salient.json`` whose ``concepts``/``relations`` keys
         hold the kept core (Stage 3 consumes it directly), plus a ``parked`` list.

Keep rules (a concept is kept if ANY holds):
  - it matches an author-marked scope term (always salient),
  - its mention_count >= --min-mentions,
  - --top-n is set and it is in the top N by salience score.
Relations are kept when mention_count >= --relation-min-mentions.

Usage:
    python rank_salience.py VOCABULARY_JSON [--scope PATH] [--min-mentions N]
                            [--top-n N] [--relation-min-mentions N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import salience  # noqa: E402


def _find_scope(payload: dict, vocab_file: Path) -> Path | None:
    src = payload.get("input_file", "")
    if src:
        stem = Path(src).stem
        orig = stem[:-len("_content")] if stem.endswith("_content") else stem
        cand = Path(src).with_name(f"{orig}_scope.json")
        if cand.is_file():
            return cand
    for p in vocab_file.parent.glob("*_scope.json"):
        return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2b: Rank by salience")
    ap.add_argument("vocabulary_file", type=Path, help="Stage 2 vocabulary JSON")
    ap.add_argument("--scope", type=Path, default=None,
                    help="Stage 0 scope JSON (default: auto-detect)")
    ap.add_argument("--min-mentions", type=int, default=3,
                    help="keep a concept seen at least this many times (default 3)")
    ap.add_argument("--relation-min-mentions", type=int, default=2,
                    help="keep a relation seen at least this many times (default 2)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="also cap kept concepts to the top N by score (0 = no cap)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.vocabulary_file.is_file():
        ap.error(f"vocabulary file not found: {args.vocabulary_file}")

    voc = json.loads(args.vocabulary_file.read_text(encoding="utf-8"))
    concepts = voc.get("concepts", [])
    relations = voc.get("relations", [])

    scope_path = args.scope or _find_scope(voc, args.vocabulary_file)
    scope = json.loads(Path(scope_path).read_text(encoding="utf-8")) if scope_path \
        and Path(scope_path).is_file() else {}
    scope_terms = salience.build_scope_terms(scope)

    src = voc.get("input_file", "")
    doc_len = len(Path(src).read_text(encoding="utf-8", errors="replace")) \
        if src and Path(src).is_file() else 0

    out_path = args.output or args.vocabulary_file.with_name(
        f"{args.vocabulary_file.stem}_salient.json")

    print(f"Stage 2b Salience :: {args.vocabulary_file.name}  "
          f"({len(concepts)} concepts, {len(relations)} relations)")
    print(f"  scope terms: {len(scope_terms)}  doc_len: {doc_len:,}")

    # Score every concept.
    max_log_mc = max((math.log1p(c.get("mention_count", 0)) for c in concepts),
                     default=1.0)
    for c in concepts:
        s, comp = salience.score(c, max_log_mc, scope_terms, doc_len)
        c["salience"] = round(s, 4)
        c["salience_components"] = comp

    concepts.sort(key=lambda c: -c["salience"])

    # Decide keep vs park.
    top_ids: set[str] = set()
    if args.top_n > 0:
        top_ids = {c["id"] for c in concepts[:args.top_n]}

    kept_concepts, parked = [], []
    for c in concepts:
        scope_hit = c["salience_components"]["scope_match"] == 1.0
        frequent = c.get("mention_count", 0) >= args.min_mentions
        in_top = (not args.top_n) or c["id"] in top_ids
        keep = (scope_hit or frequent) and in_top
        if keep:
            kept_concepts.append(c)
        else:
            reason = []
            if not scope_hit:
                reason.append("not in scope")
            if not frequent:
                reason.append(f"mention_count < {args.min_mentions}")
            if args.top_n and c["id"] not in top_ids:
                reason.append(f"below top-{args.top_n}")
            c["park_reason"] = "; ".join(reason) or "low salience"
            parked.append(c)

    kept_relations, parked_relations = [], []
    for r in relations:
        if r.get("mention_count", 0) >= args.relation_min_mentions:
            kept_relations.append(r)
        else:
            r["park_reason"] = f"mention_count < {args.relation_min_mentions}"
            parked_relations.append(r)

    stats = {
        "in_concepts": len(concepts), "kept_concepts": len(kept_concepts),
        "parked_concepts": len(parked),
        "in_relations": len(relations), "kept_relations": len(kept_relations),
        "parked_relations": len(parked_relations),
        "scope_matched": sum(1 for c in concepts
                             if c["salience_components"]["scope_match"] == 1.0),
        "min_mentions": args.min_mentions, "top_n": args.top_n,
    }

    out = {
        "stage": "2b-salience",
        "vocabulary_file": str(args.vocabulary_file),
        "scope_file": str(scope_path) if scope_path else None,
        "input_file": src,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"min_mentions": args.min_mentions, "top_n": args.top_n,
                   "relation_min_mentions": args.relation_min_mentions},
        "stats": stats,
        "concepts": kept_concepts,      # salient core -> Stage 3 reads this
        "relations": kept_relations,
        "parked": parked + parked_relations,   # long tail, feedback for next pass
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Stage 2b done")
    print(f"    concepts : kept {len(kept_concepts)} / parked {len(parked)} "
          f"(scope-matched {stats['scope_matched']})")
    print(f"    relations: kept {len(kept_relations)} / parked {len(parked_relations)}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
