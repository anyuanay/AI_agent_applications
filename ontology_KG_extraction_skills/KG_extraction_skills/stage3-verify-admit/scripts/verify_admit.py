#!/usr/bin/env python3
"""Stage 3 — Verify & Admit: validate mapped triples against the ontology.

Reads _mapped_triples.json and _ontology_index.json (both written by Stage 2),
sorts all triples by confidence, runs five checks on each triple in order:
  1. Domain check (subclass closure)
  2. Range check — object_property (subclass closure)
  3. Range check — datatype_property (XSD compatibility + repair)
  4. Disjointness check (against accumulated admitted types)
  5. Cardinality check (functional, max-cardinality, inverse-functional)

Admits triples that pass all checks and emits rdf:type assertions for every
admitted individual. Stage 3 is purely rule-based and needs no LLM or
embedding calls.

Usage:
    python verify_admit.py MAPPED_TRIPLES [ONTOLOGY_INDEX] [options]

    MAPPED_TRIPLES  : _mapped_triples.json from Stage 2
    ONTOLOGY_INDEX  : _ontology_index.json (default: inferred from mapped path)
    --output-dir DIR: where to write output files (default: same dir as input)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from confidence_sorter import sort_by_confidence
from checker import Checker


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path}  ({path.stat().st_size:,} bytes)")


def _infer_stem(mapped_path: Path) -> str:
    return mapped_path.name.replace("_mapped_triples.json", "")


def _infer_index_path(mapped_path: Path, stem: str) -> Path:
    return mapped_path.parent / f"{stem}_ontology_index.json"


def _admitted_record(t: dict, verdict: str) -> dict:
    rec: dict = {
        "subject_iri":        t["subject_iri"],
        "subject_type":       t["subject_type"],
        "predicate_iri":      t["predicate_iri"],
        "predicate_kind":     t.get("predicate_kind"),
        "sentence_index":     t.get("sentence_index"),
        "sources":            t.get("sources", []),
        "mapping_confidence": t.get("mapping_confidence"),
        "verdict":            verdict,
    }
    if t.get("predicate_kind") == "object_property":
        rec["object_iri"]  = t.get("object_iri")
        rec["object_type"] = t.get("object_type")
    else:
        rec["object_literal"]  = t.get("object_literal")
        rec["object_datatype"] = t.get("object_datatype")
    return rec


def _rejected_record(t: dict, reason: str) -> dict:
    rec = _admitted_record(t, "reject")
    rec["reason"] = reason
    return rec


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 3: Verify and admit mapped triples"
    )
    ap.add_argument("mapped_triples", type=Path,
                    help="_mapped_triples.json from Stage 2")
    ap.add_argument("ontology_index", type=Path, nargs="?", default=None,
                    help="_ontology_index.json (default: inferred from mapped path)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output directory (default: same as mapped triples)")
    args = ap.parse_args(argv)

    if not args.mapped_triples.is_file():
        ap.error(f"mapped triples file not found: {args.mapped_triples}")

    stem     = _infer_stem(args.mapped_triples)
    idx_path = args.ontology_index or _infer_index_path(args.mapped_triples, stem)
    if not idx_path.is_file():
        ap.error(f"ontology index not found: {idx_path}")

    out_dir = args.output_dir or args.mapped_triples.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\nStage 3 Verify & Admit")
    print(f"  mapped triples : {args.mapped_triples}")
    print(f"  ontology index : {idx_path}")

    # ── [1/4] Load ───────────────────────────────────────────────────────
    print("\n[1/4] Loading inputs …")
    mapped_data = _load_json(args.mapped_triples)
    idx         = _load_json(idx_path)
    triples     = mapped_data.get("mapped_triples", [])
    print(f"      mapped triples    : {len(triples):,}")
    print(f"      classes in index  : {len(idx.get('classes', [])):,}")
    print(f"      properties        : {len(idx.get('properties', [])):,}")
    print(f"      disjointness pairs: {len(idx.get('disjointness_pairs', [])):,}")

    # ── [2/4] Sort ───────────────────────────────────────────────────────
    print("\n[2/4] Sorting triples by confidence (highest first) …")
    sorted_triples = sort_by_confidence(triples)

    # ── [3/4] Verify ─────────────────────────────────────────────────────
    print(f"\n[3/4] Verifying {len(sorted_triples):,} triples …")
    checker = Checker(idx)
    admitted: list[dict] = []
    rejected: list[dict] = []

    for i, triple in enumerate(sorted_triples):
        if (i + 1) % 50 == 0 or (i + 1) == len(sorted_triples):
            print(f"    [{i+1}/{len(sorted_triples)}] "
                  f"admitted={len(admitted)} rejected={len(rejected)}")

        verdict, reason, t_out = checker.check(triple)
        if verdict in ("admit", "repaired"):
            admitted.append(_admitted_record(t_out, verdict))
        else:
            rejected.append(_rejected_record(t_out, reason))

    # ── [4/4] rdf:type emission ──────────────────────────────────────────
    print("\n[4/4] Emitting rdf:type assertions …")
    seen: set[tuple[str, str]] = set()
    for rec in admitted:
        seen.add((rec["subject_iri"], rec["subject_type"]))
        if rec.get("predicate_kind") == "object_property" and rec.get("object_iri"):
            seen.add((rec["object_iri"], rec["object_type"]))

    type_assertions = [
        {"subject_iri": ind, "predicate_iri": "rdf:type", "object_iri": cls}
        for ind, cls in sorted(seen)
    ]
    print(f"      {len(type_assertions):,} unique rdf:type assertions")

    # ── Write outputs ─────────────────────────────────────────────────────
    n_repaired   = sum(1 for r in admitted if r["verdict"] == "repaired")
    reason_counts = dict(Counter(r.get("reason") for r in rejected))

    _write_json(out_dir / f"{stem}_admitted_triples.json", {
        "stage":        "3-verify-admit",
        "generated_at": ts,
        "input":        str(args.mapped_triples),
        "stats": {
            "input_mapped": len(triples),
            "admitted":     len(admitted),
            "repaired":     n_repaired,
            "rejected":     len(rejected),
            "checker":      checker.stats,
        },
        "admitted_triples": admitted,
    })

    _write_json(out_dir / f"{stem}_type_assertions.json", {
        "stage":        "3-verify-admit",
        "generated_at": ts,
        "stats":        {"total": len(type_assertions)},
        "type_assertions": type_assertions,
    })

    _write_json(out_dir / f"{stem}_rejected_triples.json", {
        "stage":        "3-verify-admit",
        "generated_at": ts,
        "stats": {
            "total":   len(rejected),
            "reasons": reason_counts,
        },
        "rejected_triples": rejected,
    })

    # ── Summary ───────────────────────────────────────────────────────────
    dt = time.time() - t0
    print(f"\nStage 3 done in {dt:.1f}s")
    print(f"  admitted : {len(admitted):,}  (repaired: {n_repaired})")
    print(f"  rejected : {len(rejected):,}  {reason_counts}")
    print(f"  rdf:type : {len(type_assertions):,} assertions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
