#!/usr/bin/env python3
"""Stage 2 — Map Ontology: map candidate triples to ontology IRIs or typed literals.

Reads the six Stage 1 output files (using _candidate_triples.json and
_sentences.json) plus a target OWL/Turtle ontology, then:
  1. Builds and writes _ontology_index.json (used by Stage 3 directly).
  2. Pre-embeds all class + property labels (OpenAI text-embedding-3-small).
  3. For each candidate triple, maps subject → class instance, predicate →
     property, object → class instance or literal.
  4. Applies triple inversion when the predicate matched an inverse label.
  5. Aggregates mapping confidence (min of three component confidences).
  6. Writes _mapped_triples.json with all successfully mapped triples.

Usage:
    python map_ontology.py CANDIDATE_TRIPLES ONTOLOGY [options]

    --sentences PATH        _sentences.json (default: inferred from triples path)
    --output-dir DIR        directory for outputs (default: same as triples file)
    --entity-threshold F    cosine threshold for entity typing (default: 0.50)
    --pred-threshold F      cosine threshold for predicate mapping (default: 0.45)
    --no-llm                skip LLM fallback (embedding + string match only)
    --env PATH              explicit .env file
    --spacy-model MODEL     spaCy model for predicate lemmatization (default: en_core_web_sm)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ontology_index as ont_idx_mod
import embedder as emb_mod
from iri_registry import IRIRegistry
import llm_client as llm_mod
from typer import EntityTyper
from predicate_mapper import PredicateMapper
from object_mapper import ObjectMapper


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path}  ({path.stat().st_size:,} bytes)")


def _infer_sentences_path(triples_path: Path) -> Path:
    stem = triples_path.name.replace("_candidate_triples.json", "")
    return triples_path.parent / f"{stem}_sentences.json"


def _infer_stem(triples_path: Path) -> str:
    return triples_path.name.replace("_candidate_triples.json", "")


def _min_confidence(confs: list[str]) -> str:
    return "low" if "low" in confs else "high"


# ── main mapping loop ─────────────────────────────────────────────────────────

def map_triples(
    candidate_triples: list[dict],
    sentences: dict[int, str],       # sentence_index -> text
    entity_typer: EntityTyper,
    pred_mapper: PredicateMapper,
    obj_mapper: ObjectMapper,
) -> tuple[list[dict], list[dict]]:
    """Map all candidate triples.

    Returns:
        mapped   — successfully mapped triples
        unmapped — triples dropped at any mapping step
    """
    mapped:   list[dict] = []
    unmapped: list[dict] = []

    total = len(candidate_triples)
    for i, ct in enumerate(candidate_triples):
        if (i + 1) % 200 == 0:
            print(
                f"    [{i+1}/{total}] mapped={len(mapped)} "
                f"unmapped={len(unmapped)}"
            )

        sent_idx  = ct.get("sentence_index", 0)
        sentence  = sentences.get(sent_idx, "")

        # ── Subject typing ────────────────────────────────────────────────
        subj_result = entity_typer.type_entity(ct["subject"], sentence)
        if subj_result is None:
            unmapped.append({**ct, "drop_reason": "subject_untyped"})
            continue
        subj_iri, subj_type, subj_conf = subj_result

        # ── Predicate mapping ─────────────────────────────────────────────
        pred_result = pred_mapper.map(ct["predicate"], sentence)
        if pred_result is None:
            unmapped.append({**ct, "drop_reason": "predicate_unmapped"})
            continue
        pred_iri, pred_kind, inverted, pred_conf = pred_result

        # ── Object mapping ────────────────────────────────────────────────
        obj_result = obj_mapper.map(ct["object"], sentence, pred_kind)
        if obj_result is None:
            unmapped.append({**ct, "drop_reason": "object_unmapped"})
            continue
        obj_conf = obj_result["confidence"]

        # ── Triple inversion ──────────────────────────────────────────────
        if inverted and pred_kind == "object_property":
            # Swap subject and object positions
            subj_iri, obj_iri    = obj_result["object_iri"], subj_iri
            subj_type, obj_type  = obj_result["object_type"], subj_type
            obj_result = {
                "object_iri":  obj_iri,
                "object_type": obj_type,
                "confidence":  obj_result["confidence"],
            }
        # (Inversion not applied to datatype triples: only object_property can be inverted)

        # ── Aggregate confidence ──────────────────────────────────────────
        overall_conf = _min_confidence([subj_conf, pred_conf, obj_conf])

        # ── Build mapped triple record ────────────────────────────────────
        record: dict = {
            "subject_iri":        subj_iri,
            "subject_type":       subj_type,
            "predicate_iri":      pred_iri,
            "predicate_kind":     pred_kind,
            "sentence_index":     ct.get("sentence_index"),
            "sentence_indices":   ct.get("sentence_indices", [ct.get("sentence_index")]),
            "sources":            ct.get("sources", []),
            "mapping_confidence": overall_conf,
        }

        if pred_kind == "object_property":
            record["object_iri"]  = obj_result["object_iri"]
            record["object_type"] = obj_result["object_type"]
        else:
            record["object_literal"]  = obj_result["object_literal"]
            record["object_datatype"] = obj_result["object_datatype"]

        mapped.append(record)

    return mapped, unmapped


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: Map candidate triples to ontology IRIs")
    ap.add_argument("candidate_triples", type=Path,
                    help="_candidate_triples.json from Stage 1")
    ap.add_argument("ontology", type=Path,
                    help="OWL/Turtle ontology file (.ttl)")
    ap.add_argument("--sentences", type=Path, default=None,
                    help="_sentences.json (default: inferred from triples path)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output directory (default: same as triples file)")
    ap.add_argument("--entity-threshold", type=float, default=0.50,
                    help="cosine similarity threshold for entity typing (default: 0.50)")
    ap.add_argument("--pred-threshold", type=float, default=0.45,
                    help="cosine similarity threshold for predicate mapping (default: 0.45)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM fallback (embedding + string match only)")
    ap.add_argument("--env", type=Path, default=None,
                    help="explicit path to .env holding API keys")
    ap.add_argument("--spacy-model", default="en_core_web_sm",
                    help="spaCy model for predicate lemmatization")
    args = ap.parse_args(argv)

    if not args.candidate_triples.is_file():
        ap.error(f"candidate triples file not found: {args.candidate_triples}")
    if not args.ontology.is_file():
        ap.error(f"ontology file not found: {args.ontology}")

    sentences_path = args.sentences or _infer_sentences_path(args.candidate_triples)
    if not sentences_path.is_file():
        ap.error(f"sentences file not found: {sentences_path}")

    out_dir = args.output_dir or args.candidate_triples.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _infer_stem(args.candidate_triples)

    t0 = time.time()
    print(f"\nStage 2 Map Ontology")
    print(f"  triples : {args.candidate_triples}")
    print(f"  ontology: {args.ontology}")

    # ── Load Stage 1 outputs ──────────────────────────────────────────────
    print("\n[1/5] Loading Stage 1 outputs …")
    ct_data  = _load_json(args.candidate_triples)
    sent_data = _load_json(sentences_path)
    candidate_triples = ct_data.get("candidate_triples", [])
    sentences: dict[int, str] = {
        s["sentence_index"]: s["text"]
        for s in sent_data.get("sentences", [])
    }
    print(f"      candidate triples : {len(candidate_triples):,}")
    print(f"      sentences         : {len(sentences):,}")

    # ── Build ontology index ──────────────────────────────────────────────
    print("\n[2/5] Building ontology index …")
    idx = ont_idx_mod.build(args.ontology)
    ts  = datetime.now(timezone.utc).isoformat()
    _write_json(out_dir / f"{stem}_ontology_index.json", idx)
    print(f"      classes    : {len(idx['classes']):,}")
    print(f"      properties : {len(idx['properties']):,}")

    # ── Setup embedder ────────────────────────────────────────────────────
    print("\n[3/5] Setting up embedder (OpenAI text-embedding-3-small) …")
    openai_key = emb_mod.load_api_key(args.env)
    embedder   = emb_mod.Embedder(openai_key)

    # ── Setup LLM ─────────────────────────────────────────────────────────
    if not args.no_llm:
        google_key = llm_mod.load_api_key(args.env)
        gemini     = llm_mod.GeminiClient(google_key)
    else:
        gemini = None  # type: ignore

    # ── Setup spaCy for predicate lemmatization ────────────────────────────
    nlp = None
    try:
        import spacy
        nlp = spacy.load(args.spacy_model)
        print(f"      spaCy model loaded: {args.spacy_model}")
    except Exception as exc:
        print(f"      spaCy not available ({exc}); using lowercase-only normalization")

    # ── Setup IRI registry and mappers ─────────────────────────────────────
    print("\n[4/5] Pre-computing embeddings and building mappers …")
    iri_registry = IRIRegistry(idx["namespace"])

    entity_typer = EntityTyper(
        classes=idx["classes"],
        embedder=embedder,
        gemini=gemini,
        iri_registry=iri_registry,
        threshold=args.entity_threshold,
    )

    pred_mapper = PredicateMapper(
        properties=idx["properties"],
        embedder=embedder,
        gemini=gemini,
        nlp=nlp,
        threshold=args.pred_threshold,
    )

    obj_mapper = ObjectMapper(entity_typer=entity_typer)

    # ── Map all triples ───────────────────────────────────────────────────
    print(f"\n[5/5] Mapping {len(candidate_triples):,} candidate triples …")
    if args.no_llm:
        print("      (LLM fallback disabled)")

    mapped, unmapped = map_triples(
        candidate_triples, sentences,
        entity_typer, pred_mapper, obj_mapper
    )

    # ── Write outputs ─────────────────────────────────────────────────────
    high_conf = sum(1 for t in mapped if t["mapping_confidence"] == "high")
    low_conf  = len(mapped) - high_conf

    _write_json(out_dir / f"{stem}_mapped_triples.json", {
        "stage":       "2-map-ontology",
        "input_triples": str(args.candidate_triples),
        "ontology":    str(args.ontology),
        "generated_at": ts,
        "config": {
            "entity_threshold":   args.entity_threshold,
            "pred_threshold":     args.pred_threshold,
            "no_llm":             args.no_llm,
            "embedding_model":    emb_mod.DEFAULT_MODEL,
            "llm_model":          llm_mod.DEFAULT_MODEL if not args.no_llm else None,
        },
        "stats": {
            "candidate_input":    len(candidate_triples),
            "mapped":             len(mapped),
            "unmapped":           len(unmapped),
            "high_confidence":    high_conf,
            "low_confidence":     low_conf,
            "embedder_api_calls": embedder.stats()["api_calls"],
            "embedder_cache_hits": embedder.stats()["cache_hits"],
            "llm_calls":          gemini.calls if gemini else 0,
            "entity_typer":       entity_typer.stats,
            "pred_mapper":        pred_mapper.stats,
        },
        "mapped_triples": mapped,
    })

    _write_json(out_dir / f"{stem}_unmapped_triples.json", {
        "stage":        "2-map-ontology",
        "generated_at": ts,
        "stats":        {"total": len(unmapped)},
        "unmapped_triples": unmapped,
    })

    # ── Summary ───────────────────────────────────────────────────────────
    dt = time.time() - t0
    print(f"\nStage 2 done in {dt:.1f}s")
    print(f"  mapped:   {len(mapped):,} triples  "
          f"(high: {high_conf}, low: {low_conf})")
    print(f"  unmapped: {len(unmapped):,} triples  "
          f"(dropped — no ontology match)")
    print(f"  embedding API calls: {embedder.stats()['api_calls']}, "
          f"cache hits: {embedder.stats()['cache_hits']}")
    if gemini:
        print(f"  LLM (Gemini) calls: {gemini.calls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
