#!/usr/bin/env python3
"""Stage 1 — Extract: surface entity mentions and raw triples from a source text.

No ontology is consulted. Recall-first: pull everything, sort nothing.

Runs three sub-passes in order:
  1. spaCy pass: NER + noun-chunk mentions, sentence segmentation, dep-parse triples
  2. LLM pass (Gemini): batched entity/value surfacing + triple extraction
  3. Entity mention merge + triple merge

Outputs (written next to the source file, or to --output-dir):
  {stem}_sentences.json
  {stem}_entity_mentions.json
  {stem}_dep_triples.json
  {stem}_llm_triples.json
  {stem}_candidate_triples.json
  {stem}_negated_triples.json

Usage:
    python extract.py SOURCE_FILE [options]

    --no-llm                  spaCy pass only (no API key needed)
    --model MODEL             Gemini model (default: gemini-2.0-flash-lite)
    --batch-size N            sentences per LLM batch (default: 7)
    --spacy-model MODEL       spaCy model (default: en_core_web_sm)
    --env PATH                explicit path to .env holding GOOGLE_API_KEY
    --output-dir DIR          directory for output files (default: same as source)
    --max-sentences N         cap total sentences sent to LLM (evenly sampled; 0 = all)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preprocess as preprocess_mod
import spacy_pass
import llm_pass
import entity_merge
import triple_merge
from models import CandidateTriple


def _sample_sentences(sentences, max_n: int):
    """Evenly sample up to max_n sentences; 0 = all."""
    if max_n <= 0 or len(sentences) <= max_n:
        return sentences
    step = len(sentences) / max_n
    return [sentences[int(i * step)] for i in range(max_n)]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path}  ({path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: Extract entity mentions and triples")
    ap.add_argument("source_file", type=Path, help="source text/markdown file")
    ap.add_argument("--no-llm", action="store_true",
                    help="spaCy pass only (no API key needed)")
    ap.add_argument("--model", default=llm_pass.DEFAULT_MODEL,
                    help="Gemini model id for the LLM pass")
    ap.add_argument("--batch-size", type=int, default=llm_pass.DEFAULT_BATCH_SIZE,
                    help="sentences per LLM batch (default: %(default)s)")
    ap.add_argument("--spacy-model", default="en_core_web_sm",
                    help="spaCy model name (must be installed)")
    ap.add_argument("--env", type=Path, default=None,
                    help="explicit path to .env holding GOOGLE_API_KEY")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output directory (default: same as source)")
    ap.add_argument("--max-sentences", type=int, default=0,
                    help="cap sentences for LLM pass (evenly sampled; 0 = all)")
    args = ap.parse_args(argv)

    if not args.source_file.is_file():
        ap.error(f"source file not found: {args.source_file}")

    source_text = args.source_file.read_text(encoding="utf-8", errors="replace")
    out_dir = args.output_dir or args.source_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source_file.stem

    t0 = time.time()
    print(f"\nStage 1 Extract :: {args.source_file}  ({len(source_text):,} chars)")

    # ── Preprocessing ─────────────────────────────────────────────────────
    print(f"\n[0/3] Preprocessing (exercise removal + Markdown cleaning) ...")
    source_text, prep_stats = preprocess_mod.preprocess(source_text)
    print(f"      original:      {prep_stats['original_chars']:,} chars")
    print(f"      after exercises removed: {prep_stats['after_exercise_removal']:,} chars")
    print(f"      after markdown clean:    {prep_stats['after_markdown_clean']:,} chars  "
          f"({prep_stats['pct_removed']}% removed)")

    # ── spaCy pass ────────────────────────────────────────────────────────
    print(f"\n[1/3] spaCy pass (model={args.spacy_model}) ...")
    nlp = spacy_pass.load_nlp(args.spacy_model)
    sentences, spacy_raw_mentions, dep_triples = spacy_pass.run(source_text, nlp)
    print(f"      sentences:      {len(sentences):,}")
    print(f"      raw mentions:   {len(spacy_raw_mentions):,} (NER + noun_chunk)")
    print(f"      dep triples:    {len(dep_triples):,}")

    # Write sentences and dep triples immediately
    ts = datetime.now(timezone.utc).isoformat()

    _write_json(out_dir / f"{stem}_sentences.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {"sentence_count": len(sentences)},
        "sentences": [s.to_dict() for s in sentences],
    })

    _write_json(out_dir / f"{stem}_dep_triples.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {
            "total": len(dep_triples),
            "negated": sum(1 for t in dep_triples if t.negated),
        },
        "dep_triples": [t.to_dict() for t in dep_triples],
    })

    # ── LLM pass ─────────────────────────────────────────────────────────
    llm_raw_mentions = []
    llm_triples = []

    if args.no_llm:
        print("\n[2/3] LLM pass skipped (--no-llm)")
    else:
        print(f"\n[2/3] LLM pass (model={args.model}, batch_size={args.batch_size}) ...")
        api_key = llm_pass.load_api_key(args.env)
        sents_for_llm = _sample_sentences(sentences, args.max_sentences)
        if args.max_sentences > 0 and len(sents_for_llm) < len(sentences):
            print(f"      sentences sampled: {len(sents_for_llm)} of {len(sentences)}")
        llm_raw_mentions, llm_triples = llm_pass.run(
            sents_for_llm, api_key, model=args.model, batch_size=args.batch_size
        )
        print(f"      LLM mentions:   {len(llm_raw_mentions):,}")
        print(f"      LLM triples:    {len(llm_triples):,}")

    _write_json(out_dir / f"{stem}_llm_triples.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {
            "total": len(llm_triples),
            "negated": sum(1 for t in llm_triples if t.negated),
        },
        "llm_triples": [t.to_dict() for t in llm_triples],
    })

    # ── Entity mention merge ──────────────────────────────────────────────
    print("\n[3/3] Merge ...")
    all_raw_mentions = spacy_raw_mentions + llm_raw_mentions
    entity_mentions = entity_merge.merge(all_raw_mentions)
    print(f"      entity mentions: {len(entity_mentions):,}")

    _write_json(out_dir / f"{stem}_entity_mentions.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {
            "total": len(entity_mentions),
            "entity": sum(1 for m in entity_mentions if m.mention_type == "entity"),
            "value":  sum(1 for m in entity_mentions if m.mention_type == "value"),
        },
        "entity_mentions": [m.to_dict() for m in entity_mentions],
    })

    # ── Triple merge ──────────────────────────────────────────────────────
    candidate_triples, negated_triples = triple_merge.merge(
        dep_triples, llm_triples, entity_mentions, nlp
    )
    print(f"      candidate triples: {len(candidate_triples):,}")
    print(f"      negated triples:   {len(negated_triples):,}")

    multi_source = sum(
        1 for t in candidate_triples if len(t.sources) > 1
    )

    _write_json(out_dir / f"{stem}_candidate_triples.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {
            "total": len(candidate_triples),
            "multi_source": multi_source,
            "dep_only":  sum(1 for t in candidate_triples if t.sources == ["dep_parse"]),
            "llm_only":  sum(1 for t in candidate_triples if t.sources == ["llm"]),
            "both":      multi_source,
        },
        "candidate_triples": [t.to_dict() for t in candidate_triples],
    })

    _write_json(out_dir / f"{stem}_negated_triples.json", {
        "stage": "1-extract",
        "input_file": str(args.source_file),
        "generated_at": ts,
        "stats": {"total": len(negated_triples)},
        "negated_triples": [
            {**t.to_dict(), "negated": True}
            for t in negated_triples
        ],
    })

    # ── Summary ───────────────────────────────────────────────────────────
    dt = time.time() - t0
    print(f"\nStage 1 done in {dt:.1f}s")
    print(f"  sentences:         {len(sentences):,}")
    print(f"  entity_mentions:   {len(entity_mentions):,}")
    print(f"  dep_triples:       {len(dep_triples):,}")
    print(f"  llm_triples:       {len(llm_triples):,}")
    print(f"  candidate_triples: {len(candidate_triples):,}  "
          f"(both sources: {multi_source})")
    print(f"  negated_triples:   {len(negated_triples):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
