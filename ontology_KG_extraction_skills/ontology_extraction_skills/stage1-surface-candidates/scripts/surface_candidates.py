#!/usr/bin/env python3
"""Orchestrator: surface candidate ontology terms from a document.

Maximize recall of terms that could become ontology elements.
Input: one source document. Output: a bag of
candidate mentions, each a concept- or relation-mention tagged with its
provenance, written next to the input as ``<input_stem>_candidates.json``.

Mechanism:

    union( spaCy extractor , LLM extractor )  ->  string-level dedup

  - spacy extractor  : hallucination-free literal spans (recall floor + anchor)
  - LLM extractor    : the implicit / multi-word / relational lift, tagged
  - dedup            : casefold + lemmatize + merge string variants

Three design principles:
  - recall-first        : nothing is scored away here; pruning is left for later
  - provenance everywhere: every record carries source tags + char offsets
  - grounding by kind   : each mention is typed concept vs. relation now

Usage:
    python surface_candidates.py INPUT_FILE [options]

    --no-llm                 spaCy extractor only (offline, no API key needed)
    --adjectives             also surface adjective qualities as concepts
    --model MODEL            Gemini model (default: gemini-3.1-flash-lite)
    --spacy-model MODEL      spaCy model (default: en_core_web_sm)
    --llm-chunk-chars N      chars per LLM chunk (default: 8000)
    --llm-max-chunks N       cap LLM chunks, evenly sampled (0 = all; default 0)
    --env PATH               explicit .env holding GOOGLE_API_KEY
    --output PATH            override the output path
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a plain script: make sibling modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate import Candidate, Occurrence, RawCandidate, KIND_CONCEPT, KIND_RELATION  # noqa: E402
import spaCy_extractor  # noqa: E402
import llm_extractor  # noqa: E402
from dedup import dedup  # noqa: E402
from textnorm import key_from_text  # noqa: E402


def find_occurrences(raw_text: str, surface: str, cap: int = 50) -> list[Occurrence]:
    """Find literal (case-insensitive) occurrences of ``surface`` in the source.

    This is what decides literal-span vs. implicit for an LLM term: an LLM
    proposal found verbatim earns literal-span provenance; one that is not found
    is kept anyway (recall-first) but stays non-literal for review.
    """
    if not surface:
        return []
    occ: list[Occurrence] = []
    # Word-boundary match so a short term does not match inside a larger word:
    # "is a" must not match inside "is available"; "add" not inside "addition".
    pattern = re.compile(
        r"(?<![0-9A-Za-z])" + re.escape(surface) + r"(?![0-9A-Za-z])",
        re.IGNORECASE,
    )
    for m in pattern.finditer(raw_text):
        occ.append(Occurrence(start=m.start(), end=m.end(), text=m.group(0)))
        if len(occ) >= cap:
            break
    return occ


def resolve_llm_candidates(nlp, raw_text: str,
                           llm_raws: list[RawCandidate]) -> list[RawCandidate]:
    """Key each LLM term and attach literal offsets where the term is in the text.

    Returns a fresh list of RawCandidates: one per literal occurrence found, plus
    a single offset-less record when the term is implicit (not in the text). The
    offset-less record is what keeps an implicit LLM proposal in the bag.
    """
    resolved: list[RawCandidate] = []
    for r in llm_raws:
        key = key_from_text(nlp, r.text)
        if not key:
            continue
        occ = find_occurrences(raw_text, r.text)
        if occ:
            for o in occ:
                resolved.append(RawCandidate(
                    text=r.text, kind=r.kind, source=r.source, key=key,
                    start=o.start, end=o.end,
                ))
        else:
            # Implicit candidate: tracked, but no literal span.
            resolved.append(RawCandidate(
                text=r.text, kind=r.kind, source=r.source, key=key,
            ))
    return resolved


def build_stats(candidates: list[Candidate], n_spacy: int, n_llm: int,
                n_chunks: int) -> dict:
    concepts = [c for c in candidates if c.kind == KIND_CONCEPT]
    relations = [c for c in candidates if c.kind == KIND_RELATION]
    return {
        "raw_spacy_mentions": n_spacy,
        "raw_llm_mentions": n_llm,
        "llm_chunks_processed": n_chunks,
        "candidates_total": len(candidates),
        "concepts": len(concepts),
        "relations": len(relations),
        "literal_span_candidates": sum(1 for c in candidates if c.literal_span),
        "llm_only_candidates": sum(1 for c in candidates if c.llm_only),
        "implicit_candidates": sum(1 for c in candidates if not c.literal_span),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: Surface candidates")
    ap.add_argument("input_file", type=Path, help="source document to extract from")
    ap.add_argument("--no-llm", action="store_true",
                    help="spaCy extractor only (no API key needed)")
    ap.add_argument("--adjectives", action="store_true",
                    help="also surface adjective qualities (ADJ in amod/acomp) "
                         "as concepts, tagged 'adj'")
    ap.add_argument("--model", default=llm_extractor.DEFAULT_MODEL,
                    help="Gemini model id")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--llm-chunk-chars", type=int, default=8000)
    ap.add_argument("--llm-max-chunks", type=int, default=0,
                    help="cap LLM chunks (evenly sampled); 0 = all")
    ap.add_argument("--env", type=Path, default=None,
                    help="path to .env holding GOOGLE_API_KEY")
    ap.add_argument("--output", type=Path, default=None,
                    help="override output path")
    args = ap.parse_args(argv)

    if not args.input_file.is_file():
        ap.error(f"input file not found: {args.input_file}")

    raw_text = args.input_file.read_text(encoding="utf-8", errors="replace")
    out_path = args.output or args.input_file.with_name(
        f"{args.input_file.stem}_candidates.json"
    )

    t0 = time.time()
    print(f"Stage 1 Surface :: {args.input_file}  ({len(raw_text):,} chars)")

    # --- spaCy extractor (always) -----------------------------------------
    print(f"  loading spaCy '{args.spacy_model}' ...")
    nlp = spaCy_extractor.load_nlp(args.spacy_model)
    adj_note = " + adjectives" if args.adjectives else ""
    print(f"  spaCy extraction (NER + noun phrases + predicates{adj_note}) ...")
    spacy_raws = spaCy_extractor.extract_document(
        nlp, raw_text, adjectives=args.adjectives)
    print(f"    {len(spacy_raws):,} literal mentions")

    # --- LLM extractor (optional) -----------------------------------------
    llm_raws: list[RawCandidate] = []
    n_chunks = 0
    if not args.no_llm:
        key = llm_extractor.load_api_key(args.env)
        chunks = llm_extractor.chunk_text(raw_text, args.llm_chunk_chars)
        chunks = llm_extractor.sample_chunks(chunks, args.llm_max_chunks)
        n_chunks = len(chunks)
        print(f"  LLM extraction ({args.model}, {n_chunks} chunk(s)) ...")
        ext = llm_extractor.LLMExtractor(key, model=args.model)
        for i, ch in enumerate(chunks, 1):
            got = ext.extract_chunk(ch)
            llm_raws.extend(got)
            print(f"    chunk {i}/{n_chunks}: +{len(got)} terms")
        llm_raws = resolve_llm_candidates(nlp, raw_text, llm_raws)
        print(f"    {len(llm_raws):,} LLM mentions (after offset resolution)")
    else:
        print("  LLM extraction skipped (--no-llm)")

    # --- union + dedup -----------------------------------------------------
    candidates = dedup(spacy_raws + llm_raws)

    # Provenance integrity: make every recorded occurrence text the EXACT source
    # slice at its offsets. Extractors may have stored a cleaned/whitespace-
    # collapsed surface; the offsets are authoritative, so we re-slice from the
    # original file. After this, occ.text == raw_text[occ.start:occ.end] always.
    for c in candidates:
        for o in c.occurrences:
            o.text = raw_text[o.start:o.end]

    stats = build_stats(candidates, len(spacy_raws), len(llm_raws), n_chunks)

    payload = {
        "stage": "1-surface-candidates",
        "input_file": str(args.input_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "spacy_model": args.spacy_model,
            "adjectives": args.adjectives,
            "llm_model": None if args.no_llm else args.model,
            "llm_chunk_chars": args.llm_chunk_chars,
            "llm_max_chunks": args.llm_max_chunks,
        },
        "stats": stats,
        "candidates": [c.to_dict() for c in candidates],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    dt = time.time() - t0
    print(f"\n  Stage 1 done in {dt:.1f}s")
    print(f"    candidates : {stats['candidates_total']} "
          f"({stats['concepts']} concept, {stats['relations']} relation)")
    print(f"    literal    : {stats['literal_span_candidates']}  "
          f"implicit: {stats['implicit_candidates']}  "
          f"llm_only: {stats['llm_only_candidates']}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
