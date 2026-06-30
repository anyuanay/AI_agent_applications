#!/usr/bin/env python3
"""Orchestrator: name the vocabulary (Stage 2 of ontology extraction).

Turn the noisy *bag of candidate mentions* from Stage 1 into a clean, flat
**vocabulary of named elements**. This is the *term to concept* transition: many
surface strings collapse into one named thing. The output is a flat set of
concept names and a flat set of relation names, each carrying its cluster of
lexicalizations and its pooled provenance. There are still **no edges**;
structure is Stage 3's job.

Input  : the Stage 1 candidates JSON (``<stem>_candidates.json``) and the
         original source text it was extracted from (read from the candidates'
         ``input_file``, or overridden with ``--source``).
Output : ``<candidates_stem>_vocabulary.json`` next to the candidates file,
         i.e. ``<INPUT>_candidates_vocabulary.json``.

Mechanism, three steps:

    group  ->  select  ->  name

  - group  : meaning-level merge via context-enriched embeddings (OpenAI),
             lexically-guarded clustering by cosine distance. Lexical fallback
             with --no-embeddings.
  - select : drop only the obvious non-concepts; record each rejection (recall-
             first: a rejection is feedback, not a deletion).
  - name   : one canonical label per group (PascalCase concept / camelCase
             relation), every surface kept as an alt_label.

Usage:
    python name_vocabulary.py CANDIDATES_JSON [options]

    --source PATH          original text (default: candidates' input_file)
    --no-embeddings        lexical grouping only (offline, no API key needed)
    --embed-model MODEL    OpenAI embedding model (default: text-embedding-3-small)
    --embed-dim N          embedding dimensionality (default: 1536)
    --no-merge-guard       disable the lexical guard on embedding merges
    --embed-dim N          embedding dimensionality (default: 768)
    --group-threshold F    cosine-distance cutoff for merging (default: 0.15)
    --context-windows N    occurrence windows per candidate to embed (default: 3)
    --context-width N      context chars on each side of an occurrence (default: 60)
    --max-candidates N     cap candidates per kind for a quick test (0 = all)
    --batch-size N         embedding batch size (default: 100)
    --env PATH             explicit .env holding OPENAI_API_KEY
    --output PATH          override the output path
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed  # noqa: E402
import group as grouping  # noqa: E402
import name as naming  # noqa: E402
import selection as selecting  # noqa: E402
from element import KIND_CONCEPT, KIND_RELATION  # noqa: E402


def _cap_candidates(cands: list[dict], cap: int) -> list[dict]:
    """Keep the strongest ``cap`` candidates (by mention count) for a quick test."""
    if cap <= 0 or len(cands) <= cap:
        return cands
    return sorted(cands, key=lambda c: -int(c.get("mention_count") or 0))[:cap]


def _process_kind(kind: str, cands: list[dict], source: str, args,
                  embedder) -> tuple[list, list]:
    """Group, name, and select one kind. Returns (kept_elements, dropped_pairs)."""
    if not cands:
        return [], []

    print(f"\n  {kind}s: {len(cands)} candidates")

    # --- group ------------------------------------------------------------
    if embedder is not None:
        texts = [embed.context_text(c, source, args.context_windows,
                                    args.context_width) for c in cands]
        print(f"    embedding {len(texts)} candidates ...")
        vecs = embedder.embed(texts)
        groups = grouping.group_semantic(
            vecs, cands, args.group_threshold, guard=not args.no_merge_guard)
    else:
        groups = grouping.group_lexical(cands)
    print(f"    grouped into {len(groups)} meanings")

    # --- name -------------------------------------------------------------
    elements = [naming.build_element([cands[i] for i in g], kind) for g in groups]

    # --- select -----------------------------------------------------------
    kept, dropped = selecting.partition(elements)
    print(f"    kept {len(kept)}, dropped {len(dropped)} non-concepts")
    return kept, dropped


def build_stats(concepts, relations, dropped_c, dropped_r,
                n_cand_c, n_cand_r) -> dict:
    kept = concepts + relations
    return {
        "candidate_concepts_in": n_cand_c,
        "candidate_relations_in": n_cand_r,
        "concepts": len(concepts),
        "relations": len(relations),
        "elements_total": len(kept),
        "dropped_concepts": len(dropped_c),
        "dropped_relations": len(dropped_r),
        "merge_ratio_concepts": round(n_cand_c / len(concepts), 2) if concepts else 0,
        "merge_ratio_relations": round(n_cand_r / len(relations), 2) if relations else 0,
        "literal_span_elements": sum(1 for e in kept if e.literal_span),
        "llm_only_elements": sum(1 for e in kept if e.llm_only),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: Name the vocabulary")
    ap.add_argument("candidates_file", type=Path,
                    help="Stage 1 candidates JSON (<stem>_candidates.json)")
    ap.add_argument("--source", type=Path, default=None,
                    help="original text (default: candidates' input_file)")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="lexical grouping only (no API key needed)")
    ap.add_argument("--embed-model", default=embed.DEFAULT_MODEL,
                    help="OpenAI embedding model id")
    ap.add_argument("--embed-dim", type=int, default=embed.DEFAULT_DIM,
                    help="embedding dimensionality (text-embedding-3-* support "
                         "shortening below the 1536 default)")
    ap.add_argument("--group-threshold", type=float, default=0.07,
                    help="cosine-distance cutoff for merging (smaller = stricter)")
    ap.add_argument("--no-merge-guard", action="store_true",
                    help="disable the lexical guard on embedding merges "
                         "(allows synonym merges with no shared token, but also "
                         "lets co-hyponyms like numerator/denominator merge)")
    ap.add_argument("--context-windows", type=int, default=3)
    ap.add_argument("--context-width", type=int, default=60)
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="cap candidates per kind for a quick test (0 = all)")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="inputs per embedding request (OpenAI allows up to 2048)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="embedding cache path (default: <stem>_embcache.npz)")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read or write the embedding cache")
    ap.add_argument("--env", type=Path, default=None,
                    help="path to .env holding OPENAI_API_KEY")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.candidates_file.is_file():
        ap.error(f"candidates file not found: {args.candidates_file}")

    payload_in = json.loads(args.candidates_file.read_text(encoding="utf-8"))
    candidates = payload_in.get("candidates", [])

    # Source text: needed for context-aware embeddings.
    src_path = args.source or Path(payload_in.get("input_file", ""))
    source = ""
    if src_path and Path(src_path).is_file():
        source = Path(src_path).read_text(encoding="utf-8", errors="replace")
    elif not args.no_embeddings:
        print(f"  [warn] source text not found at {src_path}; "
              f"embedding on surfaces only (no context)")

    out_path = args.output or args.candidates_file.with_name(
        f"{args.candidates_file.stem}_vocabulary.json"
    )

    t0 = time.time()
    print(f"Stage 2 Name :: {args.candidates_file.name}  "
          f"({len(candidates):,} candidates)")

    concept_cands = _cap_candidates(
        [c for c in candidates if c.get("kind") == KIND_CONCEPT],
        args.max_candidates)
    relation_cands = _cap_candidates(
        [c for c in candidates if c.get("kind") == KIND_RELATION],
        args.max_candidates)

    # --- embedder (optional) ---------------------------------------------
    embedder = None
    if not args.no_embeddings:
        try:
            key = embed.load_api_key(args.env)
            cache = None
            if not args.no_cache:
                cache_path = args.cache or args.candidates_file.with_name(
                    f"{args.candidates_file.stem}_embcache.npz")
                cache = embed.EmbeddingCache(cache_path, args.embed_model,
                                             args.embed_dim)
            embedder = embed.Embedder(key, model=args.embed_model,
                                      dim=args.embed_dim,
                                      batch_size=args.batch_size, cache=cache)
            print(f"  embedding model: {args.embed_model} (dim {embedder.dim})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] embeddings unavailable ({str(exc)[:160]}); "
                  f"falling back to lexical grouping")
            embedder = None
    else:
        print("  grouping: lexical only (--no-embeddings)")

    concepts, dropped_c = _process_kind(KIND_CONCEPT, concept_cands, source,
                                        args, embedder)
    relations, dropped_r = _process_kind(KIND_RELATION, relation_cands, source,
                                         args, embedder)

    concepts.sort(key=lambda e: (-e.mention_count, e.label.casefold()))
    relations.sort(key=lambda e: (-e.mention_count, e.label.casefold()))

    stats = build_stats(concepts, relations, dropped_c, dropped_r,
                        len(concept_cands), len(relation_cands))

    out = {
        "stage": "2-name-vocabulary",
        "candidates_file": str(args.candidates_file),
        "input_file": payload_in.get("input_file"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "grouping": "lexical" if embedder is None else "semantic",
            "embed_model": None if embedder is None else args.embed_model,
            "embed_dim": None if embedder is None else embedder.dim,
            "group_threshold": args.group_threshold,
            "context_windows": args.context_windows,
            "context_width": args.context_width,
            "max_candidates": args.max_candidates,
        },
        "stats": stats,
        "concepts": [e.to_dict() for e in concepts],
        "relations": [e.to_dict() for e in relations],
        "dropped": [
            {**e.to_dict(), "drop_reason": reason}
            for e, reason in (dropped_c + dropped_r)
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    dt = time.time() - t0
    print(f"\n  Stage 2 done in {dt:.1f}s")
    print(f"    concepts : {stats['concepts']} "
          f"(from {stats['candidate_concepts_in']} candidates, "
          f"merge {stats['merge_ratio_concepts']}x)")
    print(f"    relations: {stats['relations']} "
          f"(from {stats['candidate_relations_in']} candidates, "
          f"merge {stats['merge_ratio_relations']}x)")
    print(f"    dropped  : {stats['dropped_concepts']} concept, "
          f"{stats['dropped_relations']} relation")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
