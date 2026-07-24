#!/usr/bin/env python3
"""Orchestrator: synthesize structure (alternative Stage 3 of ontology extraction).

Instead of inducing the taxonomy bottom-up (lexical heads, Hearst, clustering),
this stage has an LLM *synthesize* a parsimonious ontology from the salient
candidates, fenced by grounding and scoped by Stage 0. It yields a cleaner
hierarchy and, importantly, LLM-assigned domain/range on relations (fixing the
bottom-up co-occurrence heuristic's weak point). Output is the SAME schema as the
bottom-up Stage 3, so Stage 4 consumes it unchanged.

Input  : the Stage 2b salient vocabulary (``<stem>_salient.json``) and the Stage 0
         scope (auto-detected) for the domain statement and competency questions.
Output : ``<vocabulary_stem>_synth_structure.json`` next to the input.

Mechanism:
  1. LLM taxonomy: select classes from the candidates, attach each to a parent,
     coin a few abstract parents (flagged).
  2. Ground: map every label back to a candidate (provenance); a label that is
     neither a candidate nor a coined parent is kept but flagged llm_introduced.
  3. LLM relations: assign domain/range (class labels) to the meaningful relations.
  4. Axioms over coined-family siblings; lightweight reasoner; emit.

Usage:
    python synthesize_structure.py SALIENT_VOCAB_JSON [--scope PATH] [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ground  # noqa: E402
import reason  # noqa: E402
import llm_synthesize  # noqa: E402
from model import (ClassNode, ParentEdge, CoinedParent, Relationship,  # noqa: E402
                   slugify, VIA_LLM)


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
    ap = argparse.ArgumentParser(description="Stage 3 (synthesis): synthesize structure")
    ap.add_argument("vocabulary_file", type=Path, help="Stage 2b salient vocabulary JSON")
    ap.add_argument("--scope", type=Path, default=None)
    ap.add_argument("--llm-model", default=llm_synthesize.DEFAULT_MODEL)
    ap.add_argument("--max-concepts", type=int, default=0,
                    help="cap candidate concepts sent to the LLM (0 = all)")
    ap.add_argument("--max-relations", type=int, default=200,
                    help="cap candidate relations sent to the LLM")
    ap.add_argument("--env", type=Path, default=None)
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

    out_path = args.output or args.vocabulary_file.with_name(
        f"{args.vocabulary_file.stem}_synth_structure.json")

    print(f"Stage 3 Synthesize :: {args.vocabulary_file.name}  "
          f"({len(concepts)} concepts, {len(relations)} relations)")

    key = llm_synthesize.load_api_key(args.env)
    synth = llm_synthesize.LLMSynthesizer(key, model=args.llm_model)

    # --- 1. LLM taxonomy ------------------------------------------------
    conc_sorted = sorted(concepts, key=lambda c: -c.get("mention_count", 0))
    if args.max_concepts > 0:
        conc_sorted = conc_sorted[:args.max_concepts]
    conc_pairs = [(c["label"], c.get("mention_count", 0)) for c in conc_sorted]
    print(f"  synthesizing taxonomy over {len(conc_pairs)} concepts ...")
    taxo = synth.taxonomy(scope.get("domain_statement", ""),
                          scope.get("competency_questions", []), conc_pairs)
    print(f"    LLM returned {len(taxo)} classes")

    # --- 2. Ground -------------------------------------------------------
    index = ground.build_index(concepts)
    label_to_id: dict[str, str] = {}     # norm(LLM label) -> assigned id
    nodes: dict[str, ClassNode] = {}
    coined: dict[str, CoinedParent] = {}
    introduced = 0

    for c in taxo:
        lab = c["label"]
        nkey = ground.norm(lab)
        if not nkey or nkey in label_to_id:
            continue
        if c["coined"]:
            cid = f"coined:{slugify(lab)}"
            coined.setdefault(cid, CoinedParent(
                id=cid, label=lab, children=[],
                justification={"agreement": [VIA_LLM]}))
            label_to_id[nkey] = cid
        else:
            cand = ground.resolve(lab, index)
            if cand:
                cid = cand["id"]
                node = ClassNode(id=cid, label=cand["label"],
                                 alt_labels=cand.get("alt_labels", []),
                                 mention_count=cand.get("mention_count", 0))
            else:
                cid = f"concept:{slugify(lab)}"
                node = ClassNode(id=cid, label=lab, mention_count=0,
                                 flags=["llm_introduced"])
                introduced += 1
            nodes.setdefault(cid, node)
            label_to_id[nkey] = cid

    # Parent edges (second pass, now that all ids exist).
    for c in taxo:
        nkey = ground.norm(c["label"])
        cid = label_to_id.get(nkey)
        if not cid or cid not in nodes or not c["parent"]:
            continue
        pid = label_to_id.get(ground.norm(c["parent"]))
        if not pid:
            cand = ground.resolve(c["parent"], index)
            pid = cand["id"] if cand else None
        if not pid or pid == cid:
            continue
        is_coined_parent = pid in coined
        nodes[cid].parents.append(ParentEdge(
            parent=pid, via=[VIA_LLM], confidence=0.7,
            flagged=is_coined_parent or "llm_introduced" in nodes[cid].flags))
        if is_coined_parent:
            coined[pid].children.append(cid)

    for cp in coined.values():
        cp.children = sorted(set(cp.children))
        cp.justification["n_children"] = len(cp.children)

    classes = list(nodes.values())

    # --- 3. LLM relations (domain/range over the synthesized classes) ----
    class_labels = [n.label for n in classes] + [cp.label for cp in coined.values()]
    rels_sorted = sorted(relations, key=lambda r: -r.get("mention_count", 0))
    rel_labels = [r["label"] for r in rels_sorted[:args.max_relations]]
    rel_mc = {ground.norm(r["label"]): r.get("mention_count", 0) for r in relations}
    print(f"  assigning domain/range over {len(class_labels)} classes ...")
    llm_rels = synth.relations(class_labels, rel_labels,
                               domain=scope.get("domain_statement", ""),
                               competency=scope.get("competency_questions", []),
                               suggested=scope.get("relations", []))

    # Resolve domain/range labels to ids.
    name_to_id = dict(label_to_id)
    for n in classes:
        name_to_id.setdefault(ground.norm(n.label), n.id)
    for cp in coined.values():
        name_to_id.setdefault(ground.norm(cp.label), cp.id)

    relationships: list[Relationship] = []
    dropped_rels = 0
    seen_rel: set[str] = set()
    for r in llm_rels:
        dom = name_to_id.get(ground.norm(r["domain"]))
        rng = name_to_id.get(ground.norm(r["range"]))
        if not dom or not rng:
            dropped_rels += 1
            continue
        rid = f"relation:{slugify(r['label'])}"
        if rid in seen_rel:
            continue
        seen_rel.add(rid)
        relationships.append(Relationship(
            id=rid, label=r["label"], domain=dom, range=rng,
            domain_source="llm", range_source="llm",
            observed_subjects=[dom], observed_objects=[rng],
            evidence_count=rel_mc.get(ground.norm(r["label"]), 0)))

    # --- 4. Axioms + reasoner -------------------------------------------
    axioms = []
    for cp in coined.values():
        axioms.extend(reason.disjoint_axioms_for_family(cp.children))
    pmap = reason.build_parent_map(classes, list(coined.values()))
    cyc = reason.break_cycles(classes, pmap)
    disj = reason.check_disjointness(axioms, pmap)

    # Orphans (top-level classes with no parent are roots, not failures; only flag
    # llm-introduced ungrounded ones, already flagged).
    n_orphan = 0
    for n in classes:
        if not n.parents:
            n.orphan = True
            if "llm_introduced" in n.flags:
                n.flags.append("orphan")
            n_orphan += 1

    omitted = [c["label"] for c in concepts
               if ground.norm(c["label"]) not in label_to_id]

    via_counts = Counter(v for n in classes for e in n.parents for v in e.via)
    stats = {
        "in_concepts": len(concepts), "in_relations": len(relations),
        "classes": len(classes), "coined_parents": len(coined),
        "llm_introduced": introduced, "with_parent": sum(1 for n in classes if n.parents),
        "roots": n_orphan, "omitted_candidates": len(omitted),
        "relationships": len(relationships), "relations_dropped": dropped_rels,
        "axioms": len(axioms), "edges_by_via": dict(via_counts),
    }
    consistency = {"reasoner": "lightweight (cycles + disjointness)",
                   "consistent": True, "cycles_broken": len(cyc),
                   "axioms_relaxed": sum(1 for a in axioms if a.relaxed),
                   "issues": cyc + disj}

    out = {
        "stage": "3-synthesize-structure",
        "vocabulary_file": str(args.vocabulary_file),
        "scope_file": str(scope_path) if scope_path else None,
        "input_file": voc.get("input_file"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"llm_model": args.llm_model, "max_concepts": args.max_concepts},
        "stats": stats,
        "consistency": consistency,
        "classes": [n.to_dict() for n in classes],
        "coined_parents": [cp.to_dict() for cp in coined.values()],
        "relationships": [r.to_dict() for r in relationships],
        "axioms": [a.to_dict() for a in axioms],
        "omitted_candidates": omitted,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Stage 3 (synthesis) done")
    print(f"    classes    : {len(classes)} ({stats['with_parent']} parented, "
          f"{introduced} llm-introduced, {len(coined)} coined)")
    print(f"    relations  : {len(relationships)} with domain/range "
          f"({dropped_rels} dropped for unresolved ends)")
    print(f"    axioms     : {len(axioms)}  | omitted candidates: {len(omitted)}")
    print(f"    consistent : {consistency['consistent']} "
          f"({consistency['cycles_broken']} cycles broken)")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
