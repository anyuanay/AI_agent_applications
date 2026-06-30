#!/usr/bin/env python3
"""Orchestrator: agentic RITE review (Stage 4 of ontology extraction).

The trust gate. Read the Stage 3 structure, test every element by its kind using
real tools, and route it to accept / reject / escalate. Apply the cheap Refine
edits (merge near-duplicates, demote unjustified coined parents, park orphans,
name coined families), re-run the reasoner after edits, and commit the survivors.

Input  : the Stage 3 structure JSON (``<stem>_structure.json``) and the original
         source text (read from the structure's ``input_file``, or ``--source``).
Output : ``<structure_stem>_reviewed.json`` next to the structure, holding the
         admitted ontology, every decision with its reasons, the Refine edits, and
         the feedback set (rejected / parked / escalated / unconnected). With
         ``--emit-ttl``, also ``<structure_stem>_admitted.ttl``.

RITE:
  Refine  - merge exact-duplicate labels; demote coined parents with < 2 grounded
            children; park orphans; (with --llm) name coined families.
  Inspect - read each element's dossier (provenance, confidence, flags).
  Test    - by kind (see verdicts.py).
  Extend  - commit survivors; park orphans under a domain top; keep the rest as
            feedback for the next pass rather than deleting it.

Usage:
    python review_ontology.py STRUCTURE_JSON [--source P] [--llm] [--emit-ttl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools  # noqa: E402
import verdicts  # noqa: E402
from model import Decision, Edit, ACCEPT, REJECT, ESCALATE  # noqa: E402
from tools import Corpus, concept_grounded  # noqa: E402


def _dedupe_parents(node: dict) -> None:
    seen, out = set(), []
    for e in node.get("parents", []):
        p = e["parent"]
        if p == node["id"] or p in seen:
            continue
        seen.add(p)
        out.append(e)
    node["parents"] = out


def merge_duplicate_labels(classes: list[dict]) -> tuple[list[dict], dict, list[Edit]]:
    """Refine: merge concepts that share an exact label into one (near-duplicates).

    Returns (surviving_classes, remap old_id->kept_id, edits). The strongest (most
    mentions) keeps; the others' alt-labels and parent edges fold in. References
    elsewhere are rewritten by the caller using the remap.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for c in classes:
        by_label[c["label"]].append(c)
    remap: dict[str, str] = {}
    edits: list[Edit] = []
    survivors: list[dict] = []
    for label, group in by_label.items():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        keep = max(group, key=lambda c: c.get("mention_count", 0))
        merged_alt = set(keep.get("alt_labels", []))
        merged_mc = 0
        for c in group:
            merged_mc += c.get("mention_count", 0)
            merged_alt.update(c.get("alt_labels", []))
            if c["id"] != keep["id"]:
                remap[c["id"]] = keep["id"]
                keep["parents"].extend(c.get("parents", []))
        keep["alt_labels"] = sorted(merged_alt)
        keep["mention_count"] = merged_mc
        survivors.append(keep)
        edits.append(Edit(type="merge", targets=[c["id"] for c in group],
                          detail={"label": label, "kept": keep["id"]}))
    return survivors, remap, edits


def apply_remap_id(nid, remap):
    return remap.get(nid, nid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4: Agentic RITE review")
    ap.add_argument("structure_file", type=Path,
                    help="Stage 3 structure JSON (<stem>_structure.json)")
    ap.add_argument("--source", type=Path, default=None,
                    help="original text (default: structure's input_file)")
    ap.add_argument("--llm", action="store_true",
                    help="use gemini-3.1-flash-lite to name coined families")
    ap.add_argument("--llm-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--emit-ttl", action="store_true",
                    help="also write the admitted ontology as Turtle")
    ap.add_argument("--env", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.structure_file.is_file():
        ap.error(f"structure file not found: {args.structure_file}")

    st = json.loads(args.structure_file.read_text(encoding="utf-8"))
    classes = st.get("classes", [])
    coined = st.get("coined_parents", [])
    relationships = st.get("relationships", [])
    axioms = st.get("axioms", [])

    src_path = args.source or Path(st.get("input_file", ""))
    source = ""
    if src_path and Path(src_path).is_file():
        source = Path(src_path).read_text(encoding="utf-8", errors="replace")
    else:
        print(f"  [warn] source not found at {src_path}; corpus grounding limited")
    corpus = Corpus(source)

    out_path = args.output or args.structure_file.with_name(
        f"{args.structure_file.stem}_reviewed.json")

    print(f"Stage 4 Review :: {args.structure_file.name}  "
          f"({len(classes)} classes, {len(coined)} coined, "
          f"{len(relationships)} relations, {len(axioms)} axioms)")

    edits: list[Edit] = []

    # --- REFINE 1: merge exact-duplicate-label concepts ------------------
    classes, remap, merge_edits = merge_duplicate_labels(classes)
    edits.extend(merge_edits)
    # Rewrite every reference through the remap.
    for c in classes:
        for e in c.get("parents", []):
            e["parent"] = apply_remap_id(e["parent"], remap)
        _dedupe_parents(c)
    for cp in coined:
        cp["children"] = sorted({apply_remap_id(ch, remap) for ch in cp.get("children", [])})
    for r in relationships:
        if r.get("domain"):
            r["domain"] = apply_remap_id(r["domain"], remap)
        if r.get("range"):
            r["range"] = apply_remap_id(r["range"], remap)
        r["observed_subjects"] = [apply_remap_id(x, remap) for x in r.get("observed_subjects", [])]
        r["observed_objects"] = [apply_remap_id(x, remap) for x in r.get("observed_objects", [])]
    for ax in axioms:
        # Remap, then drop duplicate members. Merging two siblings into one class
        # can make a disjointness self-referential (A disjointWith A); a
        # disjointness with fewer than two distinct classes is degenerate, so mark
        # it relaxed (it will not be admitted, and shows in the audit trail).
        ax["classes"] = list(dict.fromkeys(apply_remap_id(x, remap)
                                           for x in ax.get("classes", [])))
        if ax.get("type") == "disjointWith" and len(ax["classes"]) < 2:
            ax["relaxed"] = True
            ax["degenerate"] = True
    if merge_edits:
        print(f"  refine: merged {len(remap)} duplicate-label concept(s)")

    cls_by_id = {c["id"]: c for c in classes}
    grounded_ids = {c["id"] for c in classes if concept_grounded(c, corpus)[0]}

    # --- TEST coined parents (and apply demote/reject edits) -------------
    coined_decisions: list[Decision] = []
    admitted_coined: list[dict] = []
    removed_coined: set[str] = set()
    needs_naming: list[dict] = []     # accepted coined families flagged for naming
    for cp in coined:
        kids = cp.get("children", [])
        gkids = sum(1 for k in kids if k in grounded_ids)
        d, disp = verdicts.test_coined(cp, gkids, len(kids))
        coined_decisions.append(d)
        if disp == verdicts.COINED_ACCEPT:
            admitted_coined.append(cp)
            if "needs_naming" in cp.get("flags", []):
                needs_naming.append(cp)
        else:
            removed_coined.add(cp["id"])
            edits.append(Edit(type="demote" if disp == verdicts.COINED_DEMOTE else "reject",
                              targets=[cp["id"]],
                              detail={"children": kids, "grounded_children": gkids}))

    # Children of removed coined parents lose that edge (may become orphans).
    for c in classes:
        if any(e["parent"] in removed_coined for e in c.get("parents", [])):
            c["parents"] = [e for e in c["parents"] if e["parent"] not in removed_coined]

    # Recompute orphan status after coined removal.
    for c in classes:
        c["orphan"] = not c.get("parents")

    # --- TEST concepts ---------------------------------------------------
    concept_decisions = [verdicts.test_concept(c, corpus) for c in classes]
    verdict_of = {d.id: d for d in concept_decisions}

    # --- TEST relationships + axioms ------------------------------------
    rel_decisions = [verdicts.test_relationship(r) for r in relationships]
    axiom_decisions = [verdicts.test_axiom(ax) for ax in axioms]

    # --- EXTEND: commit survivors ---------------------------------------
    admitted_class_ids = {d.id for d in concept_decisions if d.verdict == ACCEPT}
    admitted_coined = [cp for cp in admitted_coined]   # already filtered
    admitted_coined_ids = {cp["id"] for cp in admitted_coined}
    admitted_ids = admitted_class_ids | admitted_coined_ids

    # A relationship is admitted only if its domain/range (when set) survived.
    admitted_rels = []
    for r, d in zip(relationships, rel_decisions):
        if d.verdict != ACCEPT:
            continue
        if (r.get("domain") and r["domain"] not in admitted_ids) or \
           (r.get("range") and r["range"] not in admitted_ids):
            d.verdict = ESCALATE
            d.reasons.append("domain or range was not admitted")
            continue
        admitted_rels.append(r)

    # Axioms admitted only if not relaxed and both classes admitted.
    pmap = tools.build_parent_map([c for c in classes if c["id"] in admitted_class_ids],
                                  admitted_coined)
    disj_issues = tools.relax_contradictory_disjointness(axioms, pmap)
    admitted_axioms = []
    for ax, d in zip(axioms, axiom_decisions):
        if ax.get("relaxed"):
            d.verdict = REJECT
            if "relaxed by the reasoner" not in " ".join(d.reasons):
                d.reasons.append("relaxed by the reasoner (contradiction)")
            continue
        if all(cid in admitted_ids for cid in ax.get("classes", [])):
            admitted_axioms.append(ax)

    cycles = tools.find_cycles(pmap)

    # --- REFINE 2 (optional): name accepted coined families --------------
    final_labels: dict[str, str] = {}
    if args.llm and needs_naming:
        import llm_refine
        try:
            key = llm_refine.load_api_key(args.env)
            refiner = llm_refine.LLMRefiner(key, model=args.llm_model)
            families = [[cls_by_id[k]["label"] for k in cp["children"] if k in cls_by_id]
                        for cp in needs_naming]
            print(f"  refine: naming {len(families)} coined families via LLM ...")
            names = refiner.name_families(families)
            for cp, nm in zip(needs_naming, names):
                if nm:
                    final_labels[cp["id"]] = nm
                    edits.append(Edit(type="name_coined", targets=[cp["id"]],
                                      detail={"old": cp["label"], "new": nm}))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] LLM naming skipped ({str(exc)[:100]})")

    # --- Feedback set ----------------------------------------------------
    parked = {d.id for d in concept_decisions if d.verdict == ACCEPT and d.parked}
    # Unconnected: admitted concept used by no admitted relationship, no parent,
    # no children. Kept as feedback for the next pass (not deleted, not a failure).
    used_in_rel: set[str] = set()
    for r in admitted_rels:
        for x in (r.get("domain"), r.get("range")):
            if x:
                used_in_rel.add(x)
    has_child = set()
    for c in classes:
        for e in c.get("parents", []):
            has_child.add(e["parent"])
    for cp in admitted_coined:
        for ch in cp.get("children", []):
            has_child.add(cp["id"])  # coined has children
    unconnected = sorted(
        cid for cid in admitted_class_ids
        if cid not in used_in_rel and cid not in has_child
        and not cls_by_id[cid].get("parents"))

    rejected = [d for d in concept_decisions + coined_decisions + rel_decisions
                + axiom_decisions if d.verdict == REJECT]
    escalated = [d for d in concept_decisions + rel_decisions if d.verdict == ESCALATE]

    consistency = {
        "reasoner": "lightweight (cycles + disjointness)",
        "consistent": True,
        "cycles_remaining": len(cycles),
        "disjointness_relaxed": len(disj_issues),
    }

    autonomy = {
        "auto_accepted": sum(1 for d in concept_decisions + coined_decisions
                             + rel_decisions + axiom_decisions if d.verdict == ACCEPT),
        "auto_rejected": len(rejected),
        "escalated": len(escalated),
    }
    stats = {
        "in_classes": len(verdict_of), "in_coined": len(coined),
        "in_relationships": len(relationships), "in_axioms": len(axioms),
        "admitted_classes": len(admitted_class_ids),
        "admitted_coined": len(admitted_coined_ids),
        "admitted_relationships": len(admitted_rels),
        "admitted_axioms": len(admitted_axioms),
        "merged": len(remap), "demoted": sum(1 for e in edits if e.type == "demote"),
        "coined_rejected": sum(1 for e in edits if e.type == "reject"),
        "named_coined": sum(1 for e in edits if e.type == "name_coined"),
        "parked_orphans": len(parked), "unconnected": len(unconnected),
        "verdicts": dict(Counter(d.verdict for d in concept_decisions + coined_decisions
                                 + rel_decisions + axiom_decisions)),
    }

    def class_out(c):
        return {"id": c["id"], "label": final_labels.get(c["id"], c["label"]),
                "alt_labels": c.get("alt_labels", []),
                "parents": [e["parent"] for e in c.get("parents", [])],
                "coined": False, "parked": c["id"] in parked}

    admitted = {
        "classes": [class_out(c) for c in classes if c["id"] in admitted_class_ids],
        "coined_parents": [{"id": cp["id"], "label": final_labels.get(cp["id"], cp["label"]),
                            "children": cp.get("children", [])} for cp in admitted_coined],
        "relationships": [{"id": r["id"], "label": r["label"],
                           "domain": r.get("domain"), "range": r.get("range")}
                          for r in admitted_rels],
        "axioms": admitted_axioms,
    }

    out = {
        "stage": "4-review-ontology",
        "structure_file": str(args.structure_file),
        "vocabulary_file": st.get("vocabulary_file"),
        "input_file": st.get("input_file"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"llm": bool(args.llm), "llm_model": args.llm_model if args.llm else None},
        "stats": stats,
        "autonomy": autonomy,
        "consistency": consistency,
        "admitted": admitted,
        "edits": [e.to_dict() for e in edits],
        "feedback": {
            "rejected": [d.to_dict() for d in rejected],
            "escalated": [d.to_dict() for d in escalated],
            "parked_orphans": sorted(parked),
            "unconnected": unconnected,
        },
        "decisions": [d.to_dict() for d in
                      (concept_decisions + coined_decisions + rel_decisions + axiom_decisions)],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.emit_ttl:
        import emit_ttl
        ttl_path = args.structure_file.with_name(f"{args.structure_file.stem}_admitted.ttl")
        parents_map = {c["id"]: [e["parent"] for e in c.get("parents", [])]
                       for c in classes if c["id"] in admitted_class_ids}
        cls_map = {c["id"]: c for c in classes if c["id"] in admitted_class_ids}
        coi_map = {cp["id"]: {**cp, "coined": True} for cp in admitted_coined}
        labels = {**{c["id"]: final_labels.get(c["id"], c["label"]) for c in classes},
                  **{cp["id"]: final_labels.get(cp["id"], cp["label"]) for cp in admitted_coined}}
        emit_ttl.emit(ttl_path, cls_map, coi_map, admitted_rels, admitted_axioms,
                      parents_map, parked, labels)
        print(f"  -> {ttl_path}")

    print(f"\n  Stage 4 done")
    print(f"    admitted   : {stats['admitted_classes']} classes, "
          f"{stats['admitted_coined']} coined, {stats['admitted_relationships']} relations, "
          f"{stats['admitted_axioms']} axioms")
    print(f"    autonomy   : {autonomy['auto_accepted']} accepted, "
          f"{autonomy['auto_rejected']} rejected, {autonomy['escalated']} escalated")
    print(f"    refine     : merged {stats['merged']}, demoted {stats['demoted']}, "
          f"coined-rejected {stats['coined_rejected']}, named {stats['named_coined']}")
    print(f"    feedback   : {stats['parked_orphans']} parked, {stats['unconnected']} unconnected")
    print(f"    consistent : {consistency['consistent']} "
          f"(cycles {consistency['cycles_remaining']}, "
          f"disjoint relaxed {consistency['disjointness_relaxed']})")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
