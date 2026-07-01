"""Verification checks for Stage 3.

Checker runs five checks on each mapped triple in sorted order and maintains
running state (admitted types, cardinality counters, inverse-functional
tracker) so each triple is checked against all previously admitted triples.
"""
from __future__ import annotations

from typing import Optional

import datatype_repair


class VerificationState:
    """Accumulated state from admitted triples."""

    def __init__(self) -> None:
        # individual_iri -> set of class IRIs currently admitted
        self.admitted_types: dict[str, set[str]] = {}
        # (subject_iri, predicate_iri) -> count of admitted triples
        self.cardinality_counter: dict[tuple[str, str], int] = {}
        # (predicate_iri, object_iri) -> first subject_iri admitted
        self.inv_func_tracker: dict[tuple[str, str], str] = {}

    def admit(self, triple: dict) -> None:
        subj     = triple["subject_iri"]
        subj_type = triple["subject_type"]
        pred     = triple["predicate_iri"]

        self.admitted_types.setdefault(subj, set()).add(subj_type)
        key = (subj, pred)
        self.cardinality_counter[key] = self.cardinality_counter.get(key, 0) + 1

        if triple.get("predicate_kind") == "object_property":
            obj      = triple.get("object_iri")
            obj_type = triple.get("object_type")
            if obj and obj_type:
                self.admitted_types.setdefault(obj, set()).add(obj_type)
                self.inv_func_tracker.setdefault((pred, obj), subj)


class Checker:
    """Runs all 5 verification checks against the ontology index."""

    def __init__(self, ontology_index: dict) -> None:
        self._props   = {p["iri"]: p for p in ontology_index.get("properties", [])}
        self._closure: dict[str, list[str]] = ontology_index.get("subclass_closure", {})
        self._state   = VerificationState()

        # disjointness_map: class_iri -> set of IRIs declared disjoint with it
        self._disj_map: dict[str, set[str]] = {}
        for pair in ontology_index.get("disjointness_pairs", []):
            a, b = pair[0], pair[1]
            self._disj_map.setdefault(a, set()).add(b)
            self._disj_map.setdefault(b, set()).add(a)

        self.stats = {
            "domain_violations":       0,
            "range_violations":        0,
            "datatype_mismatches":     0,
            "datatype_repairs":        0,
            "disjointness_violations": 0,
            "cardinality_violations":  0,
        }

    @property
    def state(self) -> VerificationState:
        return self._state

    def _is_subclass(self, type_iri: str, ancestor_iri: str) -> bool:
        """True if type_iri is identical to ancestor_iri or transitively a subclass."""
        if type_iri == ancestor_iri:
            return True
        return ancestor_iri in self._closure.get(type_iri, [])

    def _disjointness_violated(self, individual_iri: str, new_type: str) -> bool:
        """True if assigning new_type to individual_iri conflicts with admitted types."""
        current = self._state.admitted_types.get(individual_iri, set())
        disjoints_of_new = self._disj_map.get(new_type, set())
        return bool(current & disjoints_of_new)

    def check(self, triple: dict) -> tuple[str, Optional[str], dict]:
        """
        Run all 5 checks on a mapped triple.

        Returns (verdict, reason, triple_out) where:
          verdict  : "admit" | "repaired" | "reject"
          reason   : None on success; a string on reject
          triple_out: possibly mutated copy (repaired datatype values)
        """
        t         = dict(triple)
        pred_iri  = t["predicate_iri"]
        prop      = self._props.get(pred_iri)
        pred_kind = t.get("predicate_kind", "object_property")
        repair_happened = False

        # ── Check 1: Domain ───────────────────────────────────────────────
        if prop:
            domain = prop.get("domain")
            if domain and not self._is_subclass(t["subject_type"], domain):
                self.stats["domain_violations"] += 1
                return "reject", "domain_violation", t

        # ── Check 2: Range (object_property) ──────────────────────────────
        if prop and pred_kind == "object_property":
            rng = prop.get("range")
            if rng:
                obj_type = t.get("object_type")
                if obj_type and not self._is_subclass(obj_type, rng):
                    self.stats["range_violations"] += 1
                    return "reject", "range_violation", t

        # ── Check 3: Range (datatype_property) ────────────────────────────
        if prop and pred_kind == "datatype_property":
            rng = prop.get("range")
            ok, val, xsd_type, rv = datatype_repair.check_and_repair(
                t.get("object_literal"),
                t.get("object_datatype"),
                rng,
            )
            if not ok:
                self.stats["datatype_mismatches"] += 1
                return "reject", "datatype_mismatch", t
            if rv == "repaired":
                t["object_literal"]  = val
                t["object_datatype"] = xsd_type
                repair_happened = True
                self.stats["datatype_repairs"] += 1

        # ── Check 4: Disjointness ─────────────────────────────────────────
        if self._disjointness_violated(t["subject_iri"], t["subject_type"]):
            self.stats["disjointness_violations"] += 1
            return "reject", "disjointness_violation", t

        if pred_kind == "object_property" and t.get("object_iri") and t.get("object_type"):
            if self._disjointness_violated(t["object_iri"], t["object_type"]):
                self.stats["disjointness_violations"] += 1
                return "reject", "disjointness_violation", t

        # ── Check 5a: Functional / max-cardinality ────────────────────────
        if prop:
            max_card: Optional[int] = None
            if prop.get("is_functional"):
                max_card = 1
            elif prop.get("max_cardinality") is not None:
                max_card = int(prop["max_cardinality"])
            if max_card is not None:
                current = self._state.cardinality_counter.get(
                    (t["subject_iri"], pred_iri), 0
                )
                if current >= max_card:
                    self.stats["cardinality_violations"] += 1
                    return "reject", "cardinality_violation", t

        # ── Check 5b: Inverse-functional ─────────────────────────────────
        if prop and prop.get("is_inverse_functional") and pred_kind == "object_property":
            obj_iri = t.get("object_iri")
            if obj_iri:
                existing_subj = self._state.inv_func_tracker.get((pred_iri, obj_iri))
                if existing_subj and existing_subj != t["subject_iri"]:
                    self.stats["cardinality_violations"] += 1
                    return "reject", "cardinality_violation", t

        # ── Admit ─────────────────────────────────────────────────────────
        self._state.admit(t)
        return "repaired" if repair_happened else "admit", None, t
