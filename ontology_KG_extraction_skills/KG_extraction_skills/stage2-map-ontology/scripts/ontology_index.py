"""Parse an OWL/Turtle ontology and build the pre-computed index.

Extracts:
  classes      — IRI, label, aliases, comment, superclasses, equivalent classes
  properties   — IRI, label, aliases, kind, domain, range, inverse info,
                  functional/inverse-functional flags, max cardinality
  subclass_closure  — transitive superclass set per class (used by Stage 3
                       domain/range checks)
  disjointness_pairs — all disjoint class pairs (owl:disjointWith +
                        owl:AllDisjointClasses)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import rdflib
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef
from rdflib.namespace import SKOS, XSD


# ── helpers ──────────────────────────────────────────────────────────────────

def _local(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    if "/" in iri:
        return iri.split("/")[-1]
    return iri


def _str_label(g: rdflib.Graph, node: URIRef, fallback: str = "") -> str:
    val = g.value(node, RDFS.label)
    return str(val) if val else fallback


def _all_labels(g: rdflib.Graph, node: URIRef) -> list[str]:
    """Primary label + all skos:altLabel strings."""
    labels: list[str] = []
    primary = g.value(node, RDFS.label)
    if primary:
        labels.append(str(primary))
    for alt in g.objects(node, SKOS.altLabel):
        labels.append(str(alt))
    return labels


def _comment(g: rdflib.Graph, node: URIRef) -> Optional[str]:
    val = g.value(node, RDFS.comment)
    return str(val) if val else None


# ── namespace detection ───────────────────────────────────────────────────────

def _detect_namespace(g: rdflib.Graph) -> str:
    for ont in g.subjects(RDF.type, OWL.Ontology):
        if isinstance(ont, URIRef):
            s = str(ont)
            return s if s.endswith("#") else s + "#"
    # Count URI prefixes across all subjects
    ns_counts: dict[str, int] = {}
    for s in g.subjects():
        if isinstance(s, URIRef):
            u = str(s)
            if "#" in u:
                ns = u[: u.rfind("#") + 1]
                ns_counts[ns] = ns_counts.get(ns, 0) + 1
    return max(ns_counts, key=ns_counts.get) if ns_counts else "http://example.org/ontology#"


# ── class extraction ──────────────────────────────────────────────────────────

def _extract_classes(g: rdflib.Graph) -> list[dict]:
    classes: list[dict] = []
    seen: set[str] = set()

    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        iri = str(cls)
        if iri in seen:
            continue
        seen.add(iri)

        label = _str_label(g, cls, _local(iri))
        aliases = [str(a) for a in g.objects(cls, SKOS.altLabel)]
        superclasses = [
            str(sup)
            for sup in g.objects(cls, RDFS.subClassOf)
            if isinstance(sup, URIRef)
        ]
        equiv = [
            str(eq)
            for eq in g.objects(cls, OWL.equivalentClass)
            if isinstance(eq, URIRef)
        ]

        classes.append({
            "iri": iri,
            "local_name": _local(iri),
            "label": label,
            "aliases": aliases,
            "comment": _comment(g, cls),
            "superclasses": superclasses,
            "equivalent_classes": equiv,
        })

    return classes


# ── property extraction ───────────────────────────────────────────────────────

def _max_cardinality(g: rdflib.Graph, prop: URIRef) -> Optional[int]:
    """Smallest owl:maxCardinality or owl:exactCardinality declared on any restriction for prop."""
    best: Optional[int] = None
    for _, _, rest in g.triples((None, OWL.onProperty, prop)):
        for card_pred in (OWL.maxCardinality, OWL.exactCardinality):
            val = g.value(rest, card_pred)
            if val is not None:
                try:
                    n = int(val)
                    if best is None or n < best:
                        best = n
                except (ValueError, TypeError):
                    pass
    return best


def _extract_properties(g: rdflib.Graph) -> list[dict]:
    seen: set[str] = set()
    props: list[dict] = []

    # Pre-index functional + inverse-functional
    functional_iris = {
        str(p)
        for p in g.subjects(RDF.type, OWL.FunctionalProperty)
        if isinstance(p, URIRef)
    }
    inv_functional_iris = {
        str(p)
        for p in g.subjects(RDF.type, OWL.InverseFunctionalProperty)
        if isinstance(p, URIRef)
    }

    # Build inverse-of map (bidirectional)
    inverse_of_map: dict[str, str] = {}
    for p, q in g.subject_objects(OWL.inverseOf):
        if isinstance(p, URIRef) and isinstance(q, URIRef):
            inverse_of_map[str(p)] = str(q)
            inverse_of_map[str(q)] = str(p)

    def _add(prop: URIRef, kind: str) -> None:
        iri = str(prop)
        if iri in seen:
            return
        seen.add(iri)

        label = _str_label(g, prop, _local(iri))
        aliases = [str(a) for a in g.objects(prop, SKOS.altLabel)]

        domain_node = g.value(prop, RDFS.domain)
        range_node  = g.value(prop, RDFS.range)
        domain = str(domain_node) if isinstance(domain_node, URIRef) else None
        range_ = str(range_node)  if isinstance(range_node,  URIRef) else (
                 str(range_node)  if isinstance(range_node,  URIRef) else
                 str(range_node)  if range_node is not None else None
        )
        # Capture XSD ranges too (for datatype properties)
        if range_ is None and range_node is not None:
            range_ = str(range_node)

        inv_iri = inverse_of_map.get(iri)
        inv_labels: list[str] = []
        inv_aliases: list[str] = []
        if inv_iri:
            inv_prop_node = URIRef(inv_iri)
            inv_label = _str_label(g, inv_prop_node, _local(inv_iri))
            inv_labels = [inv_label]
            inv_aliases = [str(a) for a in g.objects(inv_prop_node, SKOS.altLabel)]

        props.append({
            "iri":                  iri,
            "local_name":           _local(iri),
            "label":                label,
            "aliases":              aliases,
            "comment":              _comment(g, prop),
            "kind":                 kind,
            "domain":               domain,
            "range":                range_,
            "is_functional":        iri in functional_iris,
            "is_inverse_functional": iri in inv_functional_iris,
            "inverse_of":           inv_iri,
            "inverse_labels":       inv_labels,
            "inverse_aliases":      inv_aliases,
            "max_cardinality":      _max_cardinality(g, prop),
        })

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(prop, URIRef):
            _add(prop, "object_property")
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(prop, URIRef):
            _add(prop, "datatype_property")

    return props


# ── subclass closure ──────────────────────────────────────────────────────────

def _subclass_closure(g: rdflib.Graph) -> dict[str, list[str]]:
    """Transitive closure of rdfs:subClassOf (+ owl:equivalentClass as mutual subclassing)."""
    direct: dict[str, set[str]] = {}

    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef) and isinstance(sup, URIRef):
            direct.setdefault(str(sub), set()).add(str(sup))

    for c1, c2 in g.subject_objects(OWL.equivalentClass):
        if isinstance(c1, URIRef) and isinstance(c2, URIRef):
            direct.setdefault(str(c1), set()).add(str(c2))
            direct.setdefault(str(c2), set()).add(str(c1))

    closure: dict[str, list[str]] = {}

    def _ancestors(iri: str) -> set[str]:
        if iri in closure:
            return set(closure[iri])
        visited: set[str] = set()
        queue = list(direct.get(iri, []))
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            queue.extend(direct.get(cur, []))
        closure[iri] = sorted(visited)
        return visited

    all_iris = set(direct.keys())
    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef):
            all_iris.add(str(sub))
        if isinstance(sup, URIRef):
            all_iris.add(str(sup))

    for iri in all_iris:
        _ancestors(iri)

    return closure


# ── disjointness ──────────────────────────────────────────────────────────────

def _disjointness_pairs(g: rdflib.Graph) -> list[list[str]]:
    pairs: list[list[str]] = []
    seen: set[frozenset] = set()

    def _add(a: str, b: str) -> None:
        key: frozenset = frozenset([a, b])
        if key not in seen:
            seen.add(key)
            pairs.append([a, b])

    for c1, c2 in g.subject_objects(OWL.disjointWith):
        if isinstance(c1, URIRef) and isinstance(c2, URIRef):
            _add(str(c1), str(c2))

    for node in g.subjects(RDF.type, OWL.AllDisjointClasses):
        members_node = g.value(node, OWL.members)
        if members_node is None:
            continue
        members = [str(m) for m in g.items(members_node) if isinstance(m, URIRef)]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                _add(members[i], members[j])

    return pairs


# ── public entry point ────────────────────────────────────────────────────────

def build(ontology_path: Path) -> dict:
    """Parse ontology_path and return the full pre-computed index dict."""
    g = rdflib.Graph()
    g.parse(str(ontology_path))

    return {
        "ontology_path": str(ontology_path),
        "namespace":     _detect_namespace(g),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "classes":       _extract_classes(g),
        "properties":    _extract_properties(g),
        "subclass_closure":    _subclass_closure(g),
        "disjointness_pairs":  _disjointness_pairs(g),
    }
