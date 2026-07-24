"""Serialize the admitted ontology to Turtle (optional --emit-ttl).

The article ships its admitted delta as OWL Turtle (`owl:Class`, `rdfs:subClassOf`,
`rdfs:label`, `skos:altLabel`, `owl:ObjectProperty` with `rdfs:domain`/`rdfs:range`,
`owl:disjointWith`). This module produces the same shape from the admitted set so
the reviewed ontology can be loaded into any OWL tool or a full reasoner.

Parked orphans are attached to a single domain-top class so they are not floating
roots; that is exactly the "parked under a domain top" disposition from Stage 4.
"""

from __future__ import annotations

from pathlib import Path


EX = "http://example.org/ontology#"
DOMAIN_TOP = "DomainEntity"   # the domain top that parked orphans hang from


def _local(node_id: str) -> str:
    """A safe local name from an element id (concept:wholenumber -> WholeNumber-ish)."""
    tail = node_id.split(":", 1)[-1]
    return "".join(p[:1].upper() + p[1:] for p in tail.replace("-", " ").split()) or "X"


def emit(path: Path, classes: dict, coined: dict, relationships: list,
         axioms: list, parents: dict, parked: set, labels: dict) -> None:
    """Write the admitted ontology as Turtle using rdflib."""
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, OWL, SKOS

    g = Graph()
    ex = Namespace(EX)
    g.bind("ex", ex); g.bind("owl", OWL); g.bind("rdfs", RDFS); g.bind("skos", SKOS)

    def uri(nid: str) -> URIRef:
        return ex[_local(nid)]

    top = ex[DOMAIN_TOP]
    g.add((top, RDF.type, OWL.Class))
    g.add((top, RDFS.label, Literal("Domain Entity")))

    # Classes (extracted + coined).
    for nid, node in {**classes, **coined}.items():
        u = uri(nid)
        g.add((u, RDF.type, OWL.Class))
        lab = labels.get(nid) or node.get("label", "")
        if lab:
            g.add((u, RDFS.label, Literal(lab)))
        for alt in node.get("alt_labels", [])[:12]:
            g.add((u, SKOS.altLabel, Literal(alt)))
        if node.get("coined"):
            g.add((u, RDFS.comment, Literal("coined parent (justified by children)")))
        for p in parents.get(nid, ()):
            g.add((u, RDFS.subClassOf, uri(p)))
        if nid in parked:
            g.add((u, RDFS.subClassOf, top))
            g.add((u, RDFS.comment, Literal("parked orphan; flagged for re-parenting")))

    # Relationships.
    for r in relationships:
        u = uri(r["id"])
        g.add((u, RDF.type, OWL.ObjectProperty))
        g.add((u, RDFS.label, Literal(r["label"])))
        if r.get("domain"):
            g.add((u, RDFS.domain, uri(r["domain"])))
        if r.get("range"):
            g.add((u, RDFS.range, uri(r["range"])))

    # Axioms (admitted disjointness).
    for ax in axioms:
        if ax.get("type") == "disjointWith" and len(ax.get("classes", [])) == 2:
            a, b = ax["classes"]
            g.add((uri(a), OWL.disjointWith, uri(b)))

    g.serialize(destination=str(path), format="turtle")
