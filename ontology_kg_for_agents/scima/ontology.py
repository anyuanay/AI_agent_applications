"""Load and inspect versioned SCIMA-OWL ontologies.

Article 1 introduces SCIMA-OWL v0.1 as schema only (no instances). This
module wraps rdflib so the rest of the codebase, and the tests that keep
the articles honest, can ask structural questions about a given version:
how many classes, which subclasses, which properties, which axioms.

Usage:
    from scima.ontology import ScimaOntology
    onto = ScimaOntology.load("v0.1")
    print(onto.summary())
    onto.subclasses_of("scima:InfrastructureEntity")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import Namespace

SCIMA = Namespace("http://scima.city/ontology#")

_ROOT = Path(__file__).resolve().parent.parent
_ONTOLOGY_DIR = _ROOT / "ontologies"

# version label -> filename
_VERSION_FILES = {
    "v0.1": "scima_owl_v0_1.ttl",
    "v0.2": "scima_owl_v0_2.ttl",
    "v0.5": "scima_owl_v0_5.ttl",
}


@dataclass
class OntologySummary:
    version: str
    n_classes: int
    n_object_properties: int
    n_datatype_properties: int
    n_axioms: int

    @property
    def n_properties(self) -> int:
        return self.n_object_properties + self.n_datatype_properties

    def __str__(self) -> str:
        return (
            f"SCIMA-OWL {self.version}: "
            f"{self.n_classes} classes, "
            f"{self.n_properties} properties "
            f"({self.n_object_properties} object + {self.n_datatype_properties} datatype), "
            f"{self.n_axioms} axioms"
        )


class ScimaOntology:
    """A thin, queryable wrapper around one SCIMA-OWL version."""

    def __init__(self, graph: Graph, version: str):
        self.graph = graph
        self.version = version

    # ---- loading ------------------------------------------------------
    @classmethod
    def load(cls, version: str = "v0.1") -> "ScimaOntology":
        if version not in _VERSION_FILES:
            raise ValueError(
                f"Unknown SCIMA-OWL version {version!r}. "
                f"Known versions: {sorted(_VERSION_FILES)}"
            )
        path = _ONTOLOGY_DIR / _VERSION_FILES[version]
        graph = Graph()
        graph.parse(path, format="turtle")
        return cls(graph, version)

    # ---- structural queries ------------------------------------------
    def classes(self) -> list[str]:
        return sorted(self._qnames(self.graph.subjects(RDF.type, OWL.Class)))

    def object_properties(self) -> list[str]:
        return sorted(self._qnames(self.graph.subjects(RDF.type, OWL.ObjectProperty)))

    def datatype_properties(self) -> list[str]:
        return sorted(self._qnames(self.graph.subjects(RDF.type, OWL.DatatypeProperty)))

    def subclasses_of(self, parent: str) -> list[str]:
        """Direct subclasses of the given class (qname or full IRI)."""
        parent_ref = self._to_ref(parent)
        return sorted(
            self._qnames(self.graph.subjects(RDFS.subClassOf, parent_ref))
        )

    def axiom_count(self) -> int:
        """Count the 'beyond the basics' axioms that give the ontology its
        reasoning power: disjointness statements plus property
        characteristics (symmetric, functional, transitive, ...).

        Plain subClassOf / domain / range triples are not counted here; in
        the Growth Tracker they are folded into the class and property
        counts.
        """
        disjoint = len(list(self.graph.triples((None, OWL.disjointWith, None))))
        characteristic_types = [
            OWL.SymmetricProperty,
            OWL.FunctionalProperty,
            OWL.TransitiveProperty,
            OWL.InverseFunctionalProperty,
            OWL.ReflexiveProperty,
            OWL.IrreflexiveProperty,
            OWL.AsymmetricProperty,
        ]
        characteristics = sum(
            len(list(self.graph.triples((None, RDF.type, t))))
            for t in characteristic_types
        )
        return disjoint + characteristics

    def summary(self) -> OntologySummary:
        return OntologySummary(
            version=self.version,
            n_classes=len(self.classes()),
            n_object_properties=len(self.object_properties()),
            n_datatype_properties=len(self.datatype_properties()),
            n_axioms=self.axiom_count(),
        )

    # ---- helpers ------------------------------------------------------
    def _to_ref(self, name: str) -> URIRef:
        if name.startswith("scima:"):
            return SCIMA[name.split(":", 1)[1]]
        if name.startswith("http"):
            return URIRef(name)
        return SCIMA[name]

    @staticmethod
    def _qnames(refs) -> list[str]:
        out = []
        for r in refs:
            if isinstance(r, URIRef) and str(r).startswith(str(SCIMA)):
                out.append("scima:" + str(r)[len(str(SCIMA)):])
        return out


if __name__ == "__main__":
    onto = ScimaOntology.load("v0.1")
    print(onto.summary())
    print("  classes:", ", ".join(onto.classes()))
    print(
        "  infrastructure subclasses:",
        ", ".join(onto.subclasses_of("scima:InfrastructureEntity")),
    )
