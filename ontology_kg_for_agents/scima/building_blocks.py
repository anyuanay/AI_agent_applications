"""The four building blocks of an ontology, as plain Python.

Article 1 teaches that almost everything in an ontology is one of four
things: a class, an individual, a property, or an axiom. This module is a
small, readable Python mirror of those four blocks. It is not a
replacement for OWL or rdflib; it is a teaching scaffold that lets the
article's vocabulary be imported, inspected, and tested.

Each datatype property and object property is distinguished, matching the
OWL split between owl:DatatypeProperty and owl:ObjectProperty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PropertyKind(Enum):
    OBJECT = "object"      # links an individual to another individual
    DATATYPE = "datatype"  # links an individual to a literal value


class AxiomKind(Enum):
    SUBCLASS_OF = "subClassOf"
    DISJOINT_WITH = "disjointWith"
    SYMMETRIC = "SymmetricProperty"
    FUNCTIONAL = "FunctionalProperty"
    DOMAIN = "domain"
    RANGE = "range"


@dataclass(frozen=True)
class OntologyClass:
    """A category, a set of things that share a kind."""
    iri: str
    label: str = ""
    parent: str | None = None  # IRI of the superclass, if any


@dataclass(frozen=True)
class Individual:
    """A specific thing in the world, a member of one or more classes."""
    iri: str
    types: tuple[str, ...] = ()  # IRIs of the classes it belongs to


@dataclass(frozen=True)
class Property:
    """A relationship (object) or attribute (datatype)."""
    iri: str
    kind: PropertyKind
    domain: str | None = None  # IRI of the subject class
    range: str | None = None   # IRI of the object class or XSD datatype
    label: str = ""


@dataclass(frozen=True)
class Axiom:
    """A logical rule that constrains the model and enables inference."""
    kind: AxiomKind
    subject: str
    object: str | None = None


@dataclass
class Ontology:
    """An in-memory bag of the four building blocks."""
    classes: list[OntologyClass] = field(default_factory=list)
    individuals: list[Individual] = field(default_factory=list)
    properties: list[Property] = field(default_factory=list)
    axioms: list[Axiom] = field(default_factory=list)

    def object_properties(self) -> list[Property]:
        return [p for p in self.properties if p.kind is PropertyKind.OBJECT]

    def datatype_properties(self) -> list[Property]:
        return [p for p in self.properties if p.kind is PropertyKind.DATATYPE]

    def subclasses_of(self, parent_iri: str) -> list[OntologyClass]:
        return [c for c in self.classes if c.parent == parent_iri]


def scima_v0_1() -> Ontology:
    """SCIMA-OWL v0.1 expressed with the building-block scaffold.

    This is the same conceptual content as ontologies/scima_owl_v0_1.ttl,
    kept deliberately small for teaching. The Turtle file is the source of
    truth for tooling; this is the source of truth for the prose.
    """
    P = PropertyKind
    A = AxiomKind

    return Ontology(
        classes=[
            OntologyClass("scima:InfrastructureEntity", "Infrastructure Entity"),
            OntologyClass("scima:RoadSegment", "Road Segment", "scima:InfrastructureEntity"),
            OntologyClass("scima:TrafficLight", "Traffic Light", "scima:InfrastructureEntity"),
            OntologyClass("scima:PowerNode", "Power Node", "scima:InfrastructureEntity"),
            OntologyClass("scima:WaterMain", "Water Main", "scima:InfrastructureEntity"),
            OntologyClass("scima:SensorDevice", "Sensor Device"),
            OntologyClass("scima:SensorReading", "Sensor Reading"),
            OntologyClass("scima:Incident", "Incident"),
        ],
        properties=[
            Property("scima:locatedOn", P.OBJECT, "scima:TrafficLight", "scima:RoadSegment", "located on"),
            Property("scima:monitors", P.OBJECT, "scima:SensorDevice", "scima:InfrastructureEntity", "monitors"),
            Property("scima:hasReading", P.OBJECT, "scima:SensorDevice", "scima:SensorReading", "has reading"),
            Property("scima:affects", P.OBJECT, "scima:Incident", "scima:InfrastructureEntity", "affects"),
            Property("scima:connectedTo", P.OBJECT, "scima:RoadSegment", "scima:RoadSegment", "connected to"),
            Property("scima:poweredBy", P.OBJECT, "scima:InfrastructureEntity", "scima:PowerNode", "powered by"),
            Property("scima:hasSpeedLimit", P.DATATYPE, "scima:RoadSegment", "xsd:integer", "has speed limit"),
            Property("scima:hasIdentifier", P.DATATYPE, "scima:InfrastructureEntity", "xsd:string", "has identifier"),
            Property("scima:timestamp", P.DATATYPE, "scima:SensorReading", "xsd:dateTime", "timestamp"),
            Property("scima:confidenceScore", P.DATATYPE, "scima:SensorReading", "xsd:decimal", "confidence score"),
            Property("scima:hasStatus", P.DATATYPE, "scima:Incident", "xsd:string", "has status"),
            Property("scima:hasVoltage", P.DATATYPE, "scima:PowerNode", "xsd:decimal", "has voltage"),
        ],
        axioms=[
            Axiom(A.DISJOINT_WITH, "scima:RoadSegment", "scima:TrafficLight"),
            Axiom(A.DISJOINT_WITH, "scima:RoadSegment", "scima:PowerNode"),
            Axiom(A.DISJOINT_WITH, "scima:PowerNode", "scima:WaterMain"),
            Axiom(A.SYMMETRIC, "scima:connectedTo"),
            Axiom(A.FUNCTIONAL, "scima:hasIdentifier"),
        ],
    )


if __name__ == "__main__":
    o = scima_v0_1()
    print(
        f"SCIMA-OWL v0.1 (building blocks): "
        f"{len(o.classes)} classes, {len(o.properties)} properties, "
        f"{len(o.axioms)} axioms"
    )
