"""Extract an ontology-compliant knowledge graph from sources (Article 5).

Article 4 *learned the ontology* (SCIMA-OWL v0.6) from a procedures corpus: the
T-Box, the general kinds and relationships. Article 5 does the complementary
job. It takes that ontology as a **fixed, given contract** and extracts the
**instance-level knowledge graph** (the A-Box) from sources so that every
asserted triple conforms to it. The ontology is not grown here; the *graph* is.

The organizing claim of the article: extraction is not "pull out any triple you
can find," it is "populate SCIMA-OWL." The ontology drives extraction (it tells
the extractor which classes to type and which relations to fill), and it is then
enforced as a hard validation gate. A candidate triple gets one of four verdicts:

  ADMIT       conforms to the schema  -> enters the KG
  REPAIR      a coercible mismatch (datatype cast) -> repaired, then enters
  QUARANTINE  plausible but unvalidated (low confidence) -> staging, for review
  REJECT      off-schema relation/class, or a hard domain/range/disjointness
              violation -> never enters the KG

The KG that ships is therefore ontology-compliant *by construction*.

Two checking layers, two jobs (as in the article):
  * SHACL shapes  -- closed-world shape and cardinality validation generated
                     from the ontology. The standards-track shapes are shipped
                     in ``shapes/scima_shacl_v0_6.ttl``; this module runs the
                     equivalent checks natively (pyshacl is an optional dep, the
                     same way ``knowledge_graph.py`` computes haversine instead
                     of leaning on a GeoSPARQL ``geof:distance``).
  * Reasoner      -- open-world logical consistency: an individual may not hold
                     two disjoint classes; ranges may not contradict.

The "extractor" step is a *deterministic stub*, so the example is fast and
reproducible. It stands in for an ``anthropic``-backed extractor prompted with
the ontology's class and property catalog. The pipeline structure, the gate
verdicts, and the admitted A-Box are the real, testable contract.

Usage:
    python -m scima.kg_extraction --feed corpus/incident_report_I204.txt
    python -m scima.kg_extraction --shapes
    python -m scima.kg_extraction --emit build/scima_kg_I204.ttl
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = _ROOT / "corpus" / "incident_report_I204.txt"
_SHAPES_FILE = _ROOT / "shapes" / "scima_shacl_v0_6.ttl"
_DEFAULT_EMIT = _ROOT / "build" / "scima_kg_I204.ttl"

# A triple is admitted on its merits but quarantined if the extractor was not
# confident enough. Compliance and confidence are separate gates (see article).
_CONFIDENCE_FLOOR = 0.5

# Source reliability priors feed the confidence score; they do NOT gate
# compliance. An unreliable source can still produce a schema-valid triple.
SOURCE_PRIORS = {
    "gtfs_official": 0.97,
    "traffic_api": 0.89,
    "weather_service": 0.93,
    "iot_sensor_raw": 0.72,
    "incident_report": 0.81,
}


class Verdict(Enum):
    ADMIT = "admit"
    REPAIR = "repair"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass
class Candidate:
    """One extracted candidate triple, tagged with what it claims to be.

    ``kind`` routes the gate: ``type`` (rdf:type assertion), ``object`` (object
    property), ``datatype`` (datatype property). ``declared_datatype`` is the
    xsd type the extractor stamped on a literal; the gate compares it to the
    ontology's declared range and repairs a coercible mismatch.
    """
    subject: str
    predicate: str
    obj: str
    kind: str                       # type | object | datatype
    source: str = "incident_report"
    confidence: float = 0.9
    declared_datatype: str | None = None   # e.g. "xsd:string" (datatype kind only)


@dataclass
class Gated:
    """A candidate after the compliance gate: its verdict and why."""
    candidate: Candidate
    verdict: Verdict
    reason: str
    repaired_value: Literal | None = None


@dataclass
class ExtractionResult:
    """The outcome of one feed pass: the admitted A-Box plus the verdict ledger."""
    gated: list[Gated]
    admitted_graph: Graph
    conforms: bool                  # SHACL: admitted graph satisfies all shapes
    consistent: bool                # reasoner: no disjoint-class clash survives
    shacl_violations: list[str] = field(default_factory=list)

    # ---- verdict views ----
    def by(self, v: Verdict) -> list[Gated]:
        return [g for g in self.gated if g.verdict == v]

    @property
    def admitted(self) -> list[Gated]:
        return [g for g in self.gated if g.verdict in (Verdict.ADMIT, Verdict.REPAIR)]

    @property
    def counts(self) -> dict[str, int]:
        return {v.value: len(self.by(v)) for v in Verdict}

    @property
    def n_individuals(self) -> int:
        return len(set(self.admitted_graph.subjects(RDF.type, None)))

    @property
    def n_admitted_triples(self) -> int:
        return len(self.admitted_graph)


# =====================================================================
# The ontology as a contract: precompute everything the gate needs.
# =====================================================================
class OntologyContract:
    """A queryable view of the *given* ontology that the gate enforces.

    Precomputes the subclass closure, object-property domains/ranges, datatype
    ranges, and the disjointness relation (lifted through the class hierarchy)
    so each candidate can be checked in O(1)-ish time.
    """

    def __init__(self, version: str = "v0.6"):
        self.onto = ScimaOntology.load(version)
        self.version = version
        g = self.onto.graph

        self.classes: set[str] = set(self.onto.classes())

        # child -> direct parents, then transitive ancestor closure
        self._parents: dict[str, set[str]] = {c: set() for c in self.classes}
        for s, _, o in g.triples((None, RDFS.subClassOf, None)):
            cs, co = self._qn(s), self._qn(o)
            if cs in self._parents and co:
                self._parents[cs].add(co)
        self._ancestors: dict[str, set[str]] = {
            c: self._closure(c) for c in self.classes
        }

        # object property -> (domain qname, range qname)
        self.object_props: dict[str, tuple[str | None, str | None]] = {}
        for p in self.onto.object_properties():
            pr = self._to_ref(p)
            dom = self._first_qn(g.objects(pr, RDFS.domain))
            rng = self._first_qn(g.objects(pr, RDFS.range))
            self.object_props[p] = (dom, rng)

        # datatype property -> xsd range qname (e.g. "xsd:integer")
        self.datatype_props: dict[str, str | None] = {}
        for p in self.onto.datatype_properties():
            pr = self._to_ref(p)
            rng = next(iter(g.objects(pr, RDFS.range)), None)
            self.datatype_props[p] = self._xsd(rng) if rng is not None else None

        # disjointness pairs (frozenset of two qnames), declared directly
        self.disjoint_pairs: set[frozenset[str]] = {
            frozenset((self._qn(s), self._qn(o)))
            for s, _, o in g.triples((None, OWL.disjointWith, None))
        }

    # ---- membership / subsumption ----
    def is_a(self, asserted_types: set[str], target: str) -> bool:
        """True if any asserted type is ``target`` or a subclass of it."""
        return any(target == t or target in self._ancestors.get(t, set())
                   for t in asserted_types)

    def disjoint(self, a: str, b: str) -> bool:
        """True if classes a and b are disjoint, lifted through the hierarchy."""
        la = self._ancestors.get(a, set()) | {a}
        lb = self._ancestors.get(b, set()) | {b}
        return any(frozenset((x, y)) in self.disjoint_pairs for x in la for y in lb)

    def clashes(self, asserted_types: set[str], new_class: str) -> str | None:
        """Return an existing type that ``new_class`` is disjoint with, or None."""
        for t in asserted_types:
            if t != new_class and self.disjoint(t, new_class):
                return t
        return None

    # ---- datatype coercion ----
    @staticmethod
    def cast(value: str, xsd_qname: str) -> Literal | None:
        """Try to coerce a raw string to the ontology's declared datatype."""
        try:
            if xsd_qname == "xsd:integer":
                if re.fullmatch(r"-?\d+", value.strip()):
                    return Literal(int(value), datatype=XSD.integer)
            elif xsd_qname == "xsd:decimal":
                return Literal(float(value), datatype=XSD.decimal)
            elif xsd_qname == "xsd:dateTime":
                from datetime import datetime
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                return Literal(value, datatype=XSD.dateTime)
            elif xsd_qname == "xsd:string":
                return Literal(value, datatype=XSD.string)
        except (ValueError, TypeError):
            return None
        return None

    # ---- helpers ----
    def _closure(self, c: str) -> set[str]:
        seen, stack = set(), list(self._parents.get(c, set()))
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.add(p)
                stack.extend(self._parents.get(p, set()))
        return seen

    def _first_qn(self, refs) -> str | None:
        return self._qn(next(iter(refs), None))

    @staticmethod
    def _qn(ref) -> str | None:
        if ref is None:
            return None
        s = str(ref)
        return "scima:" + s[len(str(SCIMA)):] if s.startswith(str(SCIMA)) else s

    @staticmethod
    def _xsd(ref) -> str:
        s = str(ref)
        return "xsd:" + s.rsplit("#", 1)[-1] if "XMLSchema" in s else s

    @staticmethod
    def _to_ref(name: str) -> URIRef:
        return SCIMA[name.split(":", 1)[1]] if name.startswith("scima:") else URIRef(name)


# =====================================================================
# Stage 1: schema-first extraction (deterministic stub).
# =====================================================================
def extract_candidates(corpus: str) -> list[Candidate]:
    """Schema-first (closed) extraction: only classes and relations SCIMA-OWL
    declares are targeted. A real extractor would be an LLM prompted with the
    ontology's class and property catalog; this stub returns the fixed set of
    candidates the incident report yields, including a few that deliberately do
    not conform, so the gate has something to catch.

    The ``corpus`` argument anchors provenance and keeps the signature honest;
    the stub does not parse it token by token.
    """
    c = Candidate
    return [
        # ---- type assertions (the A-Box individuals) ----
        c("scima:Incident_I204", "rdf:type", "scima:HazMatSpill", "type", confidence=0.95),
        c("scima:IC_Diaz", "rdf:type", "scima:IncidentCommander", "type", confidence=0.93),
        c("scima:EZ_7", "rdf:type", "scima:EvacuationZone", "type", confidence=0.90),
        c("scima:HT_3", "rdf:type", "scima:HazmatTeam", "type", confidence=0.94),
        c("scima:FD_12", "rdf:type", "scima:FireDepartment", "type", confidence=0.92),
        c("scima:HMP_1", "rdf:type", "scima:HazardousMaterialProtocol", "type", confidence=0.88),
        c("scima:CAM_90", "rdf:type", "scima:TrafficCamera", "type", confidence=0.96),
        c("scima:WM_7B", "rdf:type", "scima:WaterMain", "type", confidence=0.91),
        c("scima:RoadSegment_Main_St_NB", "rdf:type", "scima:RoadSegment", "type", confidence=0.90),
        c("scima:Reading_R1", "rdf:type", "scima:SensorReading", "type", confidence=0.90),

        # ---- object-property relations ----
        c("scima:IC_Diaz", "scima:commands", "scima:Incident_I204", "object", confidence=0.92),
        c("scima:IC_Diaz", "scima:designates", "scima:EZ_7", "object", confidence=0.89),
        c("scima:HT_3", "scima:dispatchedTo", "scima:Incident_I204", "object", confidence=0.93),
        c("scima:FD_12", "scima:dispatchedTo", "scima:Incident_I204", "object", confidence=0.90),
        c("scima:HT_3", "scima:followsProtocol", "scima:HMP_1", "object", confidence=0.87),
        c("scima:CAM_90", "scima:monitors", "scima:RoadSegment_Main_St_NB", "object", confidence=0.94),
        c("scima:Incident_I204", "scima:affects", "scima:WM_7B", "object", confidence=0.88),
        c("scima:Reading_R1", "scima:recordedBy", "scima:CAM_90", "object", confidence=0.95),

        # ---- datatype-property assertions ----
        c("scima:Incident_I204", "scima:hasStatus", "active", "datatype",
          confidence=0.90, declared_datatype="xsd:string"),
        c("scima:WM_7B", "scima:hasIdentifier", "WM-7B", "datatype",
          confidence=0.90, declared_datatype="xsd:string"),
        # observedValue is declared xsd:integer; extractor stamped it as a string
        # -> REPAIR (cast "47" to an integer literal).
        c("scima:Reading_R1", "scima:observedValue", "47", "datatype",
          confidence=0.90, declared_datatype="xsd:string"),

        # ---- non-conforming candidates the gate must catch ----
        # off-schema relation: `authorizes` is not in SCIMA-OWL -> REJECT
        c("scima:IC_Diaz", "scima:authorizes", "scima:HMP_1", "object", confidence=0.80),
        # disjointness clash: FD-12 is already a FireDepartment, which is disjoint
        # with HazmatTeam -> REJECT (caught by the reasoner pass)
        c("scima:FD_12", "rdf:type", "scima:HazmatTeam", "type", confidence=0.55),
        # domain violation: a WaterMain cannot dispatch; dispatchedTo's domain is
        # ResponderUnit -> REJECT (entity-linking error)
        c("scima:WM_7B", "scima:dispatchedTo", "scima:Incident_I204", "object", confidence=0.60),
        # low-confidence, unconfirmed second spill: structurally fine -> QUARANTINE
        c("scima:Incident_I205", "rdf:type", "scima:HazMatSpill", "type", confidence=0.40),
    ]


# =====================================================================
# Stage 2: the compliance gate.
# =====================================================================
class KGExtractionPipeline:
    """Extract instance triples from a feed and gate them against the ontology.

    The ontology (``OntologyContract``) is the fixed target schema. Type
    assertions are processed first so relation and datatype checks can consult
    the admitted A-Box type index. Nothing reaches the KG without passing.
    """

    def __init__(self, corpus: str, version: str = "v0.6"):
        self.corpus = corpus
        self.contract = OntologyContract(version)

    def run(self) -> ExtractionResult:
        candidates = extract_candidates(self.corpus)
        gated = self._gate(candidates)
        graph = self._build_graph(gated)
        conforms, violations = self._shacl_validate(graph)
        consistent = self._reasoner_consistent(graph)
        return ExtractionResult(
            gated=gated,
            admitted_graph=graph,
            conforms=conforms,
            consistent=consistent,
            shacl_violations=violations,
        )

    # ---- the gate ----
    def _gate(self, candidates: list[Candidate]) -> list[Gated]:
        # type index of the admitted A-Box: individual -> set of asserted classes
        types: dict[str, set[str]] = {}
        # process type assertions first, in order, then the rest
        ordered = ([c for c in candidates if c.kind == "type"]
                   + [c for c in candidates if c.kind != "type"])
        out: list[Gated] = []
        for cand in ordered:
            g = self._verdict(cand, types)
            # only admitted/repaired type assertions extend the type index
            if cand.kind == "type" and g.verdict in (Verdict.ADMIT, Verdict.REPAIR):
                types.setdefault(cand.subject, set()).add(cand.obj)
            out.append(g)
        # restore original candidate order for a stable, readable trace
        order = {id(c): i for i, c in enumerate(candidates)}
        out.sort(key=lambda gg: order[id(gg.candidate)])
        return out

    def _verdict(self, cand: Candidate, types: dict[str, set[str]]) -> Gated:
        k = self.contract
        # confidence is a separate gate, applied only to otherwise-admissible triples
        low_conf = cand.confidence < _CONFIDENCE_FLOOR

        if cand.kind == "type":
            if cand.obj not in k.classes:
                return Gated(cand, Verdict.REJECT, f"class {cand.obj} not in ontology")
            clash = k.clashes(types.get(cand.subject, set()), cand.obj)
            if clash:
                return Gated(cand, Verdict.REJECT,
                             f"{cand.obj} disjoint with already-asserted {clash}")
            if low_conf:
                return Gated(cand, Verdict.QUARANTINE,
                             f"confidence {cand.confidence:.2f} < {_CONFIDENCE_FLOOR}")
            return Gated(cand, Verdict.ADMIT, "typed individual conforms")

        if cand.kind == "object":
            if cand.predicate not in k.object_props:
                return Gated(cand, Verdict.REJECT,
                             f"relation {cand.predicate} not in ontology")
            dom, rng = k.object_props[cand.predicate]
            s_types = types.get(cand.subject, set())
            o_types = types.get(cand.obj, set())
            if dom and not k.is_a(s_types, dom):
                return Gated(cand, Verdict.REJECT,
                             f"domain violation: subject is not a {dom}")
            if rng and not k.is_a(o_types, rng):
                # disjoint with the range -> hard reject; merely unproven -> quarantine
                if any(k.disjoint(t, rng) for t in o_types):
                    return Gated(cand, Verdict.REJECT,
                                 f"range conflict: object disjoint with {rng}")
                return Gated(cand, Verdict.QUARANTINE,
                             f"range unproven: object not known to be a {rng}")
            if low_conf:
                return Gated(cand, Verdict.QUARANTINE,
                             f"confidence {cand.confidence:.2f} < {_CONFIDENCE_FLOOR}")
            return Gated(cand, Verdict.ADMIT, "relation conforms")

        if cand.kind == "datatype":
            if cand.predicate not in k.datatype_props:
                return Gated(cand, Verdict.REJECT,
                             f"datatype property {cand.predicate} not in ontology")
            expected = k.datatype_props[cand.predicate]
            if cand.declared_datatype == expected:
                if low_conf:
                    return Gated(cand, Verdict.QUARANTINE,
                                 f"confidence {cand.confidence:.2f} < {_CONFIDENCE_FLOOR}")
                lit = OntologyContract.cast(cand.obj, expected)
                return Gated(cand, Verdict.ADMIT, "literal conforms", repaired_value=lit)
            # mismatch: try to coerce to the declared range
            lit = OntologyContract.cast(cand.obj, expected)
            if lit is not None:
                return Gated(cand, Verdict.REPAIR,
                             f"cast {cand.declared_datatype}->{expected}",
                             repaired_value=lit)
            return Gated(cand, Verdict.REJECT,
                         f"datatype {cand.declared_datatype} not coercible to {expected}")

        return Gated(cand, Verdict.REJECT, f"unknown candidate kind {cand.kind}")

    # ---- build the admitted A-Box ----
    def _build_graph(self, gated: list[Gated]) -> Graph:
        g = Graph()
        g.bind("scima", SCIMA)
        for item in gated:
            if item.verdict not in (Verdict.ADMIT, Verdict.REPAIR):
                continue
            cand = item.candidate
            s = self._ref(cand.subject)
            if cand.kind == "type":
                g.add((s, RDF.type, self._ref(cand.obj)))
            elif cand.kind == "object":
                g.add((self._ref(cand.subject), self._ref(cand.predicate),
                       self._ref(cand.obj)))
            elif cand.kind == "datatype":
                lit = item.repaired_value or Literal(cand.obj)
                g.add((s, self._ref(cand.predicate), lit))
        return g

    # ---- SHACL-style validation (native; pyshacl optional) ----
    def _shacl_validate(self, graph: Graph) -> tuple[bool, list[str]]:
        """Validate the admitted graph against shapes generated from the ontology.

        Equivalent to running ``shapes/scima_shacl_v0_6.ttl`` with pyshacl. Two
        shape families are checked here: object-property ranges (sh:class) and a
        cardinality shape (an IncidentCommander commands at least one Incident).
        """
        k = self.contract
        violations: list[str] = []
        node_types: dict[URIRef, set[str]] = {}
        for s, _, o in graph.triples((None, RDF.type, None)):
            node_types.setdefault(s, set()).add(self.contract._qn(o))

        # property-shape: every object-property edge satisfies its declared range
        for s, p, o in graph:
            pq = self.contract._qn(p)
            if pq in k.object_props and isinstance(o, URIRef):
                _, rng = k.object_props[pq]
                if rng and not k.is_a(node_types.get(o, set()), rng):
                    violations.append(f"{self.contract._qn(s)} {pq} -> range {rng} unmet")

        # cardinality shape: an IncidentCommander must command >= 1 Incident
        for node, classes in node_types.items():
            if k.is_a(classes, "scima:IncidentCommander"):
                has_cmd = any(
                    True for _ in graph.objects(node, self._ref("scima:commands"))
                )
                if not has_cmd:
                    violations.append(f"{self.contract._qn(node)} commands minCount 1 unmet")

        return (len(violations) == 0, violations)

    # ---- reasoner consistency (native; owlrl optional) ----
    def _reasoner_consistent(self, graph: Graph) -> bool:
        """No individual in the admitted graph holds two disjoint classes."""
        node_types: dict[URIRef, set[str]] = {}
        for s, _, o in graph.triples((None, RDF.type, None)):
            node_types.setdefault(s, set()).add(self.contract._qn(o))
        for classes in node_types.values():
            cl = list(classes)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if self.contract.disjoint(cl[i], cl[j]):
                        return False
        return True

    def emit(self, result: ExtractionResult, path: Path = _DEFAULT_EMIT) -> Graph:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.admitted_graph.serialize(destination=str(path), format="turtle")
        return result.admitted_graph

    @staticmethod
    def _ref(name: str) -> URIRef:
        if name == "rdf:type":
            return RDF.type
        if name.startswith("scima:"):
            return SCIMA[name.split(":", 1)[1]]
        return URIRef(name)


# =====================================================================
# CLI
# =====================================================================
def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract an ontology-compliant KG from a feed (Article 5).")
    parser.add_argument("--feed", metavar="PATH", default=str(_DEFAULT_CORPUS),
                        help="source feed to extract from")
    parser.add_argument("--shapes", action="store_true",
                        help="print the path to the SHACL shapes file and exit")
    parser.add_argument("--emit", metavar="PATH", nargs="?", const=str(_DEFAULT_EMIT),
                        help="serialize the admitted A-Box to Turtle")
    args = parser.parse_args(argv)

    if args.shapes:
        print(f"SHACL shapes (standards-track artifact): {_SHAPES_FILE}")
        return

    corpus = Path(args.feed).read_text(encoding="utf-8")
    pipe = KGExtractionPipeline(corpus)
    onto = pipe.contract.onto.summary()
    print(f"Loaded {onto} (fixed target schema)")

    result = pipe.run()
    c = result.counts
    n = len(result.gated)
    print(f"Extracted {n} candidate triples from 1 feed (incident_report).")
    print(f"Compliance gate: {c['admit']} admitted, {c['repair']} repaired, "
          f"{c['reject']} rejected, {c['quarantine']} quarantined.")
    print("-" * 72)
    sym = {Verdict.ADMIT: "ADMIT ", Verdict.REPAIR: "REPAIR",
           Verdict.QUARANTINE: "QUARN ", Verdict.REJECT: "REJECT"}
    for g in result.gated:
        cand = g.candidate
        triple = f"{cand.subject} {cand.predicate} {cand.obj}"
        print(f"  [{sym[g.verdict]}] {triple:62s} {g.reason}")
    print("-" * 72)
    print(f"Admitted A-Box: {result.n_admitted_triples} triples, "
          f"{result.n_individuals} typed individuals.")
    print(f"SHACL validation over admitted graph: conforms = {result.conforms} "
          f"({len(result.shacl_violations)} violations).")
    print(f"Reasoner consistency: consistent = {result.consistent}.")

    if args.emit:
        pipe.emit(result, Path(args.emit))
        print(f"Wrote admitted A-Box to {args.emit}")


if __name__ == "__main__":
    _cli()
