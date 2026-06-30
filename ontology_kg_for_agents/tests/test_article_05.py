"""Tests that keep Article 5 honest.

Two contracts:
  1. Article 5 grows the *knowledge graph*, not the ontology. The target schema
     is the fixed SCIMA-OWL v0.6 produced by Article 4; this article asserts no
     new classes, properties, or axioms.
  2. The schema-first extractor plus the compliance gate behave as the article
     describes: every candidate triple gets exactly one of four verdicts
     (ADMIT, REPAIR, QUARANTINE, REJECT); off-schema relations, disjointness
     clashes, and domain violations are rejected; a coercible datatype mismatch
     is repaired; a structurally-valid but low-confidence triple is quarantined
     (compliance and confidence are separate gates); and the admitted A-Box is
     ontology-compliant by construction (SHACL conforms, reasoner consistent).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdflib import RDF, XSD, Literal, Namespace

from scima.ontology import ScimaOntology
from scima.kg_extraction import (
    KGExtractionPipeline,
    OntologyContract,
    Verdict,
    extract_candidates,
    _CONFIDENCE_FLOOR,
    _DEFAULT_CORPUS,
    _SHAPES_FILE,
)

SCIMA = Namespace("http://scima.city/ontology#")


@pytest.fixture(scope="module")
def result():
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    return KGExtractionPipeline(corpus).run()


@pytest.fixture(scope="module")
def contract():
    return OntologyContract("v0.6")


# ===================================================================
# Article 5 grows the KG, not the ontology
# ===================================================================

def test_target_schema_is_unchanged_v0_6():
    """The ontology is a fixed contract here: still 26 / 34 / 15."""
    s = ScimaOntology.load("v0.6").summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (26, 34, 15)


# ===================================================================
# The compliance gate: verdict tally
# ===================================================================

def test_extracts_twenty_five_candidates():
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    assert len(extract_candidates(corpus)) == 25


def test_verdict_counts(result):
    c = result.counts
    assert c["admit"] == 20
    assert c["repair"] == 1
    assert c["reject"] == 3
    assert c["quarantine"] == 1
    assert sum(c.values()) == 25


def test_every_candidate_gets_exactly_one_verdict(result):
    assert len(result.gated) == 25
    assert all(isinstance(g.verdict, Verdict) for g in result.gated)


# ===================================================================
# Specific verdicts the article calls out
# ===================================================================

def _verdict_for(result, subject, predicate, obj):
    for g in result.gated:
        cnd = g.candidate
        if (cnd.subject, cnd.predicate, cnd.obj) == (subject, predicate, obj):
            return g
    raise AssertionError(f"no candidate {subject} {predicate} {obj}")


def test_off_schema_relation_is_rejected(result):
    g = _verdict_for(result, "scima:IC_Diaz", "scima:authorizes", "scima:HMP_1")
    assert g.verdict == Verdict.REJECT
    assert "not in ontology" in g.reason


def test_disjointness_clash_is_rejected(result):
    """FD-12 is a FireDepartment; typing it HazmatTeam too is a disjoint clash."""
    g = _verdict_for(result, "scima:FD_12", "rdf:type", "scima:HazmatTeam")
    assert g.verdict == Verdict.REJECT
    assert "disjoint" in g.reason


def test_domain_violation_is_rejected(result):
    """A WaterMain cannot be dispatched; dispatchedTo's domain is ResponderUnit."""
    g = _verdict_for(result, "scima:WM_7B", "scima:dispatchedTo", "scima:Incident_I204")
    assert g.verdict == Verdict.REJECT
    assert "domain" in g.reason


def test_datatype_mismatch_is_repaired(result):
    g = _verdict_for(result, "scima:Reading_R1", "scima:observedValue", "47")
    assert g.verdict == Verdict.REPAIR
    assert g.repaired_value == Literal(47, datatype=XSD.integer)


def test_low_confidence_is_quarantined_not_rejected(result):
    """Compliance and confidence are separate gates: this triple is schema-valid
    but the extractor was not confident, so it is parked, not rejected."""
    g = _verdict_for(result, "scima:Incident_I205", "rdf:type", "scima:HazMatSpill")
    assert g.verdict == Verdict.QUARANTINE
    assert g.candidate.confidence < _CONFIDENCE_FLOOR


# ===================================================================
# The admitted A-Box
# ===================================================================

def test_admitted_box_has_expected_size(result):
    assert result.n_admitted_triples == 21      # 20 admitted + 1 repaired
    assert result.n_individuals == 10


def test_admitted_box_is_shacl_conformant(result):
    assert result.conforms is True
    assert result.shacl_violations == []


def test_admitted_box_is_reasoner_consistent(result):
    assert result.consistent is True


def test_rejected_and_quarantined_never_enter_the_graph(result):
    g = result.admitted_graph
    # the rejected disjoint type assertion is absent
    assert (SCIMA.FD_12, RDF.type, SCIMA.HazmatTeam) not in g
    # the quarantined second spill never typed
    assert (SCIMA.Incident_I205, RDF.type, SCIMA.HazMatSpill) not in g
    # the off-schema relation is absent
    assert (SCIMA.IC_Diaz, SCIMA.authorizes, SCIMA.HMP_1) not in g


def test_repaired_value_lands_as_integer_in_the_graph(result):
    val = result.admitted_graph.value(SCIMA.Reading_R1, SCIMA.observedValue)
    assert val == Literal(47, datatype=XSD.integer)


# ===================================================================
# The ontology contract
# ===================================================================

def test_contract_lifts_disjointness_through_hierarchy(contract):
    assert contract.disjoint("scima:HazmatTeam", "scima:FireDepartment")
    # InfrastructureEntity is disjoint with Agent, so subclasses clash too
    assert contract.disjoint("scima:WaterMain", "scima:IncidentCommander")


def test_contract_subsumption(contract):
    assert contract.is_a({"scima:HazMatSpill"}, "scima:Incident")
    assert contract.is_a({"scima:TrafficCamera"}, "scima:SensorDevice")
    assert not contract.is_a({"scima:WaterMain"}, "scima:ResponderUnit")


def test_contract_reads_object_property_domain_range(contract):
    dom, rng = contract.object_props["scima:dispatchedTo"]
    assert dom == "scima:ResponderUnit"
    assert rng == "scima:Incident"


def test_contract_reads_datatype_ranges(contract):
    assert contract.datatype_props["scima:observedValue"] == "xsd:integer"
    assert contract.datatype_props["scima:hasStatus"] == "xsd:string"


# ===================================================================
# Shapes artifact + emit
# ===================================================================

def test_shapes_file_parses():
    from rdflib import Graph
    g = Graph()
    g.parse(_SHAPES_FILE, format="turtle")
    assert len(g) > 0


def test_emit_round_trips(tmp_path, result):
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    pipe = KGExtractionPipeline(corpus)
    out = tmp_path / "kg.ttl"
    pipe.emit(result, out)
    from rdflib import Graph
    g = Graph()
    g.parse(out, format="turtle")
    assert len(g) == result.n_admitted_triples
