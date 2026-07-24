"""Tests that keep Article 5 honest.

Two contracts:
  1. SCIMA-OWL v0.8 schema matches the Growth Tracker (cumulative): 30 classes,
     41 properties (23 object + 18 datatype), 18 axioms. v0.6 backward-compatible.
     Key change: scima:commands range updated to scima:ResponderUnit and made
     owl:FunctionalProperty; scima:observedValue and scima:reportedAt also functional.
  2. The three-stage KG extraction pipeline over Incident Report I-204 behaves
     as the article describes: Stage 1 surfaces 5 candidate triples + 1 negated;
     Stage 2 maps all 5 to ontology IRIs at high confidence; Stage 3 admits 2,
     repairs 1, and rejects 2 (domain_violation + cardinality_violation). Four
     rdf:type assertions are emitted after deduplication.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdflib import OWL, RDF, RDFS, Namespace

from scima.ontology import ScimaOntology
from scima.kg_extraction import (
    KGExtractionPipeline,
    _DEFAULT_CORPUS,
    _build_ontology_index,
)

SCIMA = Namespace("http://scima.city/ontology#")

# ---- Growth Tracker targets for v0.8 (cumulative) ----
EXPECTED_CLASSES = 30
EXPECTED_OBJ_PROPS = 23
EXPECTED_DT_PROPS = 18
EXPECTED_PROPS = 41
EXPECTED_AXIOMS = 18


# ===================================================================
# Schema: SCIMA-OWL v0.8
# ===================================================================

@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.8")


def test_v0_8_parses(onto):
    assert len(onto.graph) > 0


def test_v0_8_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_v0_8_property_split(onto):
    s = onto.summary()
    assert s.n_object_properties == EXPECTED_OBJ_PROPS
    assert s.n_datatype_properties == EXPECTED_DT_PROPS
    assert s.n_properties == EXPECTED_PROPS


def test_v0_8_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_v0_8_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES, EXPECTED_PROPS, EXPECTED_AXIOMS,
    )


def test_v0_8_is_backward_compatible_with_v0_6():
    """Every v0.6 class survives into v0.8."""
    v06 = set(ScimaOntology.load("v0.6").classes())
    v08 = set(ScimaOntology.load("v0.8").classes())
    assert v06 <= v08


def test_v0_8_new_classes_present(onto):
    classes = set(onto.classes())
    assert "scima:IncidentReport" in classes
    assert "scima:PressureObservation" in classes
    assert "scima:TemperatureObservation" in classes
    assert "scima:ControlZone" in classes


def test_v0_8_pressure_and_temperature_under_sensor_reading(onto):
    kids = set(onto.subclasses_of("scima:SensorReading"))
    assert "scima:PressureObservation" in kids
    assert "scima:TemperatureObservation" in kids


def test_v0_8_commands_range_is_responder_unit(onto):
    """In v0.8, commands range was updated from Incident to ResponderUnit."""
    commands_iri = SCIMA.commands
    ranges = list(onto.graph.objects(commands_iri, RDFS.range))
    assert len(ranges) == 1
    assert str(ranges[0]) == str(SCIMA.ResponderUnit)


def test_v0_8_functional_properties(onto):
    """commands, observedValue, and reportedAt must be FunctionalProperty in v0.8."""
    func_props = {
        str(p).split("#")[-1]
        for p in onto.graph.subjects(RDF.type, OWL.FunctionalProperty)
    }
    assert "commands" in func_props
    assert "observedValue" in func_props
    assert "reportedAt" in func_props


def test_v0_8_new_datatype_properties(onto):
    dt_props = set(onto.datatype_properties())
    assert "scima:reportedAt" in dt_props
    assert "scima:hasUnit" in dt_props
    assert "scima:measuredAt" in dt_props


def test_v0_8_new_object_properties(onto):
    obj_props = set(onto.object_properties())
    assert "scima:reportedBy" in obj_props
    assert "scima:relatesToIncident" in obj_props
    assert "scima:controlledBy" in obj_props
    assert "scima:extractedFrom" in obj_props


# ===================================================================
# Ontology Index
# ===================================================================

@pytest.fixture(scope="module")
def index(onto):
    return _build_ontology_index(onto.graph)


def test_index_subclass_closure_hazmatteam(index):
    """HazmatTeam closure must include ResponderUnit."""
    closure = index.subclass_closure.get("HazmatTeam", set())
    assert "HazmatTeam" in closure
    assert "ResponderUnit" in closure


def test_index_subclass_closure_watermain(index):
    """WaterMain closure must NOT include ResponderUnit."""
    closure = index.subclass_closure.get("WaterMain", set())
    assert "WaterMain" in closure
    assert "ResponderUnit" not in closure


def test_index_subclass_closure_hazmatspill(index):
    closure = index.subclass_closure.get("HazMatSpill", set())
    assert "HazMatSpill" in closure
    assert "Incident" in closure


def test_index_commands_is_functional(index):
    assert "commands" in index.functional_props


def test_index_observed_value_is_functional(index):
    assert "observedValue" in index.functional_props


def test_index_commands_range(index):
    assert index.properties["commands"]["range"] == "ResponderUnit"


def test_index_dispatched_to_domain(index):
    assert index.properties["dispatchedTo"]["domain"] == "ResponderUnit"


# ===================================================================
# Pipeline: Stage 1 — Extract
# ===================================================================

@pytest.fixture(scope="module")
def result():
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    return KGExtractionPipeline(corpus).run()


def test_stage1_six_sentences(result):
    assert len(result.sentences) == 6


def test_stage1_five_candidate_triples(result):
    assert len(result.candidate_triples) == 5


def test_stage1_one_negated_triple(result):
    assert len(result.negated_triples) == 1
    neg = result.negated_triples[0]
    assert neg.subject.lower() == "commander diaz"
    assert neg.object.lower() == "hazmatteam bravo"


def test_stage1_candidates_both_sources(result):
    for ct in result.candidate_triples:
        assert "dep_parse" in ct.sources
        assert "llm" in ct.sources


def test_stage1_candidate_sentence_indices(result):
    indices = [ct.sentence_index for ct in result.candidate_triples]
    assert indices == [1, 2, 3, 4, 5]


def test_stage1_object_types(result):
    """47 psi must be typed as 'value'; entity objects as 'entity'."""
    by_sent = {ct.sentence_index: ct for ct in result.candidate_triples}
    assert by_sent[3].object_type == "value"
    assert by_sent[1].object_type == "entity"


# ===================================================================
# Pipeline: Stage 2 — Map
# ===================================================================

def test_stage2_five_mapped_triples(result):
    assert len(result.mapped_triples) == 5


def test_stage2_all_high_confidence(result):
    for mt in result.mapped_triples:
        assert mt.mapping_confidence == "high"


def test_stage2_iri_minting(result):
    by_sent = {mt.sentence_index: mt for mt in result.mapped_triples}
    assert by_sent[1].subject_iri == "scima:IncidentCommander_commander_diaz"
    assert by_sent[1].object_iri == "scima:HazmatTeam_hazmatteam_alpha"
    assert by_sent[2].object_iri == "scima:HazMatSpill_incident_i204"
    assert by_sent[3].subject_iri == "scima:SensorReading_reading_r1"
    assert by_sent[4].subject_iri == "scima:WaterMain_watermain_7b"
    assert by_sent[5].object_iri == "scima:HazmatTeam_hazmatteam_gamma"


def test_stage2_predicate_kinds(result):
    by_sent = {mt.sentence_index: mt for mt in result.mapped_triples}
    assert by_sent[1].predicate_kind == "object_property"
    assert by_sent[3].predicate_kind == "datatype_property"


def test_stage2_observedvalue_literal(result):
    """'47 psi' is split; object_literal = '47', object_datatype = 'xsd:string' (conservative)."""
    by_sent = {mt.sentence_index: mt for mt in result.mapped_triples}
    mt3 = by_sent[3]
    assert mt3.object_literal == "47"
    assert mt3.object_datatype == "xsd:string"


def test_stage2_watermain_typed_as_watermain(result):
    by_sent = {mt.sentence_index: mt for mt in result.mapped_triples}
    assert "WaterMain" in by_sent[4].subject_type


# ===================================================================
# Pipeline: Stage 3 — Verify and Admit
# ===================================================================

def test_stage3_two_admit_one_repaired(result):
    verdicts = [t.verdict for t in result.admitted_triples]
    assert verdicts.count("admit") == 2
    assert verdicts.count("repaired") == 1


def test_stage3_two_rejected(result):
    assert len(result.rejected_triples) == 2


def test_stage3_reject_domain_violation(result):
    reasons = [t.reason for t in result.rejected_triples]
    assert "domain_violation" in reasons


def test_stage3_reject_cardinality_violation(result):
    reasons = [t.reason for t in result.rejected_triples]
    assert "cardinality_violation" in reasons


def test_stage3_domain_violation_is_watermain(result):
    """WaterMain 7B dispatchedTo must fail domain (WaterMain not subclass of ResponderUnit)."""
    dv = next(t for t in result.rejected_triples if t.reason == "domain_violation")
    assert "watermain" in dv.subject_iri.lower()
    assert "dispatchedTo" in dv.predicate_iri


def test_stage3_cardinality_violation_is_second_commands(result):
    """Second 'commands' triple for commander_diaz must fail FunctionalProperty check."""
    cv = next(t for t in result.rejected_triples if t.reason == "cardinality_violation")
    assert "commander_diaz" in cv.subject_iri
    assert "commands" in cv.predicate_iri


def test_stage3_repair_coerces_string_to_integer(result):
    """'47'^^xsd:string is repaired to xsd:integer for observedValue."""
    repaired = next(t for t in result.admitted_triples if t.verdict == "repaired")
    assert repaired.object_literal == "47"
    assert repaired.object_datatype == "xsd:integer"
    assert "observedValue" in repaired.predicate_iri


def test_stage3_admitted_subjects(result):
    subjs = {t.subject_iri for t in result.admitted_triples}
    assert "scima:IncidentCommander_commander_diaz" in subjs
    assert "scima:HazmatTeam_hazmatteam_alpha" in subjs
    assert "scima:SensorReading_reading_r1" in subjs


# ===================================================================
# Pipeline: Type assertions
# ===================================================================

def test_type_assertions_count(result):
    """Four individuals get type assertions: commander_diaz, hazmatteam_alpha,
    incident_i204, reading_r1. Rejected individuals (watermain_7b, hazmatteam_gamma)
    do not appear."""
    assert len(result.type_assertions) == 4


def test_type_assertions_individuals(result):
    typed = {ta.subject_iri for ta in result.type_assertions}
    assert "scima:IncidentCommander_commander_diaz" in typed
    assert "scima:HazmatTeam_hazmatteam_alpha" in typed
    assert "scima:HazMatSpill_incident_i204" in typed
    assert "scima:SensorReading_reading_r1" in typed


def test_type_assertions_classes(result):
    class_map = {ta.subject_iri: ta.object_iri for ta in result.type_assertions}
    assert class_map["scima:IncidentCommander_commander_diaz"] == "scima:IncidentCommander"
    assert class_map["scima:HazmatTeam_hazmatteam_alpha"] == "scima:HazmatTeam"
    assert class_map["scima:HazMatSpill_incident_i204"] == "scima:HazMatSpill"
    assert class_map["scima:SensorReading_reading_r1"] == "scima:SensorReading"


def test_rejected_individuals_absent_from_type_assertions(result):
    typed = {ta.subject_iri for ta in result.type_assertions}
    assert not any("watermain" in iri for iri in typed)
    assert not any("hazmatteam_gamma" in iri for iri in typed)


def test_type_assertions_predicate_is_rdf_type(result):
    assert all(ta.predicate_iri == "rdf:type" for ta in result.type_assertions)


def test_type_assertions_deduplicated(result):
    """No (individual, class) pair appears more than once."""
    pairs = [(ta.subject_iri, ta.object_iri) for ta in result.type_assertions]
    assert len(pairs) == len(set(pairs))
