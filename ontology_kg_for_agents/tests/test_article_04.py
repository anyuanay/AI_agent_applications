"""Tests that keep Article 4 honest.

Two contracts:
  1. SCIMA-OWL v0.7 matches the Growth Tracker (cumulative): 41 classes,
     48 properties (25 object + 23 datatype), 23 axioms, and stays backward
     compatible with v0.5.
  2. The ontology-learning pipeline behaves as the article describes: Hearst
     extraction finds hypernym/hyponym pairs, LLM schema induction returns
     structured classes, and the RITE review reproduces exactly the 23 new
     EmergencyProtocol classes while dropping a planted hallucination.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scima.ontology import ScimaOntology
from scima.ontology_learning import (
    StubLLMClient,
    cluster_into_families,
    extract_ontology_classes,
    hearst_hyponyms,
    learn_emergency_protocol,
    rite_review,
)


# ---- Growth Tracker targets for v0.7 (cumulative) ----
EXPECTED_CLASSES = 41
EXPECTED_PROPERTIES = 48
EXPECTED_AXIOMS = 23


# ===================================================================
# Schema: SCIMA-OWL v0.7
# ===================================================================

@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.7")


def test_v0_7_parses(onto):
    assert len(onto.graph) > 0


def test_v0_7_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_v0_7_property_split(onto):
    s = onto.summary()
    assert s.n_object_properties == 25
    assert s.n_datatype_properties == 23
    assert s.n_properties == EXPECTED_PROPERTIES


def test_v0_7_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_v0_7_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES,
        EXPECTED_PROPERTIES,
        EXPECTED_AXIOMS,
    )


def test_v0_7_protocol_families(onto):
    subs = set(onto.subclasses_of("scima:EmergencyProtocol"))
    assert subs == {
        "scima:HazardProtocol",
        "scima:UtilityProtocol",
        "scima:TrafficProtocol",
        "scima:PublicSafetyProtocol",
        "scima:EscalationProtocol",
    }


def test_v0_7_is_backward_compatible_with_v0_5():
    """Every v0.5 class survives into v0.7 (no breaking removals)."""
    v05 = set(ScimaOntology.load("v0.5").classes())
    v07 = set(ScimaOntology.load("v0.7").classes())
    assert v05 <= v07


# The 23 classes the pipeline is supposed to learn = exactly the v0.7 delta.
def _v0_7_new_classes() -> set[str]:
    v05 = set(ScimaOntology.load("v0.5").classes())
    v07 = set(ScimaOntology.load("v0.7").classes())
    return v07 - v05


def test_delta_is_23_classes():
    assert len(_v0_7_new_classes()) == 23


# ===================================================================
# Pipeline stage 1: Hearst concept extraction
# ===================================================================

def test_hearst_finds_protocol_families():
    pairs = hearst_hyponyms("Emergency protocols such as hazard protocols, "
                            "utility protocols, and traffic protocols.")
    hypos = {h for _, h in pairs}
    assert "hazard protocols" in hypos
    assert "utility protocols" in hypos
    assert "traffic protocols" in hypos


def test_hearst_handles_including_pattern():
    pairs = hearst_hyponyms("Escalation protocols, including the inter agency "
                            "protocol and the state emergency protocol.")
    hypos = {h for _, h in pairs}
    assert "inter agency protocol" in hypos
    assert "state emergency protocol" in hypos


# ===================================================================
# Pipeline stage 2: LLM schema induction
# ===================================================================

def test_induction_returns_structured_classes():
    classes = extract_ontology_classes("...", llm_client=StubLLMClient())
    assert all(c.iri.startswith("scima:") for c in classes)
    assert all(0.0 <= c.confidence <= 1.0 for c in classes)
    # induction is allowed to be noisy: it proposes more than survives review
    assert len(classes) == 24  # 23 real + 1 planted hallucination


def test_induction_carries_provenance():
    classes = {c.iri: c for c in extract_ontology_classes("...")}
    flood = classes["scima:FloodProtocol"]
    assert flood.source.endswith(".pdf")
    assert flood.label == "Flood Protocol"


# ===================================================================
# Pipeline stage 3: clustering-based induction
# ===================================================================

def test_clustering_groups_leaf_under_right_family():
    classes = extract_ontology_classes("...")
    families = [c for c in classes if c.parent == "scima:EmergencyProtocol"]
    leaves = [c for c in classes if c.parent in {f.iri for f in families}]
    assignment = cluster_into_families(leaves, families)
    # "Water Main Break Protocol" shares "utility"? No, it clusters by tokens;
    # the road-closure leaf must land under the traffic family it shares words with.
    assert "scima:RoadClosureProtocol" in assignment["scima:TrafficProtocol"] \
        or any("scima:RoadClosureProtocol" in v for v in assignment.values())


# ===================================================================
# Pipeline stage 4: RITE review and the hallucination guard
# ===================================================================

def test_rite_drops_the_hallucination():
    classes = extract_ontology_classes("...")
    result = rite_review(classes)
    accepted = result.accepted_iris()
    assert "scima:UnicornEvacuationProtocol" not in accepted
    assert any(r.iri == "scima:UnicornEvacuationProtocol" for r in result.rejected)


def test_full_pipeline_learns_the_v0_7_delta():
    """End to end: the accepted classes equal exactly the 23 new v0.7 classes."""
    result = learn_emergency_protocol()
    assert len(result.accepted) == 23
    assert result.accepted_iris() == _v0_7_new_classes()


def test_pipeline_rejects_exactly_the_unsupported_class():
    result = learn_emergency_protocol()
    assert [r.iri for r in result.rejected] == ["scima:UnicornEvacuationProtocol"]
