"""Tests that keep Article 1 honest.

These assert that the canonical SCIMA-OWL v0.1 Turtle file and the
building-block scaffold agree with each other and with the Growth Tracker
target: 8 classes, 12 properties, 5 axioms.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scima.ontology import ScimaOntology
from scima.building_blocks import scima_v0_1, PropertyKind


# ---- Growth Tracker targets for v0.1 ----
EXPECTED_CLASSES = 8
EXPECTED_PROPERTIES = 12
EXPECTED_AXIOMS = 5


@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.1")


def test_turtle_parses(onto):
    assert len(onto.graph) > 0


def test_turtle_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_turtle_property_count(onto):
    summary = onto.summary()
    assert summary.n_object_properties == 6
    assert summary.n_datatype_properties == 6
    assert summary.n_properties == EXPECTED_PROPERTIES


def test_turtle_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_infrastructure_hierarchy(onto):
    subs = onto.subclasses_of("scima:InfrastructureEntity")
    assert set(subs) == {
        "scima:RoadSegment",
        "scima:TrafficLight",
        "scima:PowerNode",
        "scima:WaterMain",
    }


def test_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES,
        EXPECTED_PROPERTIES,
        EXPECTED_AXIOMS,
    )


# ---- building-block scaffold agrees with the Turtle ----

def test_building_blocks_match_turtle():
    o = scima_v0_1()
    assert len(o.classes) == EXPECTED_CLASSES
    assert len(o.properties) == EXPECTED_PROPERTIES
    assert len(o.axioms) == EXPECTED_AXIOMS
    assert len(o.object_properties()) == 6
    assert len(o.datatype_properties()) == 6


def test_building_blocks_locatedOn_is_object_property():
    o = scima_v0_1()
    located_on = next(p for p in o.properties if p.iri == "scima:locatedOn")
    assert located_on.kind is PropertyKind.OBJECT
    assert located_on.domain == "scima:TrafficLight"
    assert located_on.range == "scima:RoadSegment"
