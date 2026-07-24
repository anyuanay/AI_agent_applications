"""Tests that keep Article 2 honest.

Two contracts:
  1. SCIMA-OWL v0.2 matches the Growth Tracker (cumulative): 12 classes,
     20 properties (8 object + 12 datatype), 8 axioms.
  2. SCIMA-KG population works: a sample lands in a named graph, and the
     "traffic lights within 500m of incident I-204" query returns the
     deterministic answer the article shows.

The sample is generated at a small scale here so the suite stays fast; the
curated I-204 scene is added regardless of scale, so the geo query is stable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scima.ontology import ScimaOntology
from scima.knowledge_graph import ScimaKnowledgeGraph, haversine_m


# ---- Growth Tracker targets for v0.2 (cumulative) ----
EXPECTED_CLASSES = 12
EXPECTED_PROPERTIES = 20
EXPECTED_AXIOMS = 8


# ===================================================================
# Schema: SCIMA-OWL v0.2
# ===================================================================

@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.2")


def test_v0_2_parses(onto):
    assert len(onto.graph) > 0


def test_v0_2_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_v0_2_property_split(onto):
    s = onto.summary()
    assert s.n_object_properties == 8
    assert s.n_datatype_properties == 12
    assert s.n_properties == EXPECTED_PROPERTIES


def test_v0_2_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_v0_2_new_sensor_subclasses(onto):
    subs = set(onto.subclasses_of("scima:SensorDevice"))
    assert subs == {"scima:TrafficCamera", "scima:FlowSensor", "scima:PowerMeter"}


def test_v0_2_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES,
        EXPECTED_PROPERTIES,
        EXPECTED_AXIOMS,
    )


def test_v0_2_is_backward_compatible_with_v0_1():
    """Every v0.1 class survives into v0.2 (no breaking removals)."""
    v01 = set(ScimaOntology.load("v0.1").classes())
    v02 = set(ScimaOntology.load("v0.2").classes())
    assert v01 <= v02


# ===================================================================
# Knowledge graph: SCIMA-KG population and queries
# ===================================================================

@pytest.fixture(scope="module")
def kg():
    g = ScimaKnowledgeGraph()
    g.populate_sample(n_cameras=200, seed=7)
    return g


def test_population_uses_one_named_graph(kg):
    stats = kg.stats()
    assert stats.n_named_graphs == 1


def test_sensor_nodes_scale_with_cameras(kg):
    # 200 cameras + 200 readings == 400 sensor nodes.
    assert kg.stats().n_sensor_nodes == 400


def test_triples_present(kg):
    assert kg.stats().n_triples > 0


def test_lights_near_i204_returns_three_in_order(kg):
    hits = kg.lights_near("Incident_I204", radius_m=500.0)
    assert [h.light for h in hits] == ["scima:TL_90", "scima:TL_88", "scima:TL_91"]
    # distances are sorted ascending and all under the radius
    dists = [h.distance_m for h in hits]
    assert dists == sorted(dists)
    assert all(d < 500.0 for d in dists)


def test_lights_near_excludes_far_light(kg):
    hits = kg.lights_near("Incident_I204", radius_m=500.0)
    assert "scima:TL_97" not in {h.light for h in hits}  # placed at ~700m


def test_lights_all_on_main_st(kg):
    hits = kg.lights_near("Incident_I204", radius_m=500.0)
    assert all(h.road == "scima:RoadSegment_Main_St_NB" for h in hits)


def test_aggregate_sparql_counts_readings(kg):
    """The Article 2 aggregate query shape: readings grouped by camera."""
    rows = list(kg.select(
        """
        PREFIX scima: <http://scima.city/ontology#>
        SELECT (COUNT(?reading) AS ?n) WHERE {
            ?reading a scima:SensorReading ;
                     scima:recordedBy    ?camera ;
                     scima:observedValue ?value .
            FILTER (?value > 0)
        }
        """
    ))
    # 200 sampled readings all have observedValue >= 1.
    assert int(rows[0][0]) == 200


def test_haversine_known_distance():
    # ~60m north of the incident origin used in the scene.
    d = haversine_m(40.7500, -73.9900, 40.7500 + 60 / 111_320.0, -73.9900)
    assert abs(d - 60.0) < 2.0
