"""Tests that keep Article 3 honest.

Two contracts:
  1. SCIMA-OWL v0.5 matches the Growth Tracker (cumulative): 18 classes,
     30 properties (15 object + 15 datatype), 12 axioms, and stays backward
     compatible with v0.2.
  2. The context graph behaves as the article describes: a budgeted k-hop
     projection scored by relevance, evolving across four turns with the
     focal entity pinned, the stale sensor evicted, and a precondition gate
     that refuses to act on an unsupported state.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scima.ontology import ScimaOntology
from scima.context_graph import ContextGraphBuilder, DEFAULT_MAX_NODES


# ---- Growth Tracker targets for v0.5 (cumulative) ----
EXPECTED_CLASSES = 18
EXPECTED_PROPERTIES = 30
EXPECTED_AXIOMS = 12

FOCAL = "scima:Incident_I204"


# ===================================================================
# Schema: SCIMA-OWL v0.5
# ===================================================================

@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.5")


def test_v0_5_parses(onto):
    assert len(onto.graph) > 0


def test_v0_5_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_v0_5_property_split(onto):
    s = onto.summary()
    assert s.n_object_properties == 15
    assert s.n_datatype_properties == 15
    assert s.n_properties == EXPECTED_PROPERTIES


def test_v0_5_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_v0_5_agent_roles(onto):
    subs = set(onto.subclasses_of("scima:Agent"))
    assert subs == {"scima:ZoneAgent", "scima:DispatchAgent"}


def test_v0_5_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES,
        EXPECTED_PROPERTIES,
        EXPECTED_AXIOMS,
    )


def test_v0_5_is_backward_compatible_with_v0_2():
    """Every v0.2 class survives into v0.5 (no breaking removals)."""
    v02 = set(ScimaOntology.load("v0.2").classes())
    v05 = set(ScimaOntology.load("v0.5").classes())
    assert v02 <= v05


# ===================================================================
# Context graph: projection and relevance scoring
# ===================================================================

@pytest.fixture()
def builder():
    return ContextGraphBuilder()


def test_build_respects_node_budget(builder):
    cg = builder.build_context_graph(FOCAL, "resolve", max_nodes=50)
    assert len(cg) <= 50


def test_build_demo_scene_is_142_nodes(builder):
    """The article's --build output: 142 nodes within the 150 budget."""
    cg = builder.build_context_graph(FOCAL, "resolve", max_nodes=DEFAULT_MAX_NODES)
    assert len(cg) == 142


def test_focal_is_always_first_and_highest(builder):
    cg = builder.build_context_graph(FOCAL, "resolve")
    assert cg[0].qname == FOCAL
    assert cg[0].hop == 0
    assert cg[0].relevance == max(s.relevance for s in cg)


def test_only_reaches_k_hops(builder):
    cg = builder.build_context_graph(FOCAL, "resolve", k=2)
    assert max(s.hop for s in cg) <= 2


def test_semantic_relevance_prefers_goal_types(builder):
    """A goal-relevant water main outranks an off-goal traffic camera at the
    same or nearer distance, because semantic relevance lifts it."""
    cg = {s.qname: s for s in builder.build_context_graph(FOCAL, "resolve")}
    water = cg["scima:WaterMain_7B"]            # hop 1, goal-relevant
    cam = next(s for q, s in cg.items() if q.startswith("scima:Camera_Adj_"))
    assert water.relevance > cam.relevance


def test_lower_budget_drops_lowest_relevance(builder):
    """Pruning keeps the high-relevance core and drops the filler ring."""
    cg = builder.build_context_graph(FOCAL, "resolve", max_nodes=8)
    kept = {s.qname for s in cg}
    assert FOCAL in kept
    assert "scima:WaterMain_7B" in kept
    # the filler camera ring should not survive an 8-node budget
    assert not any(q.startswith("scima:Camera_Adj_") for q in kept)


# ===================================================================
# Context graph: evolution across turns
# ===================================================================

def test_trace_has_four_turns(builder):
    recs = builder.trace(focal=FOCAL)
    assert [r.turn for r in recs] == [0, 1, 2, 3]


def test_focal_is_pinned_every_turn(builder):
    """Goal-anchoring: the focal entity is never evicted (Section 6)."""
    recs = builder.trace(focal=FOCAL)
    for r in recs:
        assert FOCAL in r.members


def test_turn0_is_seed(builder):
    recs = builder.trace(focal=FOCAL)
    assert set(recs[0].members) == {FOCAL, "scima:WaterMain_7B"}


def test_stale_sensor_evicted_by_end(builder):
    """A42 is in context at turn 1 but evicted once it goes stale."""
    recs = builder.trace(focal=FOCAL)
    assert "scima:FlowSensor_A42" in recs[1].members
    assert "scima:FlowSensor_A42" in recs[2].evicted
    assert "scima:FlowSensor_A42" not in recs[3].members


def test_fresh_backup_and_vehicle_pulled_in_during_planning(builder):
    recs = builder.trace(focal=FOCAL)
    assert "scima:FlowSensor_B7" in recs[2].members        # fresh backup
    assert "scima:Vehicle_Ambulance_3" in recs[2].members  # dispatch resource


def test_off_goal_bus_never_enters_context(builder):
    """The far transit bus is never pulled into the working set."""
    recs = builder.trace(focal=FOCAL)
    for r in recs:
        assert "scima:Vehicle_Bus_88" not in r.members


def test_no_turn_exceeds_budget(builder):
    recs = builder.trace(focal=FOCAL, max_nodes=DEFAULT_MAX_NODES)
    for r in recs:
        assert len(r.members) <= DEFAULT_MAX_NODES


# ===================================================================
# Grounding: the precondition gate (Section 6)
# ===================================================================

def test_precondition_holds_for_active_incident_and_free_vehicle(builder):
    assert builder.precondition_holds("scima:Incident_I204",
                                      "scima:Vehicle_Ambulance_3") is True


def test_dispatch_writes_back_and_consumes_availability(builder):
    assert builder.dispatch("scima:Vehicle_Ambulance_3", "scima:Incident_I204") is True
    # the assignment is written back to the graph
    from scima.context_graph import SCIMA
    assigned = list(builder.g.objects(SCIMA["Vehicle_Ambulance_3"], SCIMA.assignedTo))
    assert SCIMA["Incident_I204"] in assigned
    # the vehicle is now busy, so the precondition no longer holds
    assert builder.precondition_holds("scima:Incident_I204",
                                      "scima:Vehicle_Ambulance_3") is False


def test_dispatch_refused_when_vehicle_unavailable(builder):
    # an unknown / unavailable vehicle cannot be dispatched
    assert builder.dispatch("scima:Vehicle_Bus_88", "scima:Incident_I204") is False
