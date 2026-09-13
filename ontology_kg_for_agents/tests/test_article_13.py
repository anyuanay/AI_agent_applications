"""Article 13: agent belief graphs, private vs shared knowledge.

Asserts five contracts:

  * SCIMA-OWL v1.8 matches the Growth Tracker (80 classes, 110
    properties, 46 axioms), still contains everything v1.5 declared, and
    adds no axiom that condemns a stored triple.
  * A belief read at two instants is two numbers, and the article's
    figure-2 pair (0.7857 and 0.5181) is what aging both to one instant
    returns.
  * One message routes to a revision for the agent that synced twenty
    seconds ago and an update for the agent that synced four minutes
    ago, the flip age between them is a closed form, and the frame on an
    intention is what keeps the revision's cleanup small.
  * The divergence between the two agents is asymmetric, and the number
    that prices the sync decision is a different number from the one
    that ranks the pair.
  * The resolution ladder leaves at question two, because the two
    sensors watch two different approach legs of one node.
"""

import doctest
import math

import pytest

from rdflib import OWL, RDF, RDFS

import scima.agent_beliefs as ab
from scima.agent_beliefs import (
    CONGESTION_VALUES,
    GRAPH_OWNER,
    INTENTIONS,
    LANE_FACT,
    MAIN_AND_NINTH_LEGS,
    NORTH,
    NOW,
    SCENE,
    SHARED,
    SHARED_VALUE_ERROR_RATE,
    SOUTH,
    aged_belief,
    aged_distribution,
    aged_scene_distributions,
    admitted,
    chance_they_act_apart,
    chance_world_moved,
    compare_at,
    gap_is_mostly_clock,
    divergence_trend,
    half_life_for,
    kl_divergence,
    operator_flip_age,
    p_update,
    publication_floor,
    reopened_by,
    resolution_ladder,
    rollup,
    route_operator,
    rules_inherited_by_legs,
    should_publish,
    superclasses_of,
    worth_a_message,
)
from scima.ontology import ScimaOntology

NORTH_AGENT = "scima:ZoneAgent_North"
SOUTH_AGENT = "scima:ZoneAgent_South"
LANE_HALF_LIFE = 1800.0


# ---------------------------------------------------------------------------
# The Growth Tracker is the contract
# ---------------------------------------------------------------------------

def test_v1_8_matches_the_growth_tracker():
    s = ScimaOntology.load("v1.8").summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (80, 110, 46)
    assert (s.n_object_properties, s.n_datatype_properties) == (60, 50)


def test_the_delta_over_v1_5_is_the_one_the_article_claims():
    old = ScimaOntology.load("v1.5").summary()
    new = ScimaOntology.load("v1.8").summary()
    assert new.n_classes - old.n_classes == 6
    assert new.n_properties - old.n_properties == 10
    assert new.n_axioms - old.n_axioms == 5


def test_nothing_v1_5_declared_was_removed():
    """The append-only rule, asserted rather than trusted."""
    old = ScimaOntology.load("v1.5")
    new = ScimaOntology.load("v1.8")
    for getter in ("classes", "object_properties", "datatype_properties"):
        missing = set(getattr(old, getter)()) - set(getattr(new, getter)())
        assert not missing, f"{getter} lost {sorted(missing)}"


def test_the_six_new_classes_are_the_six_named_in_the_article():
    old = set(ScimaOntology.load("v1.5").classes())
    new = set(ScimaOntology.load("v1.8").classes())
    assert new - old == {
        "scima:BeliefGraph", "scima:Intention", "scima:PublicationEvent",
        "scima:DivergenceObservation", "scima:Intersection", "scima:ApproachLeg",
    }


def test_an_approach_leg_is_a_road_segment():
    """The free-inheritance move. Segment rules cover legs with no new code."""
    onto = ScimaOntology.load("v1.8")
    assert "scima:ApproachLeg" in onto.subclasses_of("scima:RoadSegment")
    assert rules_inherited_by_legs() == [
        "scima:InfrastructureEntity", "scima:RoadSegment", "scima:ZonedEntity",
    ]


def test_a_leg_and_a_junction_are_kept_apart():
    """The disjointness that stops one leg's reading standing for the node."""
    onto = ScimaOntology.load("v1.8")
    leg = onto._to_ref("scima:ApproachLeg")
    junction = onto._to_ref("scima:Intersection")
    assert (leg, OWL.disjointWith, junction) in onto.graph


def test_an_intention_must_name_the_frame_it_was_formed_under():
    """The strict axiom, and the reason it pays for itself."""
    onto = ScimaOntology.load("v1.8")
    intention = onto._to_ref("scima:Intention")
    frame_prop = onto._to_ref("scima:formedUnderFrame")
    found = False
    for restriction in onto.graph.objects(intention, RDFS.subClassOf):
        if (restriction, RDF.type, OWL.Restriction) not in onto.graph:
            continue
        if (restriction, OWL.onProperty, frame_prop) in onto.graph:
            card = onto.graph.value(restriction, OWL.cardinality)
            assert card is not None and int(card) == 1
            found = True
    assert found, "scima:Intention has no cardinality restriction on formedUnderFrame"


def test_every_new_axiom_is_safe():
    """v1.8 adds nothing that condemns a triple written under v1.5.

    Both new restrictions are attached to classes v1.8 introduces, so no
    instance of either existed before this version. That is what makes
    the release safe, and it is checkable rather than a claim.
    """
    old = ScimaOntology.load("v1.5")
    new = ScimaOntology.load("v1.8")
    old_classes = set(old.classes())
    restricted = {
        cls for cls in new.classes()
        if any((r, RDF.type, OWL.Restriction) in new.graph
               for r in new.graph.objects(new._to_ref(cls), RDFS.subClassOf))
    }
    old_restricted = {
        cls for cls in old.classes()
        if any((r, RDF.type, OWL.Restriction) in old.graph
               for r in old.graph.objects(old._to_ref(cls), RDFS.subClassOf))
    }
    newly_restricted = restricted - old_restricted
    assert newly_restricted == {"scima:BeliefGraph", "scima:Intention"}
    assert not (newly_restricted & old_classes)


# ---------------------------------------------------------------------------
# Section 1: a belief needs an owner
# ---------------------------------------------------------------------------

def test_one_statement_two_graphs_two_beliefs_no_contradiction():
    north, south = SCENE["north_congestion"], SCENE["south_congestion"]
    assert north.subject == south.subject
    assert north.predicate == south.predicate
    assert north.value != south.value
    assert north.graph != south.graph
    assert GRAPH_OWNER[north.graph] == NORTH_AGENT
    assert GRAPH_OWNER[south.graph] == SOUTH_AGENT


def test_the_shared_graph_has_no_owner():
    """A graph with an owner is private. The shared one is nobody's."""
    assert GRAPH_OWNER[SHARED] is None
    assert GRAPH_OWNER[NORTH] and GRAPH_OWNER[SOUTH]


def test_the_rate_belongs_to_the_predicate_not_the_reader():
    assert half_life_for("scima:hasCongestionLevel") == 60.0
    assert half_life_for("scima:hasLaneStatus") == LANE_HALF_LIFE
    assert math.isinf(half_life_for("scima:hasStreetAddress"))
    assert half_life_for("scima:neverMeasured") == ab.DEFAULT_HALF_LIFE_S


# ---------------------------------------------------------------------------
# Section 2: the gate between private and shared
# ---------------------------------------------------------------------------

def test_the_publication_floor_is_derived():
    assert publication_floor() == pytest.approx(0.7714, abs=1e-4)


def test_the_floor_moves_when_either_cost_moves():
    base = publication_floor()
    dearer_mistake = publication_floor(cost_of_a_wrong_shared_fact=2700.0)
    dearer_silence = publication_floor(cost_of_a_decision_made_without_it=800.0)
    assert dearer_mistake > base > dearer_silence


def test_a_faded_belief_stays_private():
    """Your uncertainty must not become everybody else's fact."""
    south = SCENE["south_congestion"]
    assert should_publish(south.belief)          # fresh, it would have published
    assert not should_publish(south.aged(NOW))   # aged to now, it does not


def test_a_floor_needs_at_least_one_positive_cost():
    with pytest.raises(ValueError):
        publication_floor(cost_of_a_wrong_shared_fact=0.0,
                          cost_of_a_decision_made_without_it=0.0)


# ---------------------------------------------------------------------------
# Section 3 and C5: compare at one instant
# ---------------------------------------------------------------------------

def test_the_figure_two_numbers():
    """The stored pair and the aged pair, as the article's figure shows them."""
    live = compare_at(SCENE["north_congestion"], SCENE["south_congestion"], NOW)
    assert live[0] == pytest.approx(0.7857, abs=1e-4)
    assert live[1] == pytest.approx(0.5181, abs=1e-4)


def test_aging_moves_a_belief_toward_no_claim_and_never_past_it():
    for age in (0.0, 30.0, 600.0, 10_000.0):
        assert 0.5 <= aged_belief(0.86, age, 60.0) <= 0.86
        assert 0.14 <= aged_belief(0.14, age, 60.0) <= 0.5
    assert aged_belief(0.86, 10_000.0, 60.0) == pytest.approx(0.5, abs=1e-6)


def test_a_predicate_that_does_not_fade_does_not_fade():
    assert aged_belief(0.90, 86_400.0, math.inf) == 0.90


def test_an_aged_distribution_still_sums_to_one():
    for age in (0.0, 20.0, 240.0, 5_000.0):
        d = aged_distribution((0.06, 0.15, 0.79), age, 60.0)
        assert sum(d) == pytest.approx(1.0, abs=1e-12)
        assert all(0.0 <= p <= 1.0 for p in d)


def test_the_camera_frame_has_crossed_four_half_lives():
    south = SCENE["south_congestion"]
    assert south.age_at(NOW) / south.half_life_s == pytest.approx(4.0)


def test_most_of_the_gap_goes_with_the_clock():
    assert gap_is_mostly_clock(SCENE["north_congestion"],
                               SCENE["south_congestion"], NOW)


def test_a_belief_cannot_be_read_before_it_was_formed():
    with pytest.raises(ValueError):
        aged_belief(0.86, -1.0, 60.0)


def test_one_gate_quarantines_what_another_admitted():
    """The quarantine case, carried forward from the planning article.

    Two values for a property allowed one leave the working graph with no
    version of the fact at all, so the two agents disagree about whether
    it exists before they disagree about what it says.
    """
    ok_one, _ = admitted(["scima:Helipad_03"])
    ok_two, why = admitted(["scima:Helipad_03", "scima:Station_07"])
    assert ok_one and not ok_two
    assert "quarantined" in why
    absent_ok, absent_why = admitted([])
    assert not absent_ok and "not the same as false" in absent_why


# ---------------------------------------------------------------------------
# Section 4 and C4: one message, two operators
# ---------------------------------------------------------------------------

def test_one_message_routes_to_two_operators():
    north, south = SCENE["north_lanes"], SCENE["south_lanes"]
    assert north.value == south.value          # they copied the same value
    assert north.source == south.source == SHARED
    assert route_operator(north.age_at(NOW), LANE_HALF_LIFE) == "revision"
    assert route_operator(south.age_at(NOW), LANE_HALF_LIFE) == "update"


def test_the_two_probabilities_behind_that_fork():
    assert p_update(20.0, LANE_HALF_LIFE) == pytest.approx(0.1281, abs=1e-4)
    assert p_update(240.0, LANE_HALF_LIFE) == pytest.approx(0.6478, abs=1e-4)


def test_p_update_rises_with_age():
    ages = [0.0, 20.0, 60.0, 240.0, 900.0, 3600.0]
    values = [p_update(a, LANE_HALF_LIFE) for a in ages]
    assert values == sorted(values)
    assert values[0] == 0.0


def test_the_flip_age_is_a_closed_form_between_the_two_agents():
    """The fork turns over where P(the world moved) equals P(a bad value)."""
    flip = operator_flip_age(LANE_HALF_LIFE)
    assert flip == pytest.approx(133.2, abs=0.1)
    assert 20.0 < flip < 240.0
    assert p_update(flip, LANE_HALF_LIFE) == pytest.approx(0.5, abs=1e-9)
    assert chance_world_moved(flip, LANE_HALF_LIFE) == pytest.approx(
        SHARED_VALUE_ERROR_RATE, abs=1e-9)


def test_a_worse_source_pushes_the_flip_later_and_a_faster_predicate_pulls_it_in():
    assert operator_flip_age(LANE_HALF_LIFE, 0.20) > operator_flip_age(LANE_HALF_LIFE)
    assert operator_flip_age(60.0) < operator_flip_age(LANE_HALF_LIFE)


def test_congestion_would_have_forked_the_other_way():
    """The same twenty seconds against a sixty-second half-life is old."""
    assert route_operator(20.0, 60.0) == "update"
    assert route_operator(20.0, LANE_HALF_LIFE) == "revision"


def test_a_revision_reopens_only_what_used_the_bad_fact():
    reopened = reopened_by(LANE_FACT, "revision", INTENTIONS, NORTH_AGENT)
    assert reopened == ["route AMB-17 up Main"]
    north_held = [i.label for i in INTENTIONS if i.agent == NORTH_AGENT]
    assert len(north_held) == 2, "the second one is what the frame protects"


def test_an_update_reopens_nothing():
    assert reopened_by(LANE_FACT, "update", INTENTIONS, SOUTH_AGENT) == []
    assert reopened_by(LANE_FACT, "update", INTENTIONS) == []


def test_without_a_frame_every_commitment_would_be_rechecked():
    """What the strict cardinality axiom buys, stated as a count."""
    frame_scoped = reopened_by(LANE_FACT, "revision", INTENTIONS, NORTH_AGENT)
    all_north = [i.label for i in INTENTIONS if i.agent == NORTH_AGENT]
    assert len(frame_scoped) < len(all_north)


def test_an_unknown_operator_is_refused():
    with pytest.raises(ValueError):
        reopened_by(LANE_FACT, "overwrite", INTENTIONS)


# ---------------------------------------------------------------------------
# Section 5 and C3: measure the gap, then price the message
# ---------------------------------------------------------------------------

def test_the_divergence_is_asymmetric():
    north, south = aged_scene_distributions()
    assert kl_divergence(north, south) == pytest.approx(0.4065, abs=1e-4)
    assert kl_divergence(south, north) == pytest.approx(0.4397, abs=1e-4)
    assert kl_divergence(north, south) != kl_divergence(south, north)


def test_divergence_is_zero_only_when_the_two_agree():
    d = (0.7, 0.2, 0.1)
    assert kl_divergence(d, d) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(d, (0.6, 0.3, 0.1)) > 0.0


def test_ranking_and_pricing_are_two_different_numbers():
    north, south = aged_scene_distributions()
    ranking = kl_divergence(north, south)
    pricing = chance_they_act_apart(north, south)
    assert pricing == pytest.approx(0.4351, abs=1e-4)
    assert ranking != pricing
    assert 0.0 <= pricing <= 1.0      # the price is a probability, the rank is not


def test_this_gap_earns_a_message():
    north, south = aged_scene_distributions()
    assert worth_a_message(north, south)
    assert not worth_a_message((0.34, 0.33, 0.33), (0.33, 0.34, 0.33))


def test_a_cheaper_wrong_action_stops_the_message():
    north, south = aged_scene_distributions()
    assert not worth_a_message(north, south, cost_of_wrong_action=0.3)


def test_the_trend_carries_more_than_the_level():
    assert divergence_trend([0.41, 0.40, 0.42, 0.41]) == "steady"
    assert divergence_trend([0.11, 0.24, 0.38, 0.53]) == "widening"
    with pytest.raises(ValueError):
        divergence_trend([0.41])


def test_distributions_over_different_value_sets_cannot_be_compared():
    with pytest.raises(ValueError):
        kl_divergence((0.5, 0.5), (0.4, 0.3, 0.3))


# ---------------------------------------------------------------------------
# Section 6: the ladder, and the class that was missing
# ---------------------------------------------------------------------------

def test_the_ladder_leaves_at_question_two():
    out = resolution_ladder(SCENE["north_congestion"], SCENE["south_congestion"])
    assert out["exits_at"] == 2
    assert out["verdict"] == "different entities, split the node"
    assert out["steps"][0]["closed_it"] is False      # the clock narrows it
    assert out["steps"][1]["closed_it"] is True       # the entity explains it


def test_the_two_sensors_watch_two_different_legs():
    north, south = SCENE["north_congestion"], SCENE["south_congestion"]
    assert north.measured_from != south.measured_from
    assert north.measured_from in MAIN_AND_NINTH_LEGS
    assert south.measured_from in MAIN_AND_NINTH_LEGS


def test_two_agents_on_one_leg_do_not_leave_at_question_two():
    """Change nothing but the leg and the ladder walks further."""
    south_on_the_same_leg = ab.Claim(
        **{**SCENE["south_congestion"].__dict__,
           "measured_from": SCENE["north_congestion"].measured_from})
    out = resolution_ladder(SCENE["north_congestion"], south_on_the_same_leg)
    assert out["exits_at"] > 2


def test_the_ladder_stops_at_question_one_when_both_have_faded():
    stale_north = ab.Claim(**{**SCENE["north_congestion"].__dict__,
                              "seen_at": NOW - 600.0})
    out = resolution_ladder(stale_north, SCENE["south_congestion"])
    assert out["exits_at"] == 1
    assert out["verdict"] == "two clocks, not two opinions"


def test_the_intersection_value_is_computed_never_stored():
    assert rollup() == "HEAVY"
    assert rollup({"a": "LIGHT", "b": "LIGHT"}) == "LIGHT"
    assert rollup({"a": "MODERATE", "b": "LIGHT"}) == "MODERATE"
    assert set(MAIN_AND_NINTH_LEGS.values()) <= set(CONGESTION_VALUES)


def test_an_intersection_with_no_legs_has_no_congestion():
    with pytest.raises(ValueError):
        rollup({})


def test_the_leg_hierarchy_is_read_at_run_time_not_typed_by_hand():
    onto = ScimaOntology.load("v1.8")
    assert superclasses_of(onto, "scima:ApproachLeg") == rules_inherited_by_legs()
    assert "scima:RoadSegment" in superclasses_of(onto, "scima:ApproachLeg")


# ---------------------------------------------------------------------------
# The module's own examples
# ---------------------------------------------------------------------------

def test_every_doctest_in_the_module_runs():
    result = doctest.testmod(ab, verbose=False)
    assert result.failed == 0, f"{result.failed} of {result.attempted} doctests failed"
    assert result.attempted >= 30
