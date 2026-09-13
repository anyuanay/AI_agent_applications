"""Article 11: ontology for agent goal achievement and planning.

Asserts four contracts:

  * SCIMA-OWL v1.1 matches the Growth Tracker (62 classes, 85
    properties, 32 axioms) and still contains everything v1.0 declared.
  * The class hierarchy does the work the article says it does: methods
    are inherited by subclasses, change rates are inherited by
    subclasses, and a hard-coded type list silently loses the one
    vehicle whose plan meets the deadline.
  * Preconditions priced as beliefs at the times their steps run reverse
    the ranking of the three candidate plans completely.
  * The constraint machinery is derived rather than tabulated: mutex
    pairs come out of axioms, and a quarantined precondition is unknown
    rather than false.
"""

import math

import pytest

from scima.agent_planning import (
    CALL_HANDLING_MIN,
    CARDIAC_PROTOCOL_MIN,
    CHANGE_RATE_PROFILES,
    COMMIT_FLOOR,
    CONTAINMENT_DEADLINE_MIN,
    METHODS,
    PROPOSED_EFFECTS,
    QUARANTINE_RECOVERY_MIN,
    QUARANTINE_RESOLUTION_MIN,
    SURGE_REGIME,
    SURGE_RATE_MULTIPLIER,
    QuarantinedFact,
    belief_weight,
    breakeven_belief,
    candidate_funnel,
    candidate_plans,
    classify_contradiction,
    classify_goal,
    critical_path_min,
    decay_rate,
    derive_mutex,
    expected_duration,
    flip_probability_per_minute,
    frame_scoped_recheck,
    grounding_cascade,
    half_life_for,
    methods_for,
    plan_shelf_life,
    rank_plans,
    resolve_or_carry,
    schedule,
    transport_deadline_min,
    verification_value,
    violates_resource_type,
)
from scima.ontology import ScimaOntology


@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v1.1")


@pytest.fixture(scope="module")
def plans():
    return candidate_plans()


# ---------------------------------------------------------------------------
# SCIMA-OWL v1.1: the Growth Tracker contract
# ---------------------------------------------------------------------------
def test_v1_1_matches_the_growth_tracker(onto):
    s = onto.summary()
    assert s.n_classes == 62
    assert s.n_object_properties == 45
    assert s.n_datatype_properties == 40
    assert s.n_properties == 85
    assert s.n_axioms == 32


def test_v1_1_delta_over_v1_0():
    """The article's callout claims 19 classes, 32 properties, 11 axioms."""
    old = ScimaOntology.load("v1.0").summary()
    new = ScimaOntology.load("v1.1").summary()
    assert new.n_classes - old.n_classes == 19
    assert new.n_properties - old.n_properties == 32
    assert new.n_axioms - old.n_axioms == 11


def test_v1_1_is_backward_compatible_with_v1_0(onto):
    """Nothing v1.0 declared may vanish, since the KG is cumulative."""
    old = ScimaOntology.load("v1.0")
    assert set(old.classes()) <= set(onto.classes())
    assert set(old.object_properties()) <= set(onto.object_properties())
    assert set(old.datatype_properties()) <= set(onto.datatype_properties())


def test_extending_the_axiom_counter_left_earlier_versions_alone():
    """Class-expression axioms arrive in v1.1, so no earlier total moves."""
    expected = {"v0.1": 5, "v0.2": 8, "v0.5": 12, "v0.6": 15, "v0.8": 18, "v1.0": 21}
    for version, count in expected.items():
        assert ScimaOntology.load(version).summary().n_axioms == count


def test_the_goal_taxonomy_is_declared(onto):
    for cls in ("scima:IncidentResolutionGoal", "scima:PatientTransportGoal",
                "scima:UrgentTransportGoal", "scima:ServiceRestorationGoal",
                "scima:WaterServiceRestorationGoal",
                "scima:PowerServiceRestorationGoal",
                "scima:TrafficClearanceGoal", "scima:PublicNotificationGoal"):
        assert cls in onto.classes()


def test_service_restoration_has_two_children(onto):
    kids = onto.subclasses_of("scima:ServiceRestorationGoal")
    assert kids == ["scima:PowerServiceRestorationGoal",
                    "scima:WaterServiceRestorationGoal"]


# ---------------------------------------------------------------------------
# Section 1: the deadline is derived, not picked
# ---------------------------------------------------------------------------
def test_the_transport_deadline_is_derived():
    assert transport_deadline_min() == 16.0
    assert CARDIAC_PROTOCOL_MIN - CALL_HANDLING_MIN == 16.0


def test_the_two_deadlines_run_on_different_clocks():
    """One tree, two deadlines, and the tighter one is not the longer branch."""
    assert transport_deadline_min() == 16.0
    assert CONTAINMENT_DEADLINE_MIN == 20.0
    assert transport_deadline_min() < CONTAINMENT_DEADLINE_MIN


# ---------------------------------------------------------------------------
# Section 2: grounding through OWL classes
# ---------------------------------------------------------------------------
def test_the_grounding_cascade():
    c = grounding_cascade()
    assert c.untyped == 3_969_000_000
    assert c.owl_typed == 4_800
    assert c.precondition_first == 7
    assert round(c.typing_factor) == 826_875
    assert round(c.precondition_factor) == 686
    assert round(c.total_factor) == 567_000_000


# ---------------------------------------------------------------------------
# Section 3: three filters, only one of which is the class hierarchy
# ---------------------------------------------------------------------------
def test_the_candidate_funnel_narrows_seven_to_five_to_four(onto):
    f = candidate_funnel(onto)
    assert len(f.type_correct) == 7
    assert len(f.applicable) == 5
    assert len(f.useful) == 4


def test_the_fire_truck_is_type_correct_but_not_useful(onto):
    """The hierarchy says a FireTruck may bind. An axiom says it cannot help."""
    f = candidate_funnel(onto)
    assert "scima:FIRE_12" in f.type_correct
    assert "scima:FIRE_12" in f.applicable
    assert "scima:FIRE_12" not in f.useful


def test_the_stale_type_list_loses_the_air_ambulance(onto):
    f = candidate_funnel(onto)
    assert len(f.stale_list_result) == 3
    assert f.missed_by_stale_list == ["scima:AIR_3"]
    # And it is the plan built on that vehicle which meets the deadline.
    plan_b = candidate_plans()[1]
    assert plan_b.vehicle == "scima:AIR_3"
    assert plan_b.expected_min() < transport_deadline_min()


def test_air_ambulance_is_reached_through_two_subclass_steps(onto):
    """AirAmbulance -> Ambulance -> EmergencyVehicle, which is why a
    one-level type list misses it."""
    assert "scima:AirAmbulance" in onto.subclasses_of("scima:Ambulance")
    assert "scima:Ambulance" in onto.subclasses_of("scima:EmergencyVehicle")


# ---------------------------------------------------------------------------
# Section 4: decomposition
# ---------------------------------------------------------------------------
def test_one_method_covers_both_service_restoration_children(onto):
    water = methods_for(onto, "scima:WaterServiceRestorationGoal")
    power = methods_for(onto, "scima:PowerServiceRestorationGoal")
    assert [m.name for m in water] == ["M3"]
    assert [m.name for m in power] == ["M3"]
    assert len(METHODS) == 4


def test_the_termination_rule_has_three_outcomes(onto):
    assert classify_goal(onto, "scima:IncidentResolutionGoal") == "decomposable"
    assert classify_goal(onto, "scima:PublicNotificationGoal") == "leaf"
    # A goal class nobody wrote a method for and no action achieves.
    assert classify_goal(onto, "scima:GoalState") == "stuck"


def test_the_critical_path_runs_through_the_water_branch():
    s = schedule()
    assert s["PumpOut"] == (6.0, 18.0)
    assert critical_path_min() == 18.0
    assert CONTAINMENT_DEADLINE_MIN - critical_path_min() == 2.0


def test_pumpout_waits_on_the_later_of_its_two_dependencies():
    s = schedule()
    assert s["IsolateValve"][1] == 6.0
    assert s["DivertTraffic"][1] == 4.5
    assert s["PumpOut"][0] == max(6.0, 4.5)


def test_the_urgent_branch_is_not_the_long_branch():
    """The patient branch finishes first and still carries the tighter
    deadline, which is exactly why partial-order planning exists."""
    s = schedule()
    assert round(s["Transport"][1], 2) == 12.6
    assert s["Transport"][1] < critical_path_min()


# ---------------------------------------------------------------------------
# Section 5: preconditions as beliefs at future times
# ---------------------------------------------------------------------------
def test_the_decay_rate_is_the_transition_law_in_closed_form():
    """lambda = ln 2 / h, and the per-minute flip probability it implies
    is a quantity an operator can actually observe."""
    lam = decay_rate(90.0)
    assert round(lam, 6) == 0.007702
    q = flip_probability_per_minute(90.0)
    assert round(q, 4) == 0.3700
    # Round trip: -ln(1 - q) per minute, back to lambda per second.
    assert round(-math.log(1 - q) / 60, 6) == round(lam, 6)
    # And the half-life comes back out.
    assert round(math.log(2) / lam, 1) == 90.0


def test_change_rates_are_inherited_down_the_class_hierarchy(onto):
    """AirAmbulance declares no profile, so it uses Ambulance's."""
    assert ("scima:AirAmbulance", "scima:hasStatus") not in CHANGE_RATE_PROFILES
    assert half_life_for(onto, "scima:AirAmbulance", "scima:hasStatus") == 90.0
    assert half_life_for(onto, "scima:LadderTruck", "scima:hasStatus") == 210.0
    assert half_life_for(onto, "scima:Ambulance", "scima:hasStatus") == 90.0


def test_a_rate_belongs_to_a_class_and_predicate_pair(onto):
    """The same predicate moves at different speeds on different classes."""
    ambulance = half_life_for(onto, "scima:Ambulance", "scima:hasStatus")
    crew = half_life_for(onto, "scima:MaintenanceCrew", "scima:hasStatus")
    assert ambulance == 90.0
    assert crew == 1500.0
    assert crew > ambulance


def test_plan_a_preconditions_all_read_true_yet_the_plan_is_a_coin_flip(plans):
    plan_a = plans[0]
    confs = [round(p.confidence(), 4) for p in plan_a.preconditions]
    assert confs == [0.8249, 0.5434, 0.8351]
    assert round(plan_a.success_probability(), 3) == 0.374


def test_the_wait_is_what_does_the_damage(plans):
    """The bridge reading would be worth 0.76 if the drive started now,
    and is worth 0.54 by the time it actually starts."""
    bridge = plans[0].preconditions[1]
    at_plan_time = belief_weight(bridge.half_life_s, bridge.evidence_age_s)
    at_execution = bridge.confidence()
    assert round(at_plan_time, 4) == 0.7579
    assert round(at_execution, 4) == 0.5434


def test_expected_duration_matches_the_articles_arithmetic():
    assert round(expected_duration(12.6, [(90, 25, 0, 3.5),
                                          (900, 360, 432, 5.5),
                                          (3600, 180, 756, 6.0)]), 2) == 16.71


def test_the_three_plans_are_priced_as_published(plans):
    nominal = [round(p.nominal_min, 2) for p in plans]
    success = [round(p.success_probability(), 3) for p in plans]
    expected = [round(p.expected_min(), 2) for p in plans]
    assert nominal == [12.60, 13.00, 13.13]
    assert success == [0.374, 0.632, 0.932]
    assert expected == [16.71, 14.99, 13.54]


def test_the_ranking_reverses_between_nominal_and_expected():
    """The whole article in one assertion."""
    nominal_order = [p.name[0] for p in rank_plans(by="nominal")]
    expected_order = [p.name[0] for p in rank_plans(by="expected")]
    assert nominal_order == ["A", "B", "C"]
    assert expected_order == ["C", "B", "A"]
    assert expected_order == list(reversed(nominal_order))


def test_only_the_fastest_looking_plan_misses_the_deadline(plans):
    deadline = transport_deadline_min()
    plan_a, plan_b, plan_c = plans
    assert plan_a.nominal_min == min(p.nominal_min for p in plans)
    assert plan_a.expected_min() > deadline
    assert round(plan_a.expected_min() - deadline, 2) == 0.71
    assert plan_b.expected_min() < deadline
    assert plan_c.expected_min() < deadline
    assert round(deadline - plan_c.expected_min(), 2) == 2.46


def test_verification_is_worth_more_the_later_it_runs():
    early, late = verification_value(900, 360, 432, 5.5)
    assert round(early, 3) == 0.955
    assert round(late, 3) == 2.511
    assert round(late / early, 2) == 2.63


# ---------------------------------------------------------------------------
# Section 5: plan shelf life
# ---------------------------------------------------------------------------
def test_plan_b_goes_stale_in_seventeen_seconds(plans):
    plan_b = plans[1]
    assert round(plan_b.shelf_life_s(), 1) == 16.9


def test_the_shelf_life_is_set_by_the_fastest_moving_precondition(plans):
    label, share = plans[1].dominant_rate_share()
    assert label == "AIR-3 available"
    assert round(100 * share, 1) == 93.0


def test_one_radio_call_extends_the_window_twenty_twofold(plans):
    plan_b = plans[1]
    confirmed = plan_b.confirm("AIR-3 available")
    assert round(confirmed.success_probability(), 4) == 0.6830
    assert round(confirmed.shelf_life_s(), 0) == 375.0
    assert round(confirmed.shelf_life_s() / plan_b.shelf_life_s(), 1) == 22.2


def test_plan_shelf_life_closed_form_matches_the_object():
    assert round(plan_shelf_life(0.632391, [0.007702, 0.000385, 0.000193]), 1) == 16.9
    assert round(plan_shelf_life(0.683020, [0.000385, 0.000193]), 0) == 375.0


# ---------------------------------------------------------------------------
# C2: the rate is not a constant
# ---------------------------------------------------------------------------
def test_a_surge_pushes_plan_b_below_its_own_commit_floor(plans):
    plan_b = plans[1]
    normal = plan_b.success_probability()
    surged = plan_b.success_probability(SURGE_REGIME)
    assert round(normal, 3) == 0.632
    assert round(surged, 3) == 0.542
    assert normal >= COMMIT_FLOOR
    assert surged < COMMIT_FLOOR


def test_the_surge_leaves_the_slow_preconditions_alone(plans):
    """A surge changes how fast vehicles flip, not how fast obstructions
    appear on a helipad, which is why a regime is a per-predicate map
    rather than one multiplier applied to everything."""
    status, landing, hospital = plans[1].preconditions
    # The status term moves.
    assert round(status.confidence(), 4) == 0.9259
    assert round(status.confidence(SURGE_REGIME), 4) == 0.7937
    # The other two do not.
    assert landing.confidence() == landing.confidence(SURGE_REGIME)
    assert hospital.confidence() == hospital.confidence(SURGE_REGIME)
    assert round(landing.confidence(), 4) == 0.8217


def test_a_frozen_rate_reports_a_committable_plan_that_is_not(plans):
    """The failure mode with nothing raising an alarm."""
    plan_b = plans[1]
    reported = plan_b.success_probability()               # peacetime lambda
    actual = plan_b.success_probability(SURGE_REGIME)
    assert reported > actual
    assert reported >= COMMIT_FLOOR > actual


# ---------------------------------------------------------------------------
# Section 6: constraints derived from axioms
# ---------------------------------------------------------------------------
def test_mutex_pairs_are_derived_not_tabulated(onto):
    pairs = derive_mutex(onto)
    assert len(pairs) == 2
    reasons = sorted(r for _a, _b, r in pairs)
    assert reasons == [
        "functional property scima:hasValveState",
        "maxQualifiedCardinality 1 on scima:assignedResource",
    ]


def test_the_cross_branch_mutex_is_found(onto):
    """Neither method knew about the other. The reasoner caught it."""
    pairs = derive_mutex(onto)
    found = {frozenset((a, b)) for a, b, _r in pairs}
    assert frozenset(("IsolateValve(V-12)", "OpenHydrant(V-12)")) in found
    assert frozenset(("Dispatch(AMB-17)", "Dispatch(AIR-3)")) in found


def test_eight_proposed_actions_leave_seven_after_mutex(onto):
    assert len(PROPOSED_EFFECTS) == 8
    assert len(PROPOSED_EFFECTS) * (len(PROPOSED_EFFECTS) - 1) // 2 == 28
    # One pair is resolved by dropping OpenHydrant, leaving the 7 in the
    # published tree; the other is a genuine choice between two vehicles.
    assert len(schedule()) == 7


def test_the_universal_restriction_eliminates_the_fire_truck(onto):
    assert violates_resource_type(onto, "scima:FireTruck") is True
    assert violates_resource_type(onto, "scima:AirAmbulance") is False
    assert violates_resource_type(onto, "scima:Ambulance") is False


# ---------------------------------------------------------------------------
# Section 7: the quarantined precondition
# ---------------------------------------------------------------------------
def test_a_quarantined_fact_is_unknown_rather_than_false():
    q = QuarantinedFact("scima:AIR_3", "scima:stationedAt",
                        "scima:Helipad_Hilltop", 0.88, 400.0, 90 * 86_400.0)
    assert round(q.belief(), 3) == 0.880
    # A roster does not move in seven minutes, which is what the rate
    # table is there to encode.
    assert q.belief() < q.extraction_confidence
    assert q.belief() > 0.87


def test_resolution_beats_carrying_at_this_belief():
    assert round(breakeven_belief(), 4) == 0.9333
    assert breakeven_belief() == 1 - QUARANTINE_RESOLUTION_MIN / QUARANTINE_RECOVERY_MIN
    assert resolve_or_carry(0.880) == "resolve"
    assert resolve_or_carry(0.95) == "carry"


def test_the_resolve_carry_comparison_is_priced(plans):
    plan_b = plans[1]
    belief = 0.880
    carry = plan_b.expected_min() + (1 - belief) * QUARANTINE_RECOVERY_MIN
    resolve = plan_b.expected_min() + QUARANTINE_RESOLUTION_MIN
    assert round(carry, 2) == 15.71
    assert round(resolve, 2) == 15.39
    assert resolve < carry


# ---------------------------------------------------------------------------
# C4 and C5: naming the operator, and naming the frame
# ---------------------------------------------------------------------------
def test_the_same_contradiction_is_an_update_or_a_revision_by_age():
    """Identical evidence, opposite operators, decided by how long the
    belief had to move."""
    late = classify_contradiction(900, 792)
    early = classify_contradiction(900, 20)
    assert round(late, 3) == 0.889
    assert round(early, 3) == 0.129
    assert late > 0.5 > early


def test_a_roster_contradiction_is_almost_certainly_a_revision():
    """stationedAt has a 90-day half-life, so in seven minutes the world
    did not move and the stored triple was simply wrong."""
    p_update = classify_contradiction(90 * 86_400.0, 400.0)
    assert p_update < 0.01


def test_frame_provenance_cuts_re_verification(onto):
    assert round(100 * frame_scoped_recheck(7, 2)) == 71
    assert "scima:checkedAgainstFrame" in onto.object_properties()
    assert "scima:checkedAt" in onto.datatype_properties()
