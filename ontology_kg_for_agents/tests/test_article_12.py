"""Article 12: continual learning, ontology and knowledge graph evolution.

Asserts four contracts:

  * SCIMA-OWL v1.5 matches the Growth Tracker (74 classes, 100
    properties, 41 axioms) and still contains everything v1.1 declared,
    including the class it deprecates.
  * The three thresholds are derived rather than chosen. Change a cost
    and the promotion floor moves. Make a class rarer and its
    retirement wait grows. Make an estimate noisier and the axiom
    retirement wait grows.
  * An axiom weight pooled over a year makes the dispatch agent act on
    a rule in the one season the rule does not hold.
  * Additions sort into safe and backward-reaching, a hand-typed list
    of type names loses the new class without reporting anything, and
    the compatibility gate refuses to pass until what the reasoner
    flags has been repaired.
"""

import math

import pytest

from scima.ontology import ScimaOntology
from scima.ontology_evolution import (
    ACTION_FLOOR,
    CHANCE_CONCEPT_LASTS,
    CHARGER_MENTION_RATE,
    COST_OF_RETIRING,
    COST_PER_HOMELESS_MENTION,
    EVACUATION_ZONE_RATE,
    EVACUATION_ZONE_SILENCE_MONTHS,
    MENTIONS_PER_MONTH,
    STORED_STATION_OPERATORS,
    TRAFFIC_JAM_WEIGHT,
    V1_5_ADDITIONS,
    ChangeLog,
    classify_addition,
    compatibility_gate,
    mine_axiom_candidates,
    missed_by_the_stale_list,
    pooled_weight,
    promotion_floor,
    promotion_month,
    quarters_before_retiring_axiom,
    seasonal_decisions,
    should_add_class,
    should_retire_class,
    silence_before_retiring,
    sites_by_hardcoded_list,
    sites_by_subclass_walk,
    spring_release,
    stations_violating_one_operator,
)


# ---------------------------------------------------------------------------
# The Growth Tracker is the contract
# ---------------------------------------------------------------------------

def test_v1_5_matches_the_growth_tracker():
    s = ScimaOntology.load("v1.5").summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (74, 100, 41)
    assert (s.n_object_properties, s.n_datatype_properties) == (53, 47)


def test_the_delta_over_v1_1_is_the_one_the_article_claims():
    old = ScimaOntology.load("v1.1").summary()
    new = ScimaOntology.load("v1.5").summary()
    assert new.n_classes - old.n_classes == 12
    assert new.n_properties - old.n_properties == 15
    assert new.n_axioms - old.n_axioms == 9


def test_nothing_v1_1_declared_was_removed():
    """The append-only rule, asserted rather than trusted."""
    old = ScimaOntology.load("v1.1")
    new = ScimaOntology.load("v1.5")
    for getter in ("classes", "object_properties", "datatype_properties"):
        missing = set(getattr(old, getter)()) - set(getattr(new, getter)())
        assert not missing, f"{getter} lost {sorted(missing)}"


def test_a_deprecated_class_is_still_a_class():
    """Retiring is not deleting. Old triples still have to read."""
    new = ScimaOntology.load("v1.5")
    assert "scima:EvacuationZone" in new.classes()
    from rdflib import OWL, Literal
    ez = new._to_ref("scima:EvacuationZone")
    assert (ez, OWL.deprecated, Literal(True)) in new.graph
    replaced = list(new.graph.objects(ez, new._to_ref("scima:replacedBy")))
    assert new._qnames(replaced) == ["scima:ControlZone"]


def test_the_drone_corridor_inherits_the_road_segment_machinery():
    """Placed under RoadSegment on purpose, so nothing is rewritten."""
    new = ScimaOntology.load("v1.5")
    assert "scima:DroneCorridorSegment" in new.subclasses_of("scima:RoadSegment")


# ---------------------------------------------------------------------------
# Section 2: the two thresholds are outputs, not settings
# ---------------------------------------------------------------------------

def test_the_promotion_floor_is_derived_from_the_two_costs():
    expected = ((1 - CHANCE_CONCEPT_LASTS) * COST_OF_RETIRING
                / (MENTIONS_PER_MONTH * COST_PER_HOMELESS_MENTION))
    assert promotion_floor() == pytest.approx(expected)
    assert promotion_floor() == pytest.approx(0.084375)


def test_march_waits_and_may_promotes():
    assert should_add_class(0.061) is False        # $976 against $1,350
    assert should_add_class(0.087) is True         # $1,392 against $1,350
    assert promotion_month() == len(CHARGER_MENTION_RATE) - 1


def test_a_cheaper_retirement_moves_the_floor_down_on_its_own():
    """Nobody edits a threshold. The cost changes and the floor follows."""
    dear = promotion_floor(cost_of_retiring=9_000.0)
    cheap = promotion_floor(cost_of_retiring=2_000.0)
    assert cheap < dear
    assert should_add_class(0.061, cost_of_retiring=2_000.0) is True


def test_silence_is_only_evidence_when_it_is_surprising():
    assert silence_before_retiring(0.5) == pytest.approx(math.log(100) / 0.5)
    assert silence_before_retiring(0.5) == pytest.approx(9.2103, abs=1e-4)
    assert should_retire_class(EVACUATION_ZONE_SILENCE_MONTHS,
                               EVACUATION_ZONE_RATE) is True


def test_a_rarer_class_earns_a_longer_wait():
    """The guard that stops the rule from retiring something healthy."""
    common = silence_before_retiring(0.5)
    rare = silence_before_retiring(0.05)
    assert rare == pytest.approx(10 * common)
    assert should_retire_class(EVACUATION_ZONE_SILENCE_MONTHS, 0.05) is False


def test_a_class_with_no_healthy_rate_cannot_be_tested():
    with pytest.raises(ValueError):
        silence_before_retiring(0.0)


# ---------------------------------------------------------------------------
# Section 3: an axiom weight is a process
# ---------------------------------------------------------------------------

def test_the_pooled_weight_hides_one_whole_season():
    assert pooled_weight() == pytest.approx(0.7525)
    assert pooled_weight() >= ACTION_FLOOR
    assert TRAFFIC_JAM_WEIGHT["summer"] < ACTION_FLOOR


def test_pooling_produces_exactly_one_wrong_quarter():
    wrong = [r["season"] for r in seasonal_decisions() if r["wrong"]]
    assert wrong == ["summer"]
    assert all(r["pooled_says_act"] for r in seasonal_decisions())


def test_a_noisier_estimate_makes_the_axiom_wait_longer():
    quiet = quarters_before_retiring_axiom(dip_probability=0.05)
    noisy = quarters_before_retiring_axiom(dip_probability=0.25)
    assert quiet < noisy
    assert noisy == 4                    # ln(0.01) / ln(0.25) = 3.32 -> 4
    assert 0.25 ** noisy < 0.01
    assert 0.25 ** (noisy - 1) > 0.01    # and four is the smallest that works


def test_a_dip_probability_outside_the_open_unit_interval_is_rejected():
    for bad in (0.0, 1.0, -0.1, 1.4):
        with pytest.raises(ValueError):
            quarters_before_retiring_axiom(dip_probability=bad)


# ---------------------------------------------------------------------------
# Section 4: safe additions, backward-reaching additions, the silent failure
# ---------------------------------------------------------------------------

def test_additions_sort_into_safe_and_backward_reaching():
    labels = [classify_addition(kind) for kind, _ in V1_5_ADDITIONS]
    assert labels.count("safe") == 6
    assert labels.count("reaches back") == 3


def test_widening_a_range_is_safe_and_narrowing_one_is_not():
    assert classify_addition("WidenRange") == "safe"
    assert classify_addition("NarrowRange") == "reaches back"


def test_an_unclassified_change_kind_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        classify_addition("RenameClassInPlace")


def test_the_cardinality_limit_condemns_data_that_was_legal_when_it_arrived():
    flagged = stations_violating_one_operator()
    assert flagged == ["scima:EVCS_02", "scima:EVCS_04", "scima:EVCS_05"]
    for station in flagged:
        assert len(STORED_STATION_OPERATORS[station]) > 1


def test_the_hand_typed_list_loses_the_new_class_and_reports_nothing():
    onto = ScimaOntology.load("v1.5")
    typed = sites_by_hardcoded_list(onto)
    walked = sites_by_subclass_walk(onto)
    assert "scima:EVChargingStation" not in typed
    assert "scima:EVChargingStation" in walked
    assert len(typed) < len(walked)          # fewer rows, no error
    assert "scima:EVChargingStation" in missed_by_the_stale_list()


def test_the_subclass_walk_includes_the_parent_itself():
    """The zero in 'zero or more times' is the part people miss."""
    onto = ScimaOntology.load("v1.5")
    assert "scima:InfrastructureEntity" in sites_by_subclass_walk(onto)


def test_the_walk_finds_the_new_class_at_v1_5_and_cannot_at_v1_1():
    old = sites_by_subclass_walk(ScimaOntology.load("v1.1"))
    new = sites_by_subclass_walk(ScimaOntology.load("v1.5"))
    assert "scima:EVChargingStation" not in old
    assert set(old) < set(new)


# ---------------------------------------------------------------------------
# Section 5: mined axioms are proposals
# ---------------------------------------------------------------------------

def test_only_a_steady_pattern_is_proposed():
    proposed = [c["pattern"] for c in mine_axiom_candidates() if c["propose"]]
    assert proposed == ["connector types must match"]


def test_a_pattern_that_is_high_in_one_window_is_not_enough():
    seasonal = {"holds in one quarter only": [0.99, 0.40, 0.42, 0.41]}
    assert mine_axiom_candidates(seasonal)[0]["propose"] is False


def test_nothing_is_ever_promoted_without_a_person():
    assert all(c["needs_human"] for c in mine_axiom_candidates())


# ---------------------------------------------------------------------------
# Sections 4 and 6: the gate and the change log
# ---------------------------------------------------------------------------

def test_the_gate_fails_until_the_flagged_triples_are_repaired():
    assert compatibility_gate(repair=False).passed is False
    assert compatibility_gate(repair=True).passed is True


def test_the_gate_loses_no_rows_because_nothing_was_removed():
    result = compatibility_gate()
    assert result.query_rows_lost == 0
    assert result.queries_replayed > 0
    assert len(result.flagged_by_reasoner) == 3


def test_the_change_log_carries_the_reading_that_caused_each_change():
    log = spring_release()
    assert len(log.changes) == 3
    addition = log.changes[0]
    assert addition["iri"] == "scima:EVChargingStation"
    assert addition["reading"] == CHARGER_MENTION_RATE[-1]
    assert addition["reading"] > promotion_floor()
    deprecation = log.changes[1]
    assert deprecation["replaced_by"] == "scima:ControlZone"
    assert deprecation["reading"] == EVACUATION_ZONE_SILENCE_MONTHS


def test_exactly_one_logged_change_reaches_backward():
    reaching = spring_release().backward_reaching()
    assert [c["iri"] for c in reaching] == ["scima:hasOperator"]


def test_an_empty_log_is_a_valid_release():
    log = ChangeLog("http://scima.city/ontology", "v1.5")
    assert log.changes == []
    assert log.backward_reaching() == []
