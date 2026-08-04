"""Article 10: knowledge graphs for cost-efficient agent search.

Asserts three contracts:

  * SCIMA-OWL v1.0 matches the Growth Tracker (43 classes, 53 properties,
    21 axioms) and still contains everything v0.8 declared.
  * The search behaves as the article says: the predicate set is derived
    rather than typed, the guided walk visits 12 nodes (11 with the
    admissible bound), and it returns AMB-17 at 4.2 minutes.
  * Every number in the article's cost and cache tables is reproducible
    from the constants in one place.
"""

import math

import pytest

from scima.agent_search import (
    ANSWER_TOKENS,
    CITY_EDGES,
    CITY_NODES,
    DECISION_BUDGET_SEC,
    FLEET_SIZE,
    HALF_LIFE_SEC,
    REQUEST_TOKENS,
    STATION_KM,
    STATUS_AGE_SEC,
    ROAD_ETA_MIN,
    RESPONSE_SPEED_KMH,
    RE_DISPATCH_COST_MIN,
    SURGE_RATE_MULTIPLIER,
    CostModel,
    DispatchScene,
    PredicateCache,
    belief_weight,
    blind_frontier,
    decay_rate,
    expected_time,
    fresh_hops_per_query,
    guided_walk,
    hit_rate,
    path_confidence,
    rank_candidates,
    schema_predicate_set,
    schema_routes,
    stale_answer_penalty,
    stale_dispatch_rate,
    ttl_for,
)
from scima.ontology import ScimaOntology


# ---------------------------------------------------------------------------
# SCIMA-OWL v1.0: the Growth Tracker contract
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v1.0")


def test_v1_0_matches_growth_tracker(onto):
    s = onto.summary()
    assert s.n_classes == 43
    assert s.n_object_properties == 30
    assert s.n_datatype_properties == 23
    assert s.n_properties == 53
    assert s.n_axioms == 21


def test_v1_0_is_backward_compatible_with_v0_8(onto):
    older = ScimaOntology.load("v0.8")
    assert set(older.classes()) <= set(onto.classes())
    assert set(older.object_properties()) <= set(onto.object_properties())
    assert set(older.datatype_properties()) <= set(onto.datatype_properties())


def test_article_8_temporal_vocabulary_present(onto):
    for cls in ("scima:KnowledgeFrame", "scima:BeliefState", "scima:RevisionEvent"):
        assert cls in onto.classes()
    for prop in ("scima:validDuring", "scima:wasRevisedFrom", "scima:describesEntity"):
        assert prop in onto.object_properties()


def test_emergency_vehicle_taxonomy_has_seven_kinds(onto):
    """The article's seven vehicle types, one of them declared late."""
    direct = set(onto.subclasses_of("scima:EmergencyVehicle"))
    assert direct == {
        "scima:Ambulance", "scima:FireTruck",
        "scima:PoliceVehicle", "scima:HazmatUnit", "scima:RescueBoat",
    }
    # AirAmbulance sits under Ambulance and LadderTruck under FireTruck, so
    # the taxonomy names seven kinds in total.
    assert onto.subclasses_of("scima:Ambulance") == ["scima:AirAmbulance"]
    assert onto.subclasses_of("scima:FireTruck") == ["scima:LadderTruck"]


# ---------------------------------------------------------------------------
# Section 4: the predicate set is derived from the schema, not typed by hand
# ---------------------------------------------------------------------------
def test_schema_routes_are_a_small_slice_of_the_vocabulary(onto):
    routes = schema_routes(onto, "scima:Incident", "scima:EmergencyVehicle")
    assert len(routes.predicates) == 10
    assert len(routes.classes) == 8
    # 10 of 53 is under a fifth of the vocabulary; the rest are dead ends.
    assert len(routes.predicates) / onto.summary().n_properties < 0.20


def test_derived_set_contains_the_dispatch_route(onto):
    via = schema_predicate_set(onto, "scima:Incident", "scima:EmergencyVehicle")
    for p in ("scima:withinZone", "scima:servedByStation", "scima:stationedAt",
              "scima:adjacentTo", "scima:affects"):
        assert p in via


def test_derived_set_excludes_provable_dead_ends(onto):
    via = schema_predicate_set(onto, "scima:Incident", "scima:EmergencyVehicle")
    # No route from an incident reaches a vehicle through a sensor reading,
    # a protocol, or a transit stop within the hop budget.
    for p in ("scima:hasReading", "scima:followsProtocol", "scima:servesStop",
              "scima:designates", "scima:extractedFrom"):
        assert p not in via


def test_derivation_adapts_when_the_goal_changes(onto):
    to_vehicle = set(schema_predicate_set(onto, "scima:Incident",
                                          "scima:EmergencyVehicle"))
    to_sensor = set(schema_predicate_set(onto, "scima:Incident",
                                         "scima:SensorDevice"))
    assert to_vehicle != to_sensor
    assert "scima:stationedAt" not in to_sensor


# ---------------------------------------------------------------------------
# Section 3: the relation-guided walk
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene():
    return DispatchScene()


@pytest.fixture(scope="module")
def via(onto):
    return schema_predicate_set(onto, "scima:Incident", "scima:EmergencyVehicle")


def test_guided_walk_visits_twelve_nodes(scene, via):
    r = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance", via,
                    prune=False)
    assert r.visited == 12
    assert r.best == "scima:AMB_17"
    assert r.cost == pytest.approx(4.2)


def test_admissible_bound_prunes_one_branch(scene, via):
    r = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance", via,
                    prune=True)
    assert r.visited == 11          # Hilltop's AMB-31 is never expanded
    assert r.best == "scima:AMB_17"
    assert r.cost == pytest.approx(4.2)


def test_the_bound_really_is_admissible(scene):
    """Straight-line time never exceeds the road ETA it bounds.

    This is the property that lets A* prune a branch and stay correct. If it
    ever fails, the pruning above becomes unsafe.
    """
    station_of = {
        "scima:AMB_17": "scima:Station_Riverside",
        "scima:AMB_22": "scima:Station_Riverside",
        "scima:FIRE_12": "scima:Station_Riverside",
        "scima:AMB_04": "scima:Station_Crosstown",
        "scima:AMB_09": "scima:Station_Crosstown",
        "scima:AMB_31": "scima:Station_Hilltop",
    }
    for vehicle, station in station_of.items():
        bound = STATION_KM[station] / RESPONSE_SPEED_KMH * 60
        assert bound <= ROAD_ETA_MIN[vehicle]


def test_greedy_beam_of_one_picks_the_wrong_ambulance(scene, via):
    """Crosstown is nearest in a straight line and slowest by road.

    A beam of 1 commits to it and returns AMB-09 at 6.1 minutes, losing 1.9
    minutes of response time. Widening the beam to 3 costs four extra node
    visits and recovers the true best.
    """
    greedy = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance", via,
                         beam_width=1, prune=False)
    wide = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance", via,
                       beam_width=3, prune=False)
    assert greedy.best == "scima:AMB_09"
    assert greedy.cost == pytest.approx(6.1)
    assert wide.best == "scima:AMB_17"
    assert wide.cost - greedy.cost == pytest.approx(-1.9)
    assert wide.visited - greedy.visited == 4


def test_goal_test_honours_the_class_hierarchy(scene):
    """Asking for an Ambulance must match an AirAmbulance too."""
    assert scene.is_instance_of("scima:AMB_17", "scima:Ambulance")
    assert scene.is_instance_of("scima:AMB_17", "scima:EmergencyVehicle")
    assert scene.is_instance_of("scima:AMB_17", "scima:Vehicle")
    assert not scene.is_instance_of("scima:FIRE_12", "scima:Ambulance")
    assert scene.is_instance_of("scima:FIRE_12", "scima:EmergencyVehicle")


def test_unavailable_vehicles_are_never_returned(scene, via):
    r = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance", via)
    assert scene.available(r.best)
    assert not scene.available("scima:AMB_22")   # out of service
    assert not scene.available("scima:AMB_04")   # en route elsewhere


def test_blind_frontier_matches_the_article(scene):
    levels, total = blind_frontier(3, CITY_NODES, CITY_EDGES)
    assert levels == [8, 56, 392]
    assert total == 456


# ---------------------------------------------------------------------------
# Section 5: cache lifetimes derived from measured change rates
# ---------------------------------------------------------------------------
def test_ttl_from_half_life_and_confidence_floor():
    assert ttl_for("scima:hasStatus", 0.80) == pytest.approx(28.97, abs=0.01)
    # A floor of one half is by definition reached at the half-life.
    assert ttl_for("scima:hasStatus", 0.50) == pytest.approx(90.0)
    assert ttl_for("scima:hasStatus", 0.90) == pytest.approx(13.68, abs=0.01)
    assert ttl_for("scima:hasLocation", 0.80) == pytest.approx(96.58, abs=0.01)
    assert ttl_for("scima:stationedAt", 0.80) / 86400 == pytest.approx(29.0, abs=0.1)
    assert ttl_for("scima:hasSpeedLimit") == math.inf


def test_tighter_floor_means_shorter_life_for_every_predicate():
    for p in HALF_LIFE_SEC:
        if HALF_LIFE_SEC[p] == math.inf:
            continue
        assert ttl_for(p, 0.90) < ttl_for(p, 0.80) < ttl_for(p, 0.50)


def test_hit_rates_and_fresh_hops():
    assert hit_rate(ttl_for("scima:hasStatus", 0.80)) == pytest.approx(0.741, abs=0.001)
    assert hit_rate(ttl_for("scima:hasStatus", 0.50)) == pytest.approx(0.917, abs=0.001)
    assert fresh_hops_per_query(0.80) == pytest.approx(0.795, abs=0.005)
    assert 12 / fresh_hops_per_query(0.80) == pytest.approx(15, abs=0.5)
    assert 12 / fresh_hops_per_query(0.50) == pytest.approx(45, abs=1.0)


def test_stale_answers_are_priced_not_wished_away():
    penalty_min = stale_answer_penalty(0.80)
    assert penalty_min * 60 == pytest.approx(11.1, abs=0.2)   # seconds
    # A 20-minute response target, so the price of caching is under 1 percent.
    assert penalty_min / 20 < 0.01
    # Tightening the floor roughly halves the penalty.
    assert stale_answer_penalty(0.90) * 60 == pytest.approx(5.4, abs=0.3)


# -------------------------------------------------------------------------
# Section 5: from an average penalty to a per-candidate one
# -------------------------------------------------------------------------
def test_the_forgetting_factor_is_the_transition_law_in_closed_form():
    """lambda is an output of the transition law, not a free setting.

    For a two-state available/busy chain with per-minute probability q of
    leaving available, (1 - q)^d = exp(-lambda d) with lambda = -ln(1 - q).
    The 90-second half-life corresponds to q = 0.37 per minute, so the two
    descriptions are the same number reached from opposite directions.
    """
    lam_per_min = decay_rate("scima:hasStatus") * 60
    q = 1 - math.exp(-lam_per_min)
    assert round(q, 4) == 0.37
    assert -math.log(1 - q) == pytest.approx(lam_per_min)
    # And the half-life falls back out of it.
    assert math.log(2) / lam_per_min * 60 == pytest.approx(90.0)


def test_expected_time_matches_the_article():
    """E[T] = t + (1 - w) * 3.5, worked for the two rival candidates."""
    assert belief_weight("scima:hasStatus", 25) == pytest.approx(0.825, abs=5e-4)
    assert belief_weight("scima:hasStatus", 3) == pytest.approx(0.977, abs=5e-4)
    assert belief_weight("scima:hasStatus", 60) == pytest.approx(0.630, abs=5e-4)

    assert expected_time(4.2, 25) == pytest.approx(4.81, abs=0.005)
    assert expected_time(5.0, 3) == pytest.approx(5.08, abs=0.005)
    assert expected_time(4.2, 60) == pytest.approx(5.50, abs=0.005)


def test_fresh_evidence_never_hurts_a_candidate():
    """The penalty is monotone in age, so it can only ever push a candidate
    down the list, never up."""
    ages = [0, 5, 15, 30, 60, 120]
    times = [expected_time(4.2, a) for a in ages]
    assert times == sorted(times)
    assert times[0] == pytest.approx(4.2)          # no age, no penalty
    # The penalty is bounded by one re-dispatch, however stale it gets, and
    # it approaches that ceiling rather than running past it.
    for age in ages:
        assert expected_time(4.2, age) <= 4.2 + RE_DISPATCH_COST_MIN
    assert expected_time(4.2, 10_000) == pytest.approx(
        4.2 + RE_DISPATCH_COST_MIN)


def test_expected_time_ranking_flips_when_the_evidence_ages(scene):
    """Raw ETA order and expected-time order are different lists."""
    fresh = rank_candidates(scene)
    assert [c.vehicle for c in fresh] == [
        "scima:AMB_17", "scima:AMB_31", "scima:AMB_09"]
    assert fresh[0].expected_min == pytest.approx(4.81, abs=0.005)
    assert fresh[1].expected_min == pytest.approx(5.08, abs=0.005)
    # On raw ETA alone AMB-17 leads by 0.8 minutes; on expected time, 0.27.
    assert fresh[1].eta_min - fresh[0].eta_min == pytest.approx(0.8)
    assert fresh[1].expected_min - fresh[0].expected_min == pytest.approx(
        0.27, abs=0.01)

    stale = dict(STATUS_AGE_SEC)
    stale["scima:AMB_17"] = 60.0
    aged = rank_candidates(scene, stale)
    assert [c.vehicle for c in aged][:2] == ["scima:AMB_31", "scima:AMB_17"]
    assert aged[0].expected_min == pytest.approx(5.08, abs=0.005)
    assert aged[1].expected_min == pytest.approx(5.50, abs=0.005)
    # The road network did not move. Only the belief did.
    etas = {c.vehicle: c.eta_min for c in aged}
    assert etas == {c.vehicle: c.eta_min for c in fresh}


def test_unavailable_and_wrong_class_vehicles_stay_out_of_the_ranking(scene):
    ranked = {c.vehicle for c in rank_candidates(scene)}
    assert "scima:AMB_22" not in ranked      # out of service
    assert "scima:AMB_04" not in ranked      # en route
    assert "scima:FIRE_12" not in ranked     # a fire truck, not an ambulance


def test_the_bound_stays_admissible_under_the_expected_time_objective(scene):
    """The penalty term is never negative, so a lower bound on road time is
    still a lower bound on expected time. Pruning keeps its guarantee."""
    station_of = {
        "scima:AMB_17": "scima:Station_Riverside",
        "scima:AMB_09": "scima:Station_Crosstown",
        "scima:AMB_31": "scima:Station_Hilltop",
    }
    for c in rank_candidates(scene):
        bound = STATION_KM[station_of[c.vehicle]] / RESPONSE_SPEED_KMH * 60
        assert bound <= c.eta_min <= c.expected_min


def test_pruning_weakens_when_the_incumbent_is_uncertain(scene):
    """Hilltop's 4.80-minute bound loses to a certain 4.2 and survives an
    uncertain 5.50, so stale evidence widens the search."""
    hilltop_bound = STATION_KM["scima:Station_Hilltop"] / RESPONSE_SPEED_KMH * 60
    assert hilltop_bound == pytest.approx(4.80)

    assert hilltop_bound > 4.2                       # Section 3, pruned
    assert hilltop_bound < expected_time(4.2, 25)    # margin already gone
    assert hilltop_bound < expected_time(4.2, 60)    # and gone by a mile

    # The incumbent's expected time rises monotonically with its own
    # staleness, so the pruning threshold only ever loosens.
    thresholds = [expected_time(4.2, a) for a in (0, 25, 60, 120)]
    assert thresholds == sorted(thresholds)


def test_path_confidence_is_the_product_over_live_hops():
    live = {"scima:hasStatus": 25.0, "scima:hasLocation": 25.0}
    assert path_confidence(live) == pytest.approx(0.779, abs=5e-4)
    # It really is the product, not an average or a minimum.
    assert path_confidence(live) == pytest.approx(
        belief_weight("scima:hasStatus", 25)
        * belief_weight("scima:hasLocation", 25))
    # Topology hops are slow enough to contribute essentially nothing. A
    # 90-day half-life gives up a couple of parts per million over 25 s.
    assert belief_weight("scima:stationedAt", 25.0) > 0.999_99
    with_topology = dict(live, **{"scima:stationedAt": 25.0})
    assert path_confidence(with_topology) == pytest.approx(
        path_confidence(live), rel=1e-5)
    # A predicate that never expires cannot lower confidence at all.
    assert belief_weight("scima:hasSpeedLimit", 10_000) == 1.0


# -------------------------------------------------------------------------
# Section 5: the rate of change is itself not constant
# -------------------------------------------------------------------------
def test_a_frozen_lifetime_nearly_triples_stale_dispatches():
    """The failure mode is silent. Nothing in the cache reports a problem."""
    floor = 0.80
    frozen = ttl_for("scima:hasStatus", floor)
    assert frozen == pytest.approx(28.97, abs=0.01)

    normal = stale_dispatch_rate(floor, 1.0, frozen)
    surged = stale_dispatch_rate(floor, SURGE_RATE_MULTIPLIER, frozen)
    assert normal == pytest.approx(0.053, abs=5e-4)
    assert surged == pytest.approx(0.142, abs=5e-4)
    assert surged / normal == pytest.approx(2.7, abs=0.05)

    # Priced, that is 11 seconds against about 30.
    assert stale_answer_penalty(floor, 1.0, frozen) * 60 == pytest.approx(
        11.1, abs=0.2)
    assert stale_answer_penalty(
        floor, SURGE_RATE_MULTIPLIER, frozen) * 60 == pytest.approx(29.9, abs=0.3)


def test_rederiving_the_lifetime_restores_the_error_rate():
    floor = 0.80
    surge_ttl = ttl_for("scima:hasStatus", floor, SURGE_RATE_MULTIPLIER)
    assert surge_ttl == pytest.approx(9.66, abs=0.01)
    # Three times the rate, one third of the lifetime.
    assert ttl_for("scima:hasStatus", floor) / surge_ttl == pytest.approx(
        SURGE_RATE_MULTIPLIER)
    assert stale_dispatch_rate(floor, SURGE_RATE_MULTIPLIER) == pytest.approx(
        stale_dispatch_rate(floor))


def test_the_stale_rate_depends_only_on_the_confidence_floor():
    """(1 - sqrt(f)) / 2, with lambda cancelled out of the algebra.

    This is why the floor is the knob to hold fixed and the lifetime is the
    thing that has to move underneath it.
    """
    for floor in (0.50, 0.80, 0.90, 0.95):
        closed_form = (1 - math.sqrt(floor)) / 2
        for multiplier in (0.5, 1.0, 3.0, 17.0):
            assert stale_dispatch_rate(floor, multiplier) == pytest.approx(
                closed_form)
            # And it agrees with pricing the re-derived lifetime the long way.
            ttl = ttl_for("scima:hasStatus", floor, multiplier)
            assert stale_dispatch_rate(floor, multiplier, ttl) == pytest.approx(
                closed_form)
    # A tighter floor really does buy a lower error rate.
    assert stale_dispatch_rate(0.90) < stale_dispatch_rate(0.80)


def test_a_surge_collapses_the_cache_hit_rate():
    """A cache is a bet that the world is holding still, so it should pay
    much less when the world speeds up."""
    floor = 0.80
    normal_hits = hit_rate(ttl_for("scima:hasStatus", floor))
    surge_hits = hit_rate(ttl_for("scima:hasStatus", floor, SURGE_RATE_MULTIPLIER))
    assert normal_hits == pytest.approx(0.741, abs=5e-4)
    assert surge_hits == pytest.approx(0.223, abs=5e-4)

    normal_fresh = fresh_hops_per_query(floor)
    surge_fresh = fresh_hops_per_query(floor, SURGE_RATE_MULTIPLIER)
    assert normal_fresh == pytest.approx(0.80, abs=0.005)
    assert surge_fresh == pytest.approx(2.35, abs=0.005)
    assert 12 / normal_fresh == pytest.approx(15.1, abs=0.1)
    assert 12 / surge_fresh == pytest.approx(5.1, abs=0.1)
    # Still better than walking all 12, just by much less.
    assert surge_fresh < 12


def test_write_invalidation_drops_only_what_depended_on_the_write():
    cache = PredicateCache()
    walk_key = cache.key("scima:Incident_I204", "scima:Ambulance",
                         ["scima:withinZone", "scima:stationedAt", "scima:hasStatus"],
                         3, 0.80)
    topo_key = cache.key("scima:ControlZone_Z7", "scima:Station",
                         ["scima:servedByStation"], 1, 0.80)
    cache.put(walk_key, "AMB-17", ["scima:withinZone", "scima:stationedAt",
                                   "scima:hasStatus"])
    cache.put(topo_key, ["Riverside", "Crosstown", "Hilltop"],
              ["scima:servedByStation"])

    assert cache.invalidate("scima:hasStatus") == 1
    assert cache.get(walk_key) is None
    assert cache.get(topo_key) is not None      # topology survives a status write


def test_cache_key_separates_different_freshness_rules():
    cache = PredicateCache()
    loose = cache.key("scima:Incident_I204", "scima:Ambulance",
                      ["scima:hasStatus"], 3, 0.50)
    tight = cache.key("scima:Incident_I204", "scima:Ambulance",
                      ["scima:hasStatus"], 3, 0.80)
    assert loose != tight


# ---------------------------------------------------------------------------
# Sections 1 and 8: the cost table
# ---------------------------------------------------------------------------
def test_looping_calls_grow_with_the_square_of_the_call_count():
    m = CostModel()
    assert m.looping_calls(FLEET_SIZE) == 39_952_000
    # Doubling the fleet roughly quadruples the token bill.
    ratio = m.looping_calls(2 * FLEET_SIZE) / m.looping_calls(FLEET_SIZE)
    assert 3.5 < ratio < 4.0


def test_cost_table_reproduces_the_article():
    rows = {r.name: r for r in CostModel().table()}

    loop = rows["scan every vehicle, one call each"]
    assert loop.input_tokens == 39_952_000
    assert loop.dollars == pytest.approx(120.58, abs=0.01)
    assert loop.seconds == pytest.approx(720.0)

    engine = rows["relation-guided walk, in the engine"]
    assert engine.input_tokens == 2_930
    assert engine.dollars == pytest.approx(0.010, abs=0.0005)
    assert engine.seconds == pytest.approx(0.96, abs=0.01)

    assert loop.input_tokens / engine.input_tokens == pytest.approx(13_635, rel=0.01)
    assert loop.dollars / engine.dollars == pytest.approx(12_058, rel=0.01)

    scan1 = rows["pull all 800 records in one call"]
    assert scan1.input_tokens == 98_000
    assert scan1.dollars == pytest.approx(0.295, abs=0.001)
    # The two independent factors that multiply out to the full spread.
    assert loop.dollars / scan1.dollars == pytest.approx(408, rel=0.02)
    assert scan1.dollars / engine.dollars == pytest.approx(29.5, rel=0.02)

    blind = rows["blind three-hop BFS"]
    assert blind.input_tokens == 56_720
    assert blind.dollars / engine.dollars == pytest.approx(17.2, rel=0.02)


def test_only_the_cheap_designs_meet_the_decision_budget():
    rows = {r.name: r for r in CostModel().table()}
    assert rows["scan every vehicle, one call each"].seconds > DECISION_BUDGET_SEC
    assert rows["blind three-hop BFS"].seconds > DECISION_BUDGET_SEC
    assert rows["relation-guided walk, agent-side"].seconds > DECISION_BUDGET_SEC
    assert rows["relation-guided walk, in the engine"].seconds <= DECISION_BUDGET_SEC
    assert rows["relation-guided walk, warm cache"].seconds <= DECISION_BUDGET_SEC


def test_cost_and_latency_do_not_rank_the_designs_the_same_way():
    """The single-call scan is 30x dearer and just as fast, and the
    agent-side walk is cheaper per hop and far slower. Optimising for one
    currency alone ships the wrong design."""
    rows = {r.name: r for r in CostModel().table()}
    scan1 = rows["pull all 800 records in one call"]
    agent_side = rows["relation-guided walk, agent-side"]
    assert scan1.dollars > agent_side.dollars * 2
    assert scan1.seconds < agent_side.seconds / 5


def test_caching_saves_graph_time_not_tokens():
    rows = {r.name: r for r in CostModel().table()}
    cold = rows["relation-guided walk, in the engine"]
    warm = rows["relation-guided walk, warm cache"]
    assert warm.input_tokens == cold.input_tokens      # no token saving at all
    assert warm.dollars == cold.dollars
    assert warm.seconds < cold.seconds
    # The whole latency saving is under 10 percent, because the model call
    # dominates a single query.
    assert (cold.seconds - warm.seconds) / cold.seconds < 0.10
