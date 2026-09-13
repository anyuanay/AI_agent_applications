"""Supplemental belief utilities from the earlier Article 13 treatment.

The revised article is implemented in ``source_actor_memory``.
Section references below refer to the earlier treatment.

Every module before this one wrote into one graph that the whole system
read. This one splits it, one graph per agent plus a shared graph, and
gets a place to put a disagreement.

The article's claim is that two agents holding different values for one
fact is a state rather than an error, and that the cheap explanations
have to be ruled out in a fixed order before anybody calls it a
conflict.

  * ``aged_belief`` / ``aged_distribution`` -- C5, every read names its
    frame. Two stored numbers were true at two different instants, so a
    coordinator that compares them is comparing two clocks. Run both
    forward to one instant first (Section 3).
  * ``publication_floor`` / ``should_publish`` -- the gate between a
    private graph and the shared one. A weak belief published reads to
    everybody else as agreed knowledge, so the floor is where the cost
    of a wrong shared fact meets the cost of a decision made without a
    right one (Section 2).
  * ``p_update`` / ``route_operator`` / ``operator_flip_age`` -- C4,
    every incoming fact names its operator. One radio call reaches two
    agents that copied the same shared value at different moments, and
    it is a revision for one and an update for the other. The two
    explanations meet at one age, where the chance the world moved
    equals the chance the source was wrong, and that age is a closed
    form (Section 4).
  * ``reopened_by`` -- what a revision costs. An intention records the
    frame it was formed under, so a revised fact re-opens the
    commitments that used it and leaves the rest alone (Sections 1
    and 6).
  * ``kl_divergence`` / ``chance_they_act_apart`` / ``worth_a_message``
    -- C3, queries run over beliefs. The divergence ranks agent pairs
    and the overlap prices the sync decision. Nobody votes and nobody
    averages (Section 5).
  * ``resolution_ladder`` -- the four questions in order. The running
    case leaves at question two, because the loop coil and the camera
    were watching two different approaches of one node (Section 6).

Usage:
    python -m scima.agent_beliefs --scene
    python -m scima.agent_beliefs --age
    python -m scima.agent_beliefs --publish
    python -m scima.agent_beliefs --operator
    python -m scima.agent_beliefs --divergence
    python -m scima.agent_beliefs --ladder
    python -m scima.agent_beliefs --diff
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rdflib import RDFS, URIRef

from scima.ontology import ScimaOntology

LN2 = math.log(2.0)

# --------------------------------------------------------------------------
# Named graphs. The fourth field on a quad, which is what holds a
# disagreement. Two of these are private and nobody else reads them.
# --------------------------------------------------------------------------

NORTH = "http://scima.city/beliefs/north/"
SOUTH = "http://scima.city/beliefs/south/"
SHARED = "http://scima.city/global/"

GRAPH_OWNER = {
    NORTH: "scima:ZoneAgent_North",
    SOUTH: "scima:ZoneAgent_South",
    SHARED: None,  # the shared graph has no owner, which is the point
}

# --------------------------------------------------------------------------
# How fast each predicate flips. This is a property of the world, so it
# belongs next to the predicate rather than in whoever reads it. The two
# that matter here differ by a factor of thirty, which is why one belief
# aged twenty seconds is fresh and another aged twenty seconds is not.
# --------------------------------------------------------------------------

HALF_LIFE_S = {
    "scima:hasCongestionLevel": 60.0,      # a city intersection, about a minute
    "scima:hasLaneStatus": 1800.0,         # a lane closes about twice an hour
    "scima:hasLocation": 97.0,             # a moving vehicle
    "scima:servedByStation": 838_000.0,    # roughly ten days
    "scima:hasStreetAddress": math.inf,    # does not fade
}
DEFAULT_HALF_LIFE_S = 300.0

#: How often a value copied out of the shared graph was wrong at the moment
#: it was copied. Measured from the publication log, not chosen.
SHARED_VALUE_ERROR_RATE = 0.05

#: What the city can quote for the publication floor. A wrong fact in the
#: shared graph is read by every agent; a right fact held back costs the
#: one decision that needed it.
COST_OF_A_WRONG_SHARED_FACT = 1_350.0
COST_OF_A_DECISION_MADE_WITHOUT_IT = 400.0

#: Sync cost against the cost of one agent acting on the wrong congestion
#: value, both in minutes of added response time.
COST_OF_SYNC_MIN = 0.20
COST_OF_WRONG_ACTION_MIN = 3.5

CONGESTION_VALUES = ("LIGHT", "MODERATE", "HEAVY")


# --------------------------------------------------------------------------
# The unit of storage. A triple says what. It does not say who thinks so.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """One agent's statement about one property of one entity.

    The fourth field, ``graph``, is the whole point. With it, the same
    statement appears in two agents' graphs with two beliefs and no
    contradiction.
    """
    subject: str
    predicate: str
    value: str
    belief: float
    seen_at: float          # seconds on a shared clock
    graph: str
    source: str
    measured_from: str | None = None   # the leg or entity the sensor watches
    distribution: tuple[float, ...] | None = None

    @property
    def half_life_s(self) -> float:
        return half_life_for(self.predicate)

    def age_at(self, now: float) -> float:
        return now - self.seen_at

    def aged(self, now: float) -> float:
        """This claim's belief run forward to ``now``.

        >>> north = SCENE["north_congestion"]
        >>> round(north.aged(NOW), 4)
        0.7857
        """
        return aged_belief(self.belief, self.age_at(now), self.half_life_s)

    def aged_dist(self, now: float) -> tuple[float, ...]:
        if self.distribution is None:
            raise ValueError(f"{self.subject} {self.predicate} carries no distribution")
        return aged_distribution(self.distribution, self.age_at(now), self.half_life_s)


@dataclass(frozen=True)
class Intention:
    """A plan the agent has committed to and stopped reconsidering.

    ``frame_facts`` is what ``scima:formedUnderFrame`` points at. Without
    it a revision has to re-check every commitment the agent holds.
    """
    label: str
    agent: str
    frame: str
    frame_facts: tuple[str, ...]
    committed_at: float


def half_life_for(predicate: str) -> float:
    """Seconds for a belief about this predicate to fade by half.

    >>> half_life_for("scima:hasCongestionLevel")
    60.0
    >>> half_life_for("scima:somethingNobodyMeasured")
    300.0
    """
    return HALF_LIFE_S.get(predicate, DEFAULT_HALF_LIFE_S)


# --------------------------------------------------------------------------
# Section 3 and C5: compare at one instant, never two
# --------------------------------------------------------------------------

def aged_belief(belief: float, age_s: float, half_life_s: float,
                midpoint: float = 0.5) -> float:
    """A belief run forward with no new evidence, fading toward no claim.

    Fading toward the midpoint rather than toward zero is the part people
    skip. An old belief does not become a belief in the opposite value.
    It becomes no belief at all.

    >>> round(aged_belief(0.86, 20.0, 60.0), 4)     # the loop, twenty seconds
    0.7857
    >>> round(aged_belief(0.79, 240.0, 60.0), 4)    # the camera, four minutes
    0.5181
    >>> aged_belief(0.90, 3600.0, math.inf)         # a street address
    0.9
    """
    if age_s < 0:
        raise ValueError("a belief cannot be read before it was formed")
    if math.isinf(half_life_s):
        return belief
    return midpoint + (belief - midpoint) * math.exp(-LN2 * age_s / half_life_s)


def aged_distribution(dist, age_s: float, half_life_s: float) -> tuple[float, ...]:
    """The same rule over a distribution, fading toward uniform.

    >>> tuple(round(x, 4) for x in aged_distribution((0.06, 0.15, 0.79), 240.0, 60.0))
    (0.3162, 0.3219, 0.3619)
    >>> round(sum(aged_distribution((0.06, 0.15, 0.79), 240.0, 60.0)), 10)
    1.0
    """
    n = len(dist)
    if n == 0:
        raise ValueError("an empty distribution has nothing to age")
    uniform = 1.0 / n
    if math.isinf(half_life_s):
        return tuple(dist)
    factor = math.exp(-LN2 * age_s / half_life_s)
    return tuple(uniform + (p - uniform) * factor for p in dist)


def compare_at(a: Claim, b: Claim, now: float) -> tuple[float, float]:
    """Both beliefs at one instant. Never compare the stored numbers.

    >>> tuple(round(x, 4) for x in compare_at(
    ...     SCENE["north_congestion"], SCENE["south_congestion"], NOW))
    (0.7857, 0.5181)
    """
    return a.aged(now), b.aged(now)


def gap_is_mostly_clock(a: Claim, b: Claim, now: float,
                        share: float = 0.5) -> bool:
    """True when aging closes more than ``share`` of the stored gap.

    >>> gap_is_mostly_clock(SCENE["north_congestion"], SCENE["south_congestion"], NOW)
    True
    """
    # The two claims assert opposite values, so subtracting one belief from
    # the other measures nothing. What shrinks with the clock is how far
    # each one is from a shared no-claim.
    stored_from_middle = abs(a.belief - 0.5) + abs(b.belief - 0.5)
    live_from_middle = abs(a.aged(now) - 0.5) + abs(b.aged(now) - 0.5)
    return live_from_middle < share * stored_from_middle


# --------------------------------------------------------------------------
# Section 2: the gate between a private graph and the shared one
# --------------------------------------------------------------------------

def publication_floor(
        cost_of_a_wrong_shared_fact: float = COST_OF_A_WRONG_SHARED_FACT,
        cost_of_a_decision_made_without_it: float = COST_OF_A_DECISION_MADE_WITHOUT_IT
) -> float:
    """The belief a claim needs before it earns a place in the shared graph.

    Publishing a wrong fact costs the group a wrong decision, weighted by
    the chance the claim is wrong. Holding a right fact back costs the
    group a decision made without it, weighted by the chance it is right.
    Setting the two equal and solving for the belief gives the floor, so
    it moves on its own when either cost moves.

    >>> round(publication_floor(), 4)
    0.7714
    >>> round(publication_floor(cost_of_a_wrong_shared_fact=200.0), 4)
    0.3333
    """
    total = cost_of_a_wrong_shared_fact + cost_of_a_decision_made_without_it
    if total <= 0:
        raise ValueError("at least one of the two costs has to be positive")
    return cost_of_a_wrong_shared_fact / total


def should_publish(belief: float, **costs) -> bool:
    """True when the claim clears this agent's own floor.

    >>> should_publish(0.86)     # the fresh loop reading
    True
    >>> should_publish(0.52)     # the camera frame, once it has aged
    False
    """
    return belief > publication_floor(**costs)


# --------------------------------------------------------------------------
# Section 4 and C4: one agent's update is another's revision
# --------------------------------------------------------------------------

def chance_world_moved(age_s: float, half_life_s: float) -> float:
    """P(the value changed at least once while it was held).

    >>> round(chance_world_moved(20.0, 1800.0), 5)
    0.00767
    >>> round(chance_world_moved(240.0, 1800.0), 5)
    0.08828
    """
    if math.isinf(half_life_s):
        return 0.0
    return 1.0 - math.exp(-LN2 * age_s / half_life_s)


def p_update(age_s: float, half_life_s: float,
             source_error_rate: float = SHARED_VALUE_ERROR_RATE) -> float:
    """P(this contradiction is the world moving) against P(a bad value).

    Exactly one of two things explains a contradicted belief. Either the
    world moved while the agent held the value, or the value was wrong at
    the moment it was written. Normalizing over those two gives the fork.

    >>> round(p_update(20.0, 1800.0), 4)      # synced twenty seconds ago
    0.1281
    >>> round(p_update(240.0, 1800.0), 4)     # synced four minutes ago
    0.6478
    """
    if not 0.0 < source_error_rate < 1.0:
        raise ValueError("a source error rate has to be strictly between 0 and 1")
    moved = chance_world_moved(age_s, half_life_s)
    world_moved = moved * (1.0 - source_error_rate)
    value_was_wrong = (1.0 - moved) * source_error_rate
    return world_moved / (world_moved + value_was_wrong)


def route_operator(age_s: float, half_life_s: float,
                   source_error_rate: float = SHARED_VALUE_ERROR_RATE) -> str:
    """'update' or 'revision'. The two have different cleanup costs.

    >>> route_operator(20.0, 1800.0)
    'revision'
    >>> route_operator(240.0, 1800.0)
    'update'
    """
    return "update" if p_update(age_s, half_life_s, source_error_rate) >= 0.5 \
        else "revision"


def operator_flip_age(half_life_s: float,
                      source_error_rate: float = SHARED_VALUE_ERROR_RATE) -> float:
    """The age where the fork turns over, in seconds.

    The two explanations are equally good when the chance the world moved
    equals the chance the source was wrong, so 1 - e^(-lambda * T) = e and
    T = h * ln(1 / (1 - e)) / ln 2. Nobody chooses this number. It falls
    out of the flip rate and the source's own error rate.

    >>> round(operator_flip_age(1800.0), 1)
    133.2
    >>> round(operator_flip_age(60.0), 2)          # a faster predicate
    4.44
    >>> round(operator_flip_age(1800.0, 0.20), 1)  # a worse source
    579.5
    """
    if math.isinf(half_life_s):
        return math.inf
    return half_life_s * math.log(1.0 / (1.0 - source_error_rate)) / LN2


def reopened_by(fact: str, operator: str, intentions,
                agent: str | None = None) -> list[str]:
    """Commitments a contradicted fact sends back to the planner.

    The operator is per agent, so this is too. An update re-opens
    nothing, because every conclusion was right about the world at the
    time it was drawn. A revision re-opens the intentions whose recorded
    frame names the bad fact, and only those. That is what
    ``scima:formedUnderFrame`` buys, and without it every commitment the
    agent holds has to be re-checked.

    >>> north = "scima:ZoneAgent_North"
    >>> reopened_by(LANE_FACT, "revision", INTENTIONS, north)
    ['route AMB-17 up Main']
    >>> reopened_by(LANE_FACT, "update", INTENTIONS, "scima:ZoneAgent_South")
    []
    >>> [i.label for i in INTENTIONS if i.agent == north]     # the rest are left alone
    ['route AMB-17 up Main', 'reroute the shuttle around the water main work']
    """
    if operator not in ("update", "revision"):
        raise ValueError(f"unknown operator {operator!r}")
    if operator == "update":
        return []
    return [i.label for i in intentions
            if fact in i.frame_facts and (agent is None or i.agent == agent)]


# --------------------------------------------------------------------------
# Section 5 and C3: measure the gap before you spend a message
# --------------------------------------------------------------------------

def kl_divergence(p, q) -> float:
    """How surprised an agent holding q would be by an agent holding p.

    Asymmetric on purpose, since a coordinator wants both directions.

    >>> round(kl_divergence((0.5, 0.5), (0.5, 0.5)), 10)
    0.0
    >>> n, s = aged_scene_distributions()
    >>> round(kl_divergence(n, s), 4), round(kl_divergence(s, n), 4)
    (0.4065, 0.4397)
    """
    if len(p) != len(q):
        raise ValueError("two distributions over different value sets cannot be compared")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            return math.inf
        total += pi * math.log(pi / qi)
    return total


def chance_they_act_apart(p, q) -> float:
    """One minus the overlap. Two agents can act apart only where they differ.

    >>> n, s = aged_scene_distributions()
    >>> round(chance_they_act_apart(n, s), 4)
    0.4351
    >>> chance_they_act_apart((0.3, 0.7), (0.3, 0.7))
    0.0
    """
    return 1.0 - sum(min(pi, qi) for pi, qi in zip(p, q))


def worth_a_message(p, q,
                    cost_of_wrong_action: float = COST_OF_WRONG_ACTION_MIN,
                    cost_of_sync: float = COST_OF_SYNC_MIN) -> bool:
    """Spend the message when the expected error beats the message.

    The divergence ranks agent pairs. This prices the decision, which is
    a different question and needs a different number.

    >>> n, s = aged_scene_distributions()
    >>> worth_a_message(n, s)
    True
    >>> worth_a_message((0.34, 0.33, 0.33), (0.33, 0.34, 0.33))
    False
    """
    return chance_they_act_apart(p, q) * cost_of_wrong_action > cost_of_sync


def divergence_trend(series, slack: float = 0.02) -> str:
    """'steady' or 'widening'. The trend carries more than the level.

    Steady divergence is information asymmetry, two agents with different
    evidence about one thing. Widening divergence is a symptom, and no
    amount of message passing repairs it.

    >>> divergence_trend([0.41, 0.40, 0.42, 0.41])
    'steady'
    >>> divergence_trend([0.11, 0.24, 0.38, 0.53])
    'widening'
    """
    if len(series) < 2:
        raise ValueError("a trend needs at least two readings")
    return "widening" if series[-1] - series[0] > slack * (len(series) - 1) \
        else "steady"


# --------------------------------------------------------------------------
# The scene. Main and Ninth, on the border between two zones.
# --------------------------------------------------------------------------

NOW = 1000.0

SCENE = {
    # Congestion. The two agents assert opposite values and both are right.
    "north_congestion": Claim(
        subject="scima:MainAndNinth", predicate="scima:hasCongestionLevel",
        value="LIGHT", belief=0.86, seen_at=NOW - 20.0, graph=NORTH,
        source="scima:LoopCoil_MN_SB", measured_from="scima:MainNinth_SB",
        distribution=(0.86, 0.10, 0.04)),
    "south_congestion": Claim(
        subject="scima:MainAndNinth", predicate="scima:hasCongestionLevel",
        value="HEAVY", belief=0.79, seen_at=NOW - 240.0, graph=SOUTH,
        source="scima:Camera_MN_EB", measured_from="scima:MainNinth_EB",
        distribution=(0.06, 0.15, 0.79)),
    # Lane status. Both agents copied one value out of the shared graph, at
    # different moments, and one radio call contradicts both copies.
    "north_lanes": Claim(
        subject="scima:Main_NB", predicate="scima:hasLaneStatus",
        value="OPEN", belief=0.90, seen_at=NOW - 20.0, graph=NORTH,
        source=SHARED, measured_from="scima:Main_NB"),
    "south_lanes": Claim(
        subject="scima:Main_NB", predicate="scima:hasLaneStatus",
        value="OPEN", belief=0.90, seen_at=NOW - 240.0, graph=SOUTH,
        source=SHARED, measured_from="scima:Main_NB"),
}

#: What the radio call says. It contradicts both copies above.
RADIO_CALL = ("scima:Main_NB", "scima:hasLaneStatus", "BLOCKED")

#: The fact both copies came from, named so a frame can point at it.
LANE_FACT = "main_nb_lanes_open"

INTENTIONS = (
    Intention("route AMB-17 up Main", "scima:ZoneAgent_North", "F-91",
              (LANE_FACT, "amb17_available"), NOW - 18.0),
    Intention("reroute the shuttle around the water main work",
              "scima:ZoneAgent_North", "F-92",
              ("watermain_v12_closed",), NOW - 300.0),
    Intention("hold green for the northbound platoon", "scima:ZoneAgent_South",
              "F-88", (LANE_FACT,), NOW - 235.0),
)

#: One node, four approaches. The old schema offered the first and not the
#: rest, so two correct measurements had to fight over one slot.
MAIN_AND_NINTH_LEGS = {
    "scima:MainNinth_NB": "MODERATE",
    "scima:MainNinth_SB": "LIGHT",
    "scima:MainNinth_EB": "HEAVY",
    "scima:MainNinth_WB": "LIGHT",
}


def aged_scene_distributions(now: float = NOW):
    """Both agents' congestion distributions, aged to one instant.

    >>> n, s = aged_scene_distributions()
    >>> tuple(round(x, 4) for x in n)
    (0.7513, 0.1481, 0.1005)
    >>> tuple(round(x, 4) for x in s)
    (0.3162, 0.3219, 0.3619)
    """
    return (SCENE["north_congestion"].aged_dist(now),
            SCENE["south_congestion"].aged_dist(now))


# --------------------------------------------------------------------------
# Section 3: the fact one gate admitted and another quarantined
# --------------------------------------------------------------------------

def admitted(values, functional: bool = True) -> tuple[bool, str]:
    """An agent's admission gate, and what it does with two values.

    A property allowed one value that arrives with two is held in
    quarantine, so the working graph carries no version of it at all. One
    agent's gate can quarantine what another agent admitted, and then the
    two disagree about whether the fact exists before they disagree about
    what it says.

    >>> admitted(["scima:Helipad_03"])
    (True, 'stored')
    >>> admitted(["scima:Helipad_03", "scima:Station_07"])
    (False, 'quarantined, two values for a functional property')
    """
    if functional and len(values) > 1:
        return False, "quarantined, two values for a functional property"
    if not values:
        return False, "absent, which is not the same as false"
    return True, "stored"


# --------------------------------------------------------------------------
# Section 6: the ladder, and the class that was missing
# --------------------------------------------------------------------------

def resolution_ladder(a: Claim, b: Claim, now: float = NOW) -> dict:
    """Four questions in order, and where this pair leaves the ladder.

    Skip question two and a schema bug becomes a permanent argument
    between two agents that are both working right.

    >>> out = resolution_ladder(SCENE["north_congestion"], SCENE["south_congestion"])
    >>> out["exits_at"], out["verdict"]
    (2, 'different entities, split the node')
    >>> out["steps"][0]["closed_it"]
    False
    """
    steps = []

    live_a, live_b = compare_at(a, b, now)
    closed = live_a < 0.55 and live_b < 0.55       # both faded to no claim
    steps.append({
        "question": "aged to the same instant?",
        "stored": (a.belief, b.belief),
        "live": (round(live_a, 4), round(live_b, 4)),
        "closed_it": closed,
    })
    if closed:
        return {"exits_at": 1, "verdict": "two clocks, not two opinions",
                "steps": steps}

    same_entity = a.measured_from == b.measured_from
    steps.append({
        "question": "same entity?",
        "measured_from": (a.measured_from, b.measured_from),
        "closed_it": not same_entity,
    })
    if not same_entity:
        return {"exits_at": 2, "verdict": "different entities, split the node",
                "steps": steps}

    ops = (route_operator(a.age_at(now), a.half_life_s),
           route_operator(b.age_at(now), b.half_life_s))
    steps.append({
        "question": "which operator on each side?",
        "operators": ops,
        "closed_it": ops[0] != ops[1],
    })
    if ops[0] != ops[1]:
        return {"exits_at": 3, "verdict": "one update and one revision, not a conflict",
                "steps": steps}

    steps.append({
        "question": "apart after all three?",
        "closed_it": False,
    })
    return {"exits_at": 4, "verdict": "a real conflict, weigh by belief and freshness",
            "steps": steps}


def rollup(legs=None, order=CONGESTION_VALUES) -> str:
    """Congestion for the whole intersection, computed when somebody asks.

    No agent writes this, so no agent can overwrite it.

    >>> rollup()
    'HEAVY'
    >>> rollup({"a": "LIGHT", "b": "LIGHT"})
    'LIGHT'
    """
    legs = MAIN_AND_NINTH_LEGS if legs is None else legs
    if not legs:
        raise ValueError("an intersection with no legs has no congestion")
    return max(legs.values(), key=order.index)


def superclasses_of(onto: ScimaOntology, cls: str) -> list[str]:
    """Walk rdfs:subClassOf upward, which is where inherited rules come from.

    >>> onto = ScimaOntology.load("v1.8")
    >>> "scima:RoadSegment" in superclasses_of(onto, "scima:ApproachLeg")
    True
    """
    ref = onto._to_ref(cls)
    seen, frontier = set(), [ref]
    while frontier:
        child = frontier.pop()
        for parent in onto.graph.objects(child, RDFS.subClassOf):
            if isinstance(parent, URIRef) and parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    return sorted(onto._qnames(seen))


def rules_inherited_by_legs(version: str = "v1.8") -> list[str]:
    """What an approach leg gets for nothing, by being placed under a segment.

    Placing the new class under scima:RoadSegment is the same move the
    spring release made with drone corridors. Every closure, capacity, and
    route rule written for segments covers legs with no new code.

    >>> rules_inherited_by_legs()[:2]
    ['scima:InfrastructureEntity', 'scima:RoadSegment']
    """
    return superclasses_of(ScimaOntology.load(version), "scima:ApproachLeg")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _scene():
    print(f"clock now = {NOW:.0f} s\n")
    for key in ("north_congestion", "south_congestion", "north_lanes", "south_lanes"):
        c = SCENE[key]
        owner = GRAPH_OWNER[c.graph] or "(shared)"
        print(f"  {key:<18} {c.subject:<22} {c.predicate:<28} {c.value:<8} "
              f"b={c.belief:.2f}  age={c.age_at(NOW):5.0f}s  {owner}")
    print("\nradio call contradicts both lane copies:",
          " ".join(RADIO_CALL))


def _age():
    a, b = SCENE["north_congestion"], SCENE["south_congestion"]
    live_a, live_b = compare_at(a, b, NOW)
    print(f"stored     loop {a.value} b={a.belief:.2f}   "
          f"camera {b.value} b={b.belief:.2f}   reads as a hard conflict")
    print(f"at one now loop {a.value} b={live_a:.4f}   "
          f"camera {b.value} b={live_b:.4f}   one real claim, one faded one")
    print(f"the camera frame has crossed "
          f"{b.age_at(NOW) / b.half_life_s:.0f} half-lives")
    n, s = aged_scene_distributions()
    print("\nas distributions over", CONGESTION_VALUES)
    print("  north aged", tuple(round(x, 4) for x in n))
    print("  south aged", tuple(round(x, 4) for x in s))


def _publish():
    floor = publication_floor()
    print(f"publication floor = C_wrong / (C_wrong + C_without) = {floor:.4f}")
    for label, belief in (("loop reading, fresh", 0.86),
                          ("camera frame, aged to now", 0.5181),
                          ("shared lane status", 0.90)):
        verdict = "publish" if should_publish(belief) else "keep it private"
        print(f"  {label:<28} b={belief:.4f}  {verdict}")
    print("\n  double the cost of a wrong shared fact and the floor moves:",
          f"{publication_floor(cost_of_a_wrong_shared_fact=2700.0):.4f}")


def _operator():
    hl = half_life_for("scima:hasLaneStatus")
    flip = operator_flip_age(hl)
    print(f"lane status half-life {hl:.0f} s, source error rate "
          f"{SHARED_VALUE_ERROR_RATE:.2f}")
    print(f"the fork turns over at T = h x ln(1/(1-e)) / ln2 = {flip:.1f} s "
          f"({flip / 60:.2f} min)\n")
    for key in ("north_lanes", "south_lanes"):
        c = SCENE[key]
        age = c.age_at(NOW)
        op = route_operator(age, hl)
        owner = GRAPH_OWNER[c.graph]
        reopened = reopened_by(LANE_FACT, op, INTENTIONS, owner)
        print(f"  {owner:<26} synced {age:5.0f}s ago   "
              f"P(update)={p_update(age, hl):.4f}   {op}")
        print(f"    re-opens: {reopened or 'nothing'}")
    print("\n  one message in, two operators out, two different amounts of work")


def _divergence():
    n, s = aged_scene_distributions()
    print("aged distributions over", CONGESTION_VALUES)
    print("  north", tuple(round(x, 4) for x in n))
    print("  south", tuple(round(x, 4) for x in s))
    print(f"\n  KL(north || south) = {kl_divergence(n, s):.4f} nats")
    print(f"  KL(south || north) = {kl_divergence(s, n):.4f} nats   "
          f"asymmetric, and a coordinator wants both")
    apart = chance_they_act_apart(n, s)
    print(f"\n  chance they act apart = 1 - overlap = {apart:.4f}")
    print(f"  expected error {apart:.4f} x {COST_OF_WRONG_ACTION_MIN} min = "
          f"{apart * COST_OF_WRONG_ACTION_MIN:.3f} min   "
          f"against a sync at {COST_OF_SYNC_MIN} min")
    print(f"  worth a message: {worth_a_message(n, s)}")
    steady = [0.41, 0.40, 0.42, 0.41]
    widening = [0.11, 0.24, 0.38, 0.53]
    print(f"\n  trend {steady} -> {divergence_trend(steady)}"
          f"      (information asymmetry)")
    print(f"  trend {widening} -> {divergence_trend(widening)}"
          f"    (something has broken)")


def _ladder():
    out = resolution_ladder(SCENE["north_congestion"], SCENE["south_congestion"])
    for i, step in enumerate(out["steps"], start=1):
        mark = "  <-- leaves here" if step["closed_it"] else ""
        detail = {k: v for k, v in step.items()
                  if k not in ("question", "closed_it")}
        print(f"  {i}. {step['question']:<34} {detail}{mark}")
    print(f"\n  verdict: {out['verdict']}")
    print(f"  legs of scima:MainAndNinth: {', '.join(MAIN_AND_NINTH_LEGS)}")
    print(f"  rollup over the legs: {rollup()}")
    print(f"  an approach leg inherits from: {', '.join(rules_inherited_by_legs())}")


def _diff():
    old_onto = ScimaOntology.load("v1.5")
    new_onto = ScimaOntology.load("v1.8")
    old, new = old_onto.summary(), new_onto.summary()
    print(f"  {old}")
    print(f"  {new}")
    print(f"  delta: +{new.n_classes - old.n_classes} classes, "
          f"+{new.n_properties - old.n_properties} properties, "
          f"+{new.n_axioms - old.n_axioms} axioms")
    added = sorted(set(new_onto.classes()) - set(old_onto.classes()))
    print("  new classes:", ", ".join(added))
    print("  every one of the five new axioms is safe, so the gate finds "
          "nothing to repair")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", action="store_true",
                        help="the two agents, their graphs, and what each holds")
    parser.add_argument("--age", action="store_true",
                        help="compare at one instant instead of two")
    parser.add_argument("--publish", action="store_true",
                        help="the derived floor on the private/shared gate")
    parser.add_argument("--operator", action="store_true",
                        help="one message routed to update on one side, revision on the other")
    parser.add_argument("--divergence", action="store_true",
                        help="the gap measured, and priced against a message")
    parser.add_argument("--ladder", action="store_true",
                        help="four questions in order, and where this pair leaves")
    parser.add_argument("--diff", action="store_true",
                        help="the v1.5 to v1.8 schema diff")
    args = parser.parse_args(argv)

    ran = False
    for flag, fn in (("scene", _scene), ("age", _age), ("publish", _publish),
                     ("operator", _operator), ("divergence", _divergence),
                     ("ladder", _ladder), ("diff", _diff)):
        if getattr(args, flag):
            fn()
            ran = True
    if not ran:
        _ladder()


if __name__ == "__main__":
    main()
