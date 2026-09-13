"""Goal achievement and planning over a stochastic KG (Article 11).

Article 10 asked where to find something in the graph, cheaply. This
module asks what to do once you have found it, and the difference
between the two turns out to be time. A search returns an answer about
the present. A plan is a commitment about the future, and the facts a
plan depends on go on moving while it executes.

The pieces the article works through:

  * ``ground_actions`` -- bind action-schema parameters by OWL class
    rather than by a flat type list, so subsumption decides who
    qualifies. The cascade is 3,969,000,000 untyped groundings ->
    4,800 typed -> 7 once the precondition runs as a query (Section 2).
  * ``candidate_funnel`` -- the three filters that are constantly
    conflated: type-correct (the class hierarchy), applicable (the
    precondition), and useful (an axiom on the goal). 7 -> 5 -> 4, and
    only the first of them is about the taxonomy (Section 3).
  * ``decompose`` -- HTN decomposition where methods attach to a goal
    CLASS and are inherited by its subclasses, with the three-way
    termination rule that distinguishes a leaf from a stuck goal
    (Section 4).
  * ``belief_weight`` / ``expected_duration`` / ``plan_shelf_life`` --
    preconditions priced as beliefs at the times their steps will
    actually run, which reverses the ranking of the three candidate
    plans completely (Section 5).
  * ``resolve_or_carry`` -- what to do with a precondition sitting in
    quarantine, which is neither true nor false (Section 7).
  * ``classify_contradiction`` -- update or revision, decided by how
    long the belief had to move (Section 9).

The scene is incident I-204, carried forward from Article 10 with the
same vehicles, stations, and ETAs. Two deadlines drive everything:
containment 20.0 minutes from tasking, and patient transport 16.0
minutes, derived as the county's 20-minute call-to-door cardiac
protocol minus the 4.0 minutes of call handling already spent.

Usage:
    python -m scima.agent_planning --ground
    python -m scima.agent_planning --decompose
    python -m scima.agent_planning --plans
    python -m scima.agent_planning --shelf
    python -m scima.agent_planning --quarantine
    python -m scima.agent_planning --surge
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import RDF, RDFS, OWL, Graph, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")

_ROOT = Path(__file__).resolve().parent.parent
_V1_1 = _ROOT / "ontologies" / "scima_owl_v1_1.ttl"


# =========================================================================
# Section 1: the two deadlines, derived rather than picked
# =========================================================================
CARDIAC_PROTOCOL_MIN = 20.0      # county target, measured from the call
CALL_HANDLING_MIN = 4.0          # triage before the dispatch agent was tasked
CONTAINMENT_DEADLINE_MIN = 20.0  # measured from tasking, not from the call


def transport_deadline_min(protocol_min: float = CARDIAC_PROTOCOL_MIN,
                           already_spent_min: float = CALL_HANDLING_MIN) -> float:
    """Budget left for the transport goal, measured from tasking.

    The deadline is a consequence of a protocol and a clock that has
    already been running, not a number somebody liked.

    >>> transport_deadline_min()
    16.0
    """
    return protocol_min - already_spent_min


# =========================================================================
# Section 5: change rates belong to a (class, predicate) pair
# =========================================================================
# Measured over ordinary operations. A pump truck's availability does not
# flip on an ambulance's clock even though both facts use hasStatus, so a
# rate keyed on the predicate alone would be wrong for one of them.
CHANGE_RATE_PROFILES: dict[tuple[str, str], float] = {
    ("scima:Ambulance", "scima:hasStatus"): 90.0,
    ("scima:FireTruck", "scima:hasStatus"): 210.0,
    ("scima:MaintenanceCrew", "scima:hasStatus"): 1_500.0,
    ("scima:Vehicle", "scima:hasLocation"): 300.0,
    ("scima:RoadSegment", "scima:hasCongestionLevel"): 900.0,
    ("scima:LandingZone", "scima:hasLandingClearance"): 1_800.0,
    ("scima:Hospital", "scima:hasDivertStatus"): 3_600.0,
    ("scima:Vehicle", "scima:stationedAt"): 90 * 86_400.0,
}

# A multi-casualty surge assigns, reassigns, and releases vehicles far
# faster than ordinary operations. The logged half-life is an estimate of
# a quantity that itself moves, so every consumer takes a multiplier and
# holding the rate fixed is never a hidden assumption.
SURGE_RATE_MULTIPLIER = 3.0

# A regime maps a predicate to the multiplier in force for it. This is a
# mapping rather than one number on purpose, since a surge changes how
# fast vehicles flip and does nothing to how fast obstructions appear on
# a helipad. Applying one multiplier to every precondition would be the
# same mistake as keying a change rate on the predicate alone.
SURGE_REGIME: dict[str, float] = {"scima:hasStatus": SURGE_RATE_MULTIPLIER}


def _multiplier(regime: dict[str, float] | None, predicate: str) -> float:
    """The rate multiplier in force for one predicate.

    >>> _multiplier(SURGE_REGIME, "scima:hasStatus")
    3.0
    >>> _multiplier(SURGE_REGIME, "scima:hasLandingClearance")
    1.0
    >>> _multiplier(None, "scima:hasStatus")
    1.0
    """
    return (regime or {}).get(predicate, 1.0)

# The planner refuses to commit a plan whose joint confidence is below
# this. Under surge some plans fail it at the moment they are written.
COMMIT_FLOOR = 0.55


def half_life_for(onto: ScimaOntology, cls: str, predicate: str) -> float:
    """Half-life in seconds for ``predicate`` on ``cls``, inherited.

    A class with no profile of its own walks up the hierarchy until it
    finds one, which is the same subsumption that decided AMB-17 was an
    EmergencyVehicle, applied to rates. AirAmbulance declares nothing and
    therefore uses Ambulance's 90 seconds.

    >>> onto = ScimaOntology.load("v1.1")
    >>> half_life_for(onto, "scima:AirAmbulance", "scima:hasStatus")
    90.0
    >>> half_life_for(onto, "scima:LadderTruck", "scima:hasStatus")
    210.0
    """
    key = (cls, predicate)
    if key in CHANGE_RATE_PROFILES:
        return CHANGE_RATE_PROFILES[key]
    ref = SCIMA[cls.split(":", 1)[1]]
    for anc in onto.graph.transitive_objects(ref, RDFS.subClassOf):
        if not str(anc).startswith(str(SCIMA)):
            continue
        qn = "scima:" + str(anc)[len(str(SCIMA)):]
        if (qn, predicate) in CHANGE_RATE_PROFILES:
            return CHANGE_RATE_PROFILES[(qn, predicate)]
    raise KeyError(f"no change-rate profile for {cls} {predicate}")


def decay_rate(half_life_s: float, rate_multiplier: float = 1.0) -> float:
    """lambda = ln 2 / h, scaled by the regime actually in force.

    >>> round(decay_rate(90.0), 6)
    0.007702
    >>> round(decay_rate(90.0, SURGE_RATE_MULTIPLIER), 6)
    0.023105
    """
    return rate_multiplier * math.log(2) / half_life_s


def belief_weight(half_life_s: float, delta_s: float,
                  rate_multiplier: float = 1.0) -> float:
    """How much of its original credibility a fact still carries after
    ``delta_s`` with no new evidence, w(d) = exp(-lambda * d).

    This is the closed form of the transition law applied for d steps
    with nothing observed, not an independent modelling choice. For a
    two-state chain with per-step probability q of leaving the current
    state, (1 - q)^d = exp(d * ln(1 - q)), so lambda = -ln(1 - q).

    >>> round(belief_weight(90.0, 25), 4)
    0.8249
    >>> round(belief_weight(90.0, 10, SURGE_RATE_MULTIPLIER), 4)
    0.7937
    """
    return math.exp(-decay_rate(half_life_s, rate_multiplier) * delta_s)


def flip_probability_per_minute(half_life_s: float) -> float:
    """The per-minute q that the decay rate is the closed form of.

    Included because it is the quantity an operator can actually observe
    on a dispatch board, and it is what makes lambda checkable.

    >>> round(flip_probability_per_minute(90.0), 4)
    0.37
    """
    return 1 - belief_weight(half_life_s, 60.0)


# =========================================================================
# Section 5: preconditions checked at the time their step will run
# =========================================================================
@dataclass
class Precondition:
    """One fact a step depends on, with everything needed to price it.

    ``until_exec_s`` is the part a classical planner drops. The check is
    a claim about the world at the time the step runs, so the evidence
    has to survive the wait as well as its own age.

    ``predicate`` is carried so that a regime shift can be applied to
    the rates it actually moves rather than to all of them.
    """
    label: str
    predicate: str
    half_life_s: float
    evidence_age_s: float
    until_exec_s: float
    recovery_min: float

    def rate(self, regime: dict[str, float] | None = None) -> float:
        return decay_rate(self.half_life_s, _multiplier(regime, self.predicate))

    def confidence(self, regime: dict[str, float] | None = None) -> float:
        return belief_weight(self.half_life_s,
                             self.evidence_age_s + self.until_exec_s,
                             _multiplier(regime, self.predicate))

    def penalty_min(self, regime: dict[str, float] | None = None) -> float:
        return (1 - self.confidence(regime)) * self.recovery_min


@dataclass
class CandidatePlan:
    name: str
    vehicle: str
    nominal_min: float
    preconditions: list[Precondition] = field(default_factory=list)

    def success_probability(self, regime: dict[str, float] | None = None) -> float:
        """Product over preconditions, assuming independent failures.

        The assumption is worth stating, since correlated staleness such
        as one silent radio or a citywide power event would make this
        optimistic.
        """
        p = 1.0
        for pre in self.preconditions:
            p *= pre.confidence(regime)
        return p

    def expected_min(self, regime: dict[str, float] | None = None) -> float:
        """T_nominal + sum over preconditions of (1 - conf) * recovery."""
        return self.nominal_min + sum(
            pre.penalty_min(regime) for pre in self.preconditions)

    def shelf_life_s(self, floor: float = COMMIT_FLOOR,
                     regime: dict[str, float] | None = None) -> float:
        """Seconds this finished plan may sit before its own confidence
        test fails.

        Every second of delay adds equally to every precondition's age,
        so the rates simply add and the answer is
        ln(P(0) / floor) / sum(lambda_i). A negative result means the
        plan fails its own test at the moment it is written, which is
        what a surge does to a plan built on stored availability.
        """
        rates = [pre.rate(regime) for pre in self.preconditions]
        return math.log(self.success_probability(regime) / floor) / sum(rates)

    def committable(self, floor: float = COMMIT_FLOOR,
                    regime: dict[str, float] | None = None) -> bool:
        return self.success_probability(regime) >= floor

    def dominant_rate_share(self, regime: dict[str, float] | None = None
                            ) -> tuple[str, float]:
        """Which precondition supplies most of the decay, and how much.

        A plan's shelf life is governed by its fastest-moving
        precondition rather than by the average, so this says what to
        confirm.
        """
        rates = {pre.label: pre.rate(regime) for pre in self.preconditions}
        total = sum(rates.values())
        label = max(rates, key=rates.get)
        return label, rates[label] / total

    def confirm(self, label: str) -> "CandidatePlan":
        """The same plan after one precondition is confirmed by radio.

        Confirming drops that precondition's term from both the joint
        confidence and the decay sum, which is why one call is worth so
        much more than its 20 seconds.
        """
        return CandidatePlan(
            name=f"{self.name} (confirmed {label})",
            vehicle=self.vehicle,
            nominal_min=self.nominal_min,
            preconditions=[p for p in self.preconditions if p.label != label],
        )


def expected_duration(nominal_min: float,
                      checks: list[tuple[float, float, float, float]]) -> float:
    """Plain-tuple form of :meth:`CandidatePlan.expected_min`.

    ``checks`` holds (half_life_s, evidence_age_s, until_exec_s,
    recovery_min) per precondition.

    >>> round(expected_duration(12.6, [(90, 25, 0, 3.5),
    ...                                (900, 360, 432, 5.5),
    ...                                (3600, 180, 756, 6.0)]), 2)
    16.71
    """
    total = nominal_min
    for half_life, age, until_exec, recovery in checks:
        total += (1 - belief_weight(half_life, age + until_exec)) * recovery
    return total


def plan_shelf_life(joint_conf: float, rates: list[float],
                    floor: float = COMMIT_FLOOR) -> float:
    """Plain-number form of :meth:`CandidatePlan.shelf_life_s`.

    >>> round(plan_shelf_life(0.632391, [0.007702, 0.000385, 0.000193]), 1)
    16.9
    >>> round(plan_shelf_life(0.683020, [0.000385, 0.000193]), 0)
    375.0
    """
    return math.log(joint_conf / floor) / sum(rates)


# =========================================================================
# The three candidate plans for the transport goal at I-204
# =========================================================================
def candidate_plans() -> list[CandidatePlan]:
    """Plans A, B, and C, exactly as the article prices them.

    A is the ground ambulance with its preconditions checked once at
    plan time. B is the air ambulance. C is A with verification steps
    inserted, two of which run concurrently with patient loading and so
    cost nothing on the critical path.
    """
    plan_a = CandidatePlan(
        name="A: AMB-17, checked once at plan time",
        vehicle="scima:AMB_17",
        nominal_min=4.2 + 3.0 + 5.4,
        preconditions=[
            Precondition("AMB-17 available", "scima:hasStatus",
                         90.0, 25.0, 0.0, 3.5),
            Precondition("bridge passable", "scima:hasCongestionLevel",
                         900.0, 360.0, 432.0, 5.5),
            Precondition("Mercy General accepting", "scima:hasDivertStatus",
                         3600.0, 180.0, 756.0, 6.0),
        ],
    )
    plan_b = CandidatePlan(
        name="B: AIR-3, checked once at plan time",
        vehicle="scima:AIR_3",
        nominal_min=6.5 + 3.0 + 3.5,
        preconditions=[
            Precondition("AIR-3 available", "scima:hasStatus",
                         90.0, 10.0, 0.0, 3.5),
            Precondition("landing zone clear", "scima:hasLandingClearance",
                         1800.0, 120.0, 390.0, 4.0),
            Precondition("Mercy General accepting", "scima:hasDivertStatus",
                         3600.0, 180.0, 780.0, 6.0),
        ],
    )
    # C pays 0.33 min for a radio call and 0.20 min for a camera read on
    # the critical path. Both reset their evidence to zero age AND are
    # taken late enough that no wait follows, so both weigh 1.0 and drop
    # out. Only the hospital status carries residual risk, refreshed at
    # 7.0 min and needed at 13.13, so its delta is 368 seconds.
    plan_c = CandidatePlan(
        name="C: AMB-17 with four verification steps",
        vehicle="scima:AMB_17",
        nominal_min=4.2 + 3.0 + 5.4 + 0.33 + 0.20,
        preconditions=[
            Precondition("Mercy General accepting", "scima:hasDivertStatus",
                         3600.0, 0.0, 368.0, 6.0),
        ],
    )
    return [plan_a, plan_b, plan_c]


def rank_plans(plans: list[CandidatePlan] | None = None,
               regime: dict[str, float] | None = None,
               by: str = "expected") -> list[CandidatePlan]:
    """Plans sorted by expected duration or by nominal duration.

    Sorting on one column rather than the other changes which vehicle
    leaves the station, which is the article's whole point.

    >>> [p.name[0] for p in rank_plans(by="nominal")]
    ['A', 'B', 'C']
    >>> [p.name[0] for p in rank_plans(by="expected")]
    ['C', 'B', 'A']
    """
    plans = candidate_plans() if plans is None else plans
    key = ((lambda p: p.expected_min(regime)) if by == "expected"
           else (lambda p: p.nominal_min))
    return sorted(plans, key=key)


def verification_value(half_life_s: float, evidence_age_s: float,
                       until_exec_s: float, recovery_min: float) -> tuple[float, float]:
    """Minutes saved by refreshing a precondition early against late.

    Refreshing at plan time removes the evidence age but not the wait.
    Refreshing when the step begins removes both, which is why the same
    spend is worth more later.

    >>> early, late = verification_value(900, 360, 432, 5.5)
    >>> round(early, 3), round(late, 3)
    (0.955, 2.511)
    >>> round(late / early, 2)
    2.63
    """
    base = (1 - belief_weight(half_life_s, evidence_age_s + until_exec_s)) * recovery_min
    refreshed_early = (1 - belief_weight(half_life_s, until_exec_s)) * recovery_min
    return base - refreshed_early, base


# =========================================================================
# Section 2 and 3: grounding, and the three filters people conflate
# =========================================================================
CITY_NODES = 63_000
FLEET_SIZE = 800
OPEN_INCIDENTS = 6


@dataclass
class GroundingCascade:
    untyped: int
    owl_typed: int
    precondition_first: int

    @property
    def typing_factor(self) -> float:
        return self.untyped / self.owl_typed

    @property
    def precondition_factor(self) -> float:
        return self.owl_typed / self.precondition_first

    @property
    def total_factor(self) -> float:
        return self.untyped / self.precondition_first


def grounding_cascade(reachable_vehicles: int = 7,
                      reachable_incidents: int = 1) -> GroundingCascade:
    """How many ground actions a two-parameter schema yields, three ways.

    Typing the parameters with OWL classes replaces the cross product
    over every node with the cross product over the fleet and the open
    incidents. Running the precondition as a query first, rather than
    generating and filtering, cuts it again.

    >>> c = grounding_cascade()
    >>> c.untyped, c.owl_typed, c.precondition_first
    (3969000000, 4800, 7)
    >>> round(c.typing_factor), round(c.precondition_factor)
    (826875, 686)
    """
    return GroundingCascade(
        untyped=CITY_NODES * CITY_NODES,
        owl_typed=FLEET_SIZE * OPEN_INCIDENTS,
        precondition_first=reachable_vehicles * reachable_incidents,
    )


# The vehicles a relation-guided walk reaches from I-204, with the status
# the graph holds for each. Carried forward from Article 10's scene, plus
# the air ambulance the county brought online last month.
SCENE_VEHICLES: list[tuple[str, str, str]] = [
    ("scima:AMB_17", "scima:Ambulance", "available"),
    ("scima:AMB_22", "scima:Ambulance", "out_of_service"),
    ("scima:FIRE_12", "scima:FireTruck", "available"),
    ("scima:AMB_04", "scima:Ambulance", "en_route"),
    ("scima:AMB_09", "scima:Ambulance", "available"),
    ("scima:AMB_31", "scima:Ambulance", "available"),
    ("scima:AIR_3", "scima:AirAmbulance", "available"),
]

# What the dispatch agent's hard-coded list said when it was written,
# which was before AirAmbulance was declared. Nobody updated it.
STALE_TYPE_LIST = [
    "scima:Ambulance", "scima:FireTruck", "scima:PoliceVehicle",
    "scima:HazmatUnit", "scima:RescueBoat", "scima:LadderTruck",
]


@dataclass
class Funnel:
    type_correct: list[str]
    applicable: list[str]
    useful: list[str]
    stale_list_result: list[str]

    @property
    def missed_by_stale_list(self) -> list[str]:
        return [v for v in self.useful if v not in self.stale_list_result]


def candidate_funnel(onto: ScimaOntology,
                     param_class: str = "scima:EmergencyVehicle",
                     goal_resource_class: str = "scima:Ambulance") -> Funnel:
    """The three filters, kept separate on purpose.

    Type-correct is the class hierarchy and nothing else. Applicable is
    the precondition, which is a belief about the world. Useful is an
    axiom on the goal. Conflating them is how people end up believing a
    taxonomy will make their decisions for them.

    >>> onto = ScimaOntology.load("v1.1")
    >>> f = candidate_funnel(onto)
    >>> len(f.type_correct), len(f.applicable), len(f.useful)
    (7, 5, 4)
    >>> f.missed_by_stale_list
    ['scima:AIR_3']
    """
    def subsumed_by(cls: str, parent: str) -> bool:
        ref = SCIMA[cls.split(":", 1)[1]]
        target = SCIMA[parent.split(":", 1)[1]]
        return target in set(onto.graph.transitive_objects(ref, RDFS.subClassOf))

    type_correct = [v for v, cls, _ in SCENE_VEHICLES
                    if subsumed_by(cls, param_class)]
    applicable = [v for v, cls, status in SCENE_VEHICLES
                  if subsumed_by(cls, param_class) and status == "available"]
    useful = [v for v, cls, status in SCENE_VEHICLES
              if subsumed_by(cls, goal_resource_class) and status == "available"]
    # The stale list matches on exact type names, so a subclass declared
    # after it was written simply does not appear. No error is raised,
    # because a query that finds fewer things is still a valid query.
    stale = [v for v, cls, status in SCENE_VEHICLES
             if cls in STALE_TYPE_LIST and status == "available"
             and subsumed_by(cls, goal_resource_class)]
    return Funnel(type_correct, applicable, useful, stale)


# =========================================================================
# Section 4: decomposition, with a three-way termination rule
# =========================================================================
@dataclass
class Method:
    name: str
    method_for: str            # a goal CLASS, so subclasses inherit it
    steps: tuple[str, ...]


METHODS = [
    Method("M1", "scima:IncidentResolutionGoal",
           ("TransportPatient", "RestoreWater", "ClearTraffic", "NotifyPublic")),
    Method("M2", "scima:PatientTransportGoal", ("Dispatch", "Load", "Transport")),
    Method("M3", "scima:ServiceRestorationGoal", ("IsolateValve", "RepairMain")),
    Method("M4", "scima:TrafficClearanceGoal", ("DivertTraffic", "PumpOut")),
]

# Goal classes that one action schema achieves directly.
LEAF_GOAL_CLASSES = {"scima:PublicNotificationGoal"}


def methods_for(onto: ScimaOntology, goal_class: str) -> list[Method]:
    """Methods whose head subsumes ``goal_class``.

    A method attaches to a class, so one written at
    ServiceRestorationGoal covers both the water and power children, and
    will cover a gas child the day somebody declares one.

    >>> onto = ScimaOntology.load("v1.1")
    >>> [m.name for m in methods_for(onto, "scima:WaterServiceRestorationGoal")]
    ['M3']
    >>> [m.name for m in methods_for(onto, "scima:PowerServiceRestorationGoal")]
    ['M3']
    """
    ref = SCIMA[goal_class.split(":", 1)[1]]
    ancestors = {str(a) for a in onto.graph.transitive_objects(ref, RDFS.subClassOf)}
    ancestors.add(str(ref))
    return [m for m in METHODS
            if str(SCIMA[m.method_for.split(":", 1)[1]]) in ancestors]


def classify_goal(onto: ScimaOntology, goal_class: str) -> str:
    """One of 'decomposable', 'leaf', or 'stuck'.

    The third case is the one people leave out, and leaving it out has a
    specific consequence. A planner with a two-way rule finds no method,
    finds no action, returns an empty step list, and reports the PARENT
    plan complete. The subgoal has been silently dropped.

    >>> onto = ScimaOntology.load("v1.1")
    >>> classify_goal(onto, "scima:IncidentResolutionGoal")
    'decomposable'
    >>> classify_goal(onto, "scima:PublicNotificationGoal")
    'leaf'
    >>> classify_goal(onto, "scima:PowerServiceRestorationGoal")
    'decomposable'
    """
    if methods_for(onto, goal_class):
        return "decomposable"
    if goal_class in LEAF_GOAL_CLASSES:
        return "leaf"
    return "stuck"


# The seven steps that survive, with durations and precedence. PumpOut
# waits on both IsolateValve (pumping a live break moves water around)
# and DivertTraffic (the pump truck needs the lane).
PLAN_STEPS: dict[str, tuple[float, tuple[str, ...]]] = {
    "Broadcast": (1.5, ()),
    "DivertTraffic": (4.5, ()),
    "IsolateValve": (6.0, ()),
    "Dispatch": (4.2, ()),
    "Load": (3.0, ("Dispatch",)),
    "Transport": (5.4, ("Load",)),
    "PumpOut": (12.0, ("IsolateValve", "DivertTraffic")),
}


def schedule(steps: dict[str, tuple[float, tuple[str, ...]]] | None = None
             ) -> dict[str, tuple[float, float]]:
    """Earliest start and finish per step, honouring the partial order.

    >>> s = schedule()
    >>> s["PumpOut"]
    (6.0, 18.0)
    >>> round(s["Transport"][1], 2)
    12.6
    """
    steps = PLAN_STEPS if steps is None else steps
    out: dict[str, tuple[float, float]] = {}

    def finish(name: str) -> float:
        if name in out:
            return out[name][1]
        duration, deps = steps[name]
        start = max((finish(d) for d in deps), default=0.0)
        out[name] = (start, start + duration)
        return out[name][1]

    for name in steps:
        finish(name)
    return out


def critical_path_min(steps: dict[str, tuple[float, tuple[str, ...]]] | None = None
                      ) -> float:
    """The longest branch, which is not always the most urgent one.

    >>> critical_path_min()
    18.0
    """
    return max(f for _s, f in schedule(steps).values())


# =========================================================================
# Section 6: mutual exclusion derived from axioms, not tabulated
# =========================================================================
# Each proposed ground action and the effect it would assert. Two of these
# conflict, and neither method knew about the other.
PROPOSED_EFFECTS: dict[str, tuple[str, str, str]] = {
    "Dispatch(AMB-17)": ("scima:Goal_Transport_P88", "scima:assignedResource", "scima:AMB_17"),
    "Dispatch(AIR-3)": ("scima:Goal_Transport_P88", "scima:assignedResource", "scima:AIR_3"),
    "IsolateValve(V-12)": ("scima:Valve_V12", "scima:hasValveState", "closed"),
    "OpenHydrant(V-12)": ("scima:Valve_V12", "scima:hasValveState", "open"),
    "DivertTraffic": ("scima:RoadSegment_Main_St_NB", "scima:hasDetour", "true"),
    "PumpOut": ("scima:RoadSegment_Main_St_NB", "scima:hasStandingWater", "false"),
    "Broadcast": ("scima:ControlZone_Z7", "scima:hasAdvisory", "issued"),
    "Load": ("scima:Patient_P88", "scima:isLoaded", "true"),
}


def derive_mutex(onto: ScimaOntology,
                 effects: dict[str, tuple[str, str, str]] | None = None
                 ) -> list[tuple[str, str, str]]:
    """Mutually exclusive action pairs, read off the axioms.

    Returns (action_a, action_b, reason). Two rules fire here. A
    functional datatype property cannot hold two values at once, and a
    qualified cardinality restriction of 1 on a defined class limits how
    many resources one goal may take.

    >>> onto = ScimaOntology.load("v1.1")
    >>> pairs = derive_mutex(onto)
    >>> len(pairs)
    2
    >>> sorted(r for _a, _b, r in pairs)
    ['functional property scima:hasValveState', 'maxQualifiedCardinality 1 on scima:assignedResource']
    """
    effects = PROPOSED_EFFECTS if effects is None else effects
    functional = {
        "scima:" + str(p)[len(str(SCIMA)):]
        for p in onto.graph.subjects(RDF.type, OWL.FunctionalProperty)
        if str(p).startswith(str(SCIMA))
    }
    capped = _max_one_properties(onto)

    names = list(effects)
    out: list[tuple[str, str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, pa, oa = effects[a]
            sb, pb, ob = effects[b]
            if sa != sb or pa != pb or oa == ob:
                continue
            if pa in functional:
                out.append((a, b, f"functional property {pa}"))
            elif pa in capped:
                out.append((a, b, f"maxQualifiedCardinality 1 on {pa}"))
    return out


def _max_one_properties(onto: ScimaOntology) -> set[str]:
    """Properties carrying a maxQualifiedCardinality of 1 somewhere."""
    out: set[str] = set()
    for restriction in onto.graph.subjects(RDF.type, OWL.Restriction):
        card = onto.graph.value(restriction, OWL.maxQualifiedCardinality)
        if card is None or int(card) != 1:
            continue
        prop = onto.graph.value(restriction, OWL.onProperty)
        if prop is not None and str(prop).startswith(str(SCIMA)):
            out.add("scima:" + str(prop)[len(str(SCIMA)):])
    return out


def violates_resource_type(onto: ScimaOntology, vehicle_class: str,
                           required_class: str = "scima:Ambulance") -> bool:
    """True when assigning this vehicle would make the graph inconsistent.

    The universal restriction lets the reasoner derive that the assigned
    resource is an Ambulance. If the graph also says it is a FireTruck,
    and the two are disjoint, the two conclusions contradict.

    >>> onto = ScimaOntology.load("v1.1")
    >>> violates_resource_type(onto, "scima:FireTruck")
    True
    >>> violates_resource_type(onto, "scima:AirAmbulance")
    False
    """
    ref = SCIMA[vehicle_class.split(":", 1)[1]]
    target = SCIMA[required_class.split(":", 1)[1]]
    if target in set(onto.graph.transitive_objects(ref, RDFS.subClassOf)):
        return False
    for a, b in onto.graph.subject_objects(OWL.disjointWith):
        pair = {str(a), str(b)}
        if str(ref) in pair and str(target) in pair:
            return True
    return False


# =========================================================================
# Section 7: the precondition that is neither true nor false
# =========================================================================
QUARANTINE_RESOLUTION_MIN = 0.4   # a call to the county dispatcher
QUARANTINE_RECOVERY_MIN = 6.0     # falling back to a ground ambulance mid-flight


@dataclass
class QuarantinedFact:
    """A candidate triple that failed the admission gate.

    Absence from the working graph and absence from the world are
    indistinguishable to a boolean precondition, which is what makes
    treating one as the other so damaging.
    """
    subject: str
    predicate: str
    obj: str
    extraction_confidence: float
    observed_age_s: float
    half_life_s: float

    def belief(self) -> float:
        """Extraction confidence discounted for age.

        >>> q = QuarantinedFact("scima:AIR_3", "scima:stationedAt",
        ...                     "scima:Helipad_Hilltop", 0.88, 400.0, 90 * 86400.0)
        >>> round(q.belief(), 3)
        0.88
        """
        return self.extraction_confidence * belief_weight(self.half_life_s,
                                                          self.observed_age_s)


def breakeven_belief(resolution_min: float = QUARANTINE_RESOLUTION_MIN,
                     recovery_min: float = QUARANTINE_RECOVERY_MIN) -> float:
    """Belief at which resolving and carrying cost the same.

    Resolving is worth it while (1 - belief) * recovery > resolution,
    which rearranges to belief < 1 - resolution / recovery.

    >>> round(breakeven_belief(), 4)
    0.9333
    """
    return 1 - resolution_min / recovery_min


def resolve_or_carry(belief: float,
                     resolution_min: float = QUARANTINE_RESOLUTION_MIN,
                     recovery_min: float = QUARANTINE_RECOVERY_MIN) -> str:
    """'resolve' or 'carry', decided by the breakeven above.

    >>> resolve_or_carry(0.88)
    'resolve'
    >>> resolve_or_carry(0.95)
    'carry'
    """
    return "resolve" if belief < breakeven_belief(resolution_min, recovery_min) else "carry"


# =========================================================================
# Section 9: update or revision, decided by how long the belief had to move
# =========================================================================
def classify_contradiction(half_life_s: float, belief_age_s: float,
                           likelihood_if_changed: float = 0.95,
                           likelihood_if_unchanged: float = 0.10) -> float:
    """P(the world moved | a contradicting observation arrived).

    The prior comes from the transition law rather than from a setting,
    since the chance the world changed during the gap is exactly
    1 - w(gap). An update leaves completed steps justified. A revision
    means the earlier information was bad, so everything derived from it
    has to be re-examined, including commitments already made.

    >>> round(classify_contradiction(900, 792), 3)
    0.889
    >>> round(classify_contradiction(900, 20), 3)
    0.129
    """
    prior_changed = 1 - belief_weight(half_life_s, belief_age_s)
    joint_changed = prior_changed * likelihood_if_changed
    joint_unchanged = (1 - prior_changed) * likelihood_if_unchanged
    return joint_changed / (joint_changed + joint_unchanged)


def frame_scoped_recheck(total_steps: int = 7, steps_reading_frame: int = 2) -> float:
    """Fraction of re-verification avoided by recording which frame each
    step checked against.

    >>> round(100 * frame_scoped_recheck(), 0)
    71.0
    """
    return 1 - steps_reading_frame / total_steps


# =========================================================================
# CLI
# =========================================================================
def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Goal achievement and planning over a stochastic KG (Article 11).")
    parser.add_argument("--ground", action="store_true",
                        help="the grounding cascade and the three candidate filters")
    parser.add_argument("--decompose", action="store_true",
                        help="methods, the three-way termination rule, and the schedule")
    parser.add_argument("--plans", action="store_true",
                        help="price the three candidate plans two ways")
    parser.add_argument("--shelf", action="store_true",
                        help="how long a finished plan stays committable")
    parser.add_argument("--quarantine", action="store_true",
                        help="a precondition that is neither true nor false")
    parser.add_argument("--surge", action="store_true",
                        help="the same plan when the flip rate triples")
    args = parser.parse_args(argv)

    onto = ScimaOntology.load("v1.1")
    print(f"Loaded SCIMA-OWL v1.1: {onto.summary().n_classes} classes, "
          f"{onto.summary().n_properties} properties, {onto.summary().n_axioms} axioms")
    print(f"Transport deadline: {CARDIAC_PROTOCOL_MIN:.0f} min protocol "
          f"- {CALL_HANDLING_MIN:.0f} min already spent "
          f"= {transport_deadline_min():.1f} min from tasking")

    if args.ground:
        c = grounding_cascade()
        print("\nGrounding DispatchVehicle(?v, ?i), three ways:")
        print(f"  untyped, every node x every node : {c.untyped:>13,}")
        print(f"  parameters typed by OWL class    : {c.owl_typed:>13,}"
              f"   ({c.typing_factor:,.0f}x fewer)")
        print(f"  precondition run as a query first: {c.precondition_first:>13,}"
              f"   ({c.precondition_factor:,.0f}x fewer)")
        print(f"  total reduction                  : {c.total_factor:,.0f}x")

        f = candidate_funnel(onto)
        print("\nThe three filters, which answer three different questions:")
        print(f"  type-correct (class hierarchy) : {len(f.type_correct)}  "
              f"{', '.join(v.split(':')[1] for v in f.type_correct)}")
        print(f"  applicable   (precondition)    : {len(f.applicable)}  "
              f"{', '.join(v.split(':')[1] for v in f.applicable)}")
        print(f"  useful       (axiom on goal)   : {len(f.useful)}  "
              f"{', '.join(v.split(':')[1] for v in f.useful)}")
        print(f"  same query, hard-coded 6-type list: {len(f.stale_list_result)}  "
              f"{', '.join(v.split(':')[1] for v in f.stale_list_result)}")
        print(f"  silently dropped: "
              f"{', '.join(v.split(':')[1] for v in f.missed_by_stale_list)}"
              f"   <- the only plan that meets the deadline")

    if args.decompose:
        print("\nMethods attach to a goal class, so subclasses inherit them:")
        for goal in ("scima:IncidentResolutionGoal", "scima:PatientTransportGoal",
                     "scima:WaterServiceRestorationGoal",
                     "scima:PowerServiceRestorationGoal",
                     "scima:TrafficClearanceGoal", "scima:PublicNotificationGoal"):
            ms = methods_for(onto, goal)
            print(f"  {goal.split(':')[1]:34s} {classify_goal(onto, goal):13s} "
                  f"{', '.join(m.name for m in ms) if ms else '-'}")
        print(f"  {len(METHODS)} methods cover 6 goal classes, because M3 sits at the parent")

        sched = schedule()
        print("\nSchedule (partial order, earliest start):")
        for name in sorted(sched, key=lambda n: (sched[n][1], n)):
            start, fin = sched[name]
            deps = PLAN_STEPS[name][1]
            print(f"  {name:14s} start {start:5.1f}  finish {fin:5.1f}"
                  f"   after: {', '.join(deps) if deps else 'nothing'}")
        cp = critical_path_min()
        print(f"  critical path {cp:.1f} min, "
              f"slack {CONTAINMENT_DEADLINE_MIN - cp:.1f} against the {CONTAINMENT_DEADLINE_MIN:.1f} deadline")

        pairs = derive_mutex(onto)
        print(f"\nMutex over {len(PROPOSED_EFFECTS)} proposed actions "
              f"({len(PROPOSED_EFFECTS) * (len(PROPOSED_EFFECTS) - 1) // 2} candidate pairs), derived not tabulated:")
        for a, b, reason in pairs:
            print(f"  {a} x {b}\n      because {reason}")

    if args.plans:
        deadline = transport_deadline_min()
        print("\nThree plans for the transport goal, priced two ways:")
        print(f"  {'plan':46s} {'nominal':>8s} {'P(succ)':>8s} {'E[T]':>8s}  verdict")
        for p in candidate_plans():
            exp = p.expected_min()
            verdict = (f"misses by {exp - deadline:.2f}" if exp > deadline
                       else f"meets, slack {deadline - exp:.2f}")
            print(f"  {p.name:46s} {p.nominal_min:8.2f} "
                  f"{p.success_probability():8.3f} {exp:8.2f}  {verdict}")
        print(f"\n  nominal order : "
              f"{', '.join(p.name[0] for p in rank_plans(by='nominal'))}")
        print(f"  expected order: "
              f"{', '.join(p.name[0] for p in rank_plans(by='expected'))}")
        print("  The ranking reverses. The fastest plan on paper is the one that misses.")

        early, late = verification_value(900, 360, 432, 5.5)
        print(f"\n  Same 0.20 min camera read, refreshed at plan time : "
              f"{early:.3f} min saved")
        print(f"  Same 0.20 min camera read, refreshed at execution : "
              f"{late:.3f} min saved  ({late / early:.2f}x better)")

    if args.shelf:
        plan_b = candidate_plans()[1]
        label, share = plan_b.dominant_rate_share()
        print(f"\nPlan B joint confidence {plan_b.success_probability():.3f}, "
              f"commit floor {COMMIT_FLOOR}")
        print(f"  shelf life as written        : {plan_b.shelf_life_s():.1f} s")
        print(f"  dominant term                : {label} at "
              f"{100 * share:.1f}% of the total decay rate")
        confirmed = plan_b.confirm(label)
        print(f"  after one radio call         : {confirmed.shelf_life_s():.0f} s "
              f"({confirmed.shelf_life_s() / 60:.2f} min), "
              f"{confirmed.shelf_life_s() / plan_b.shelf_life_s():.1f}x longer")
        print("  A plan's shelf life is set by its fastest-moving precondition,")
        print("  which is also what tells you which one to confirm.")

    if args.quarantine:
        q = QuarantinedFact("scima:AIR_3", "scima:stationedAt",
                            "scima:Helipad_Hilltop", 0.88, 400.0, 90 * 86_400.0)
        print(f"\nAIR-3's stationedAt failed the admission gate "
              f"(two values, and the property is functional).")
        print(f"  extraction confidence {q.extraction_confidence:.2f}, "
              f"observed {q.observed_age_s:.0f} s ago, half-life 90 days")
        print(f"  precondition belief = {q.belief():.3f}  (UNKNOWN, not FALSE)")
        print(f"  breakeven belief    = 1 - {QUARANTINE_RESOLUTION_MIN} / "
              f"{QUARANTINE_RECOVERY_MIN} = {breakeven_belief():.4f}")
        print(f"  decision            = {resolve_or_carry(q.belief()).upper()}")
        plan_b = candidate_plans()[1]
        carry = plan_b.expected_min() + (1 - q.belief()) * QUARANTINE_RECOVERY_MIN
        resolve = plan_b.expected_min() + QUARANTINE_RESOLUTION_MIN
        print(f"  carry the risk : {carry:.2f} min")
        print(f"  resolve first  : {resolve:.2f} min   "
              f"(wins by {carry - resolve:.2f}, and removes the variance)")
        print("\n  A planner treating absence as falsehood never gets this choice.")
        print("  It drops AIR-3 and reports a plan that misses the deadline.")

    if args.surge:
        m = SURGE_RATE_MULTIPLIER
        plan_b = candidate_plans()[1]
        normal = plan_b.success_probability()
        surged = plan_b.success_probability(SURGE_REGIME)
        print(f"\nA surge multiplies the status flip rate by {m:.0f}, "
              f"so the half-life falls 90 s -> {90 / m:.0f} s.")
        print(f"  lambda {decay_rate(90.0):.6f} -> {decay_rate(90.0, m):.6f} per second")
        print("  The regime applies per predicate, since a surge does not change")
        print("  how fast obstructions appear on a helipad:")
        for pre in plan_b.preconditions:
            print(f"    {pre.label:26s} {pre.predicate:28s} "
                  f"w {pre.confidence():.3f} -> {pre.confidence(SURGE_REGIME):.3f}")
        print(f"  plan B joint confidence {normal:.3f} -> {surged:.3f}, "
              f"floor {COMMIT_FLOOR}")
        print(f"  verdict: {'COMMITTABLE' if plan_b.committable(regime=SURGE_REGIME) else 'BELOW THE FLOOR'}"
              f" at the moment it is written")
        print("  A system holding the peacetime rate reports "
              f"{normal:.3f}, commits, and raises no alarm.")


if __name__ == "__main__":
    _cli()
