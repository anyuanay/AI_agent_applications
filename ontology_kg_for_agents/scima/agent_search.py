"""Cost-efficient agent search over SCIMA-KG (Article 10).

Articles 1 through 9 built a place to keep knowledge and a way to keep it
honest and fresh. None of that helps if finding the right few dozen nodes
costs more than everything else the agent does. Article 10 is about the
search itself, and this module implements the four pieces it works through:

  * ``schema_predicate_set`` -- derive the predicates a walk may follow by
    searching the *schema* graph for paths from the start class to the goal
    class, instead of hand-listing them (Section 4).
  * ``guided_walk`` -- a relation-guided beam walk with an admissible
    straight-line bound, which visits 12 nodes without the bound and 11 with
    it, against 456 for a blind three-hop search and 800 for a full scan
    (Section 3).
  * ``ttl_for`` and ``PredicateCache`` -- per-predicate cache lifetimes read
    off each predicate's measured half-life, plus write invalidation
    (Section 5).
  * ``CostModel`` -- the token, dollar, and latency accounting that produces
    every row of the article's comparison table (Sections 1 and 8).

The demo runs over a deterministic dispatch scene around incident I-204
rather than a full 250K-triple city graph, so the example is fast and
reproducible. The walk, the schema derivation, and the cache logic are the
same logic that would run against the full KG; only the fan-out is smaller.
The 456-node blind-BFS figure is an analytic estimate from the city graph's
average degree (see ``blind_frontier``), not a count over this small scene.

Usage:
    python -m scima.agent_search --predicates
    python -m scima.agent_search --dispatch I-204
    python -m scima.agent_search --cache
    python -m scima.agent_search --costs
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import RDF, RDFS, OWL, Graph, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")

_ROOT = Path(__file__).resolve().parent.parent
_V1_0 = _ROOT / "ontologies" / "scima_owl_v1_0.ttl"

# ---- the article's cost model (Article 10, Section 1) ---------------------
# One place for every constant, so the comparison table is reproducible.
PRICE_INPUT_PER_TOKEN = 3.0 / 1_000_000     # dollars per input token
PRICE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000   # dollars per output token
FRAMING_TOKENS = 2_000        # system prompt plus task framing
RECORD_TOKENS = 120           # one vehicle record returned by a tool
REQUEST_TOKENS = 60           # one request the model writes
ANSWER_TOKENS = 80            # the model's final answer
ROUND_TRIP_SEC = 0.9          # one model call, request to response
HOP_SEC = 0.005               # one graph hop inside the engine
VECTOR_STAGE_SEC = 0.040      # one top-k similarity search
DECISION_BUDGET_SEC = 2.0     # a dispatch decision that arrives later is late

# City-graph shape, used for the blind-BFS branching estimate.
CITY_NODES = 63_000
CITY_EDGES = 250_000
FLEET_SIZE = 800

# Average emergency-response speed on city streets, km/h. Straight-line
# distance divided by this is a lower bound on road ETA, which is what makes
# it an admissible heuristic.
RESPONSE_SPEED_KMH = 40.0


# =========================================================================
# Section 4: derive the predicate set from the schema, do not type it
# =========================================================================
@dataclass
class SchemaRoutes:
    """What a backward search over the schema found.

    ``predicates`` is the set a walk may follow, and ``classes`` are the
    classes those routes pass through. Both are small because the schema is
    small, and neither depends on how much data sits under it.
    """
    predicates: list[str]
    classes: list[str]


def schema_routes(onto: ScimaOntology, start_class: str,
                  goal_class: str, max_hops: int = 4) -> SchemaRoutes:
    """Predicates lying on a schema path from ``start_class`` to ``goal_class``.

    Every object property declares a domain and a range, so the schema is
    itself a small graph with classes as nodes and properties as labelled
    edges. Walking it tells us which predicates could possibly lead from an
    incident to a vehicle within the hop budget, and everything else is a
    provable dead end for this goal.

    The schema graph has 43 classes and 30 object properties regardless of
    how much data sits under it, so this runs in well under a millisecond
    and is recomputed only when the schema changes.

    Edges are followed in both directions, because a walk may follow
    ``stationedAt`` backwards from a station to its vehicles.

    Two rules keep the derivation from over-generalising. A property is
    usable at a class only if that class is *comparable* to the property's
    declared domain, meaning one is a subclass of the other, and following
    it lands you on the declared range rather than on everything the range
    happens to share an ancestor with. Lateral properties, whose domain and
    range are comparable to each other (``adjacentTo`` between two zones,
    ``connectedTo`` between two segments), are added separately, because at
    the class level they are self-loops that a simple-path search can never
    traverse even though they move between different individuals.
    """
    g = onto.graph

    def qn(ref) -> str:
        s = str(ref)
        return "scima:" + s[len(str(SCIMA)):] if s.startswith(str(SCIMA)) else s

    def ancestors(cls: str) -> set[str]:
        ref = SCIMA[cls.split(":", 1)[1]]
        return {qn(x) for x in g.transitive_objects(ref, RDFS.subClassOf)
                if str(x).startswith(str(SCIMA))}

    def descendants(cls: str) -> set[str]:
        ref = SCIMA[cls.split(":", 1)[1]]
        return {qn(x) for x in g.transitive_subjects(RDFS.subClassOf, ref)
                if str(x).startswith(str(SCIMA))}

    def comparable(a: str, b: str) -> bool:
        """True when one class subsumes the other, so an instance of ``a``
        may satisfy a constraint declared on ``b``."""
        return a == b or b in ancestors(a) or a in ancestors(b)

    # Collect the declared object properties as (predicate, domain, range).
    declared: list[tuple[str, str, str]] = []
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        dom = g.value(prop, RDFS.domain)
        rng = g.value(prop, RDFS.range)
        if dom is None or rng is None:
            continue
        if not (str(dom).startswith(str(SCIMA)) and str(rng).startswith(str(SCIMA))):
            continue
        declared.append((qn(prop), qn(dom), qn(rng)))

    lateral = [(p, d, r) for p, d, r in declared if comparable(d, r)]
    directed = [(p, d, r) for p, d, r in declared if not comparable(d, r)]

    def steps(node: str) -> list[tuple[str, str]]:
        """Where one hop from ``node`` can land, and by which predicate."""
        out: list[tuple[str, str]] = []
        for p, d, r in directed:
            if comparable(node, d):
                out.append((p, r))
            if comparable(node, r):
                out.append((p, d))
        return out

    # Every simple path from start to goal within max_hops contributes its
    # predicates. Depth-first with a visited set keeps the paths simple.
    found: set[str] = set()
    on_path: set[str] = set()
    goals = ancestors(goal_class) | descendants(goal_class) | {goal_class}

    def walk(node: str, depth: int, seen: frozenset[str],
             used: tuple[str, ...], route: tuple[str, ...]) -> None:
        if node in goals and used:
            found.update(used)
            on_path.update(route)
            return
        if depth >= max_hops:
            return
        for pred, nxt in steps(node):
            if nxt in seen:
                continue
            walk(nxt, depth + 1, seen | {nxt}, used + (pred,), route + (nxt,))

    walk(start_class, 0, frozenset({start_class}), (), (start_class,))

    # A lateral property is worth following if some class it applies to is
    # already on a route to the goal, since taking it keeps you on that route.
    for p, d, _r in lateral:
        if any(comparable(c, d) for c in on_path):
            found.add(p)
            on_path.add(d)

    return SchemaRoutes(predicates=sorted(found), classes=sorted(on_path))


def schema_predicate_set(onto: ScimaOntology, start_class: str,
                         goal_class: str, max_hops: int = 4) -> list[str]:
    """Just the predicate list from :func:`schema_routes`."""
    return schema_routes(onto, start_class, goal_class, max_hops).predicates


def blind_frontier(hops: int = 3, n_nodes: int = CITY_NODES,
                   n_edges: int = CITY_EDGES) -> tuple[list[int], int]:
    """Nodes a blind breadth-first search touches, level by level.

    Each edge touches two nodes, so the average node sits on
    2 * n_edges / n_nodes edges. Expanding a node reaches that many
    neighbours; expanding each of those reaches one fewer, because one edge
    leads back where you came from.

    >>> blind_frontier(3)
    ([8, 56, 392], 456)
    """
    degree = round(2 * n_edges / n_nodes)
    levels: list[int] = []
    reached = degree
    for h in range(hops):
        levels.append(reached)
        reached *= degree - 1
    return levels, sum(levels)


# =========================================================================
# The dispatch scene: incident I-204 and the vehicles that could reach it
# =========================================================================
# Straight-line distance from each station to the scene, in km. Used only
# for the admissible bound, never as the answer.
STATION_KM = {
    "scima:Station_Crosstown": 0.9,
    "scima:Station_Riverside": 1.4,
    "scima:Station_Hilltop": 3.2,
    "scima:ControlZone_Z4": 5.5,   # the adjacent zone, far enough to lose the beam
}

# The predicates the I-204 scene actually populates. The schema admits more
# routes than this; the rest are declared but carry no data here, which costs
# the walk nothing because a predicate with no edges expands to nothing.
SCENE_PREDICATES = {
    "scima:withinZone", "scima:affects", "scima:adjacentTo",
    "scima:servedByStation", "scima:stationedAt",
}

# Road ETA in minutes, as the routing service reports it. Crosstown is
# nearest in a straight line and slowest by road, because its route crosses
# the flooded block of Main Street.
ROAD_ETA_MIN = {
    "scima:AMB_17": 4.2,
    "scima:AMB_22": 4.4,
    "scima:FIRE_12": 4.5,
    "scima:AMB_04": 6.0,
    "scima:AMB_09": 6.1,
    # Hilltop is farthest in a straight line at 3.2 km, so its 4.80-minute
    # bound already loses to a complete 4.2-minute answer and the branch is
    # pruned. Its true ETA still has to respect that bound, and at 5.0 it
    # does. That closeness is what makes AMB-31 the interesting rival once
    # the age of AMB-17's status evidence is priced in.
    "scima:AMB_31": 5.0,
}


class DispatchScene:
    """A deterministic scene around incident I-204, on top of SCIMA-OWL v1.0.

    Three stations serve the incident's zone, six vehicles sit at those
    stations, and three of the six are available ambulances. The correct
    answer is AMB-17 at Riverside, 1.4 km out, 4.2 minutes by road.
    """

    def __init__(self) -> None:
        self.g = Graph()
        self.g.parse(_V1_0, format="turtle")
        self._build()

    # ---- scene ---------------------------------------------------------
    def _add(self, s: str, p: URIRef, o) -> None:
        obj = o if isinstance(o, (URIRef, Literal)) else SCIMA[o]
        self.g.add((SCIMA[s], p, obj))

    def _build(self) -> None:
        A = self._add

        # --- hop 0: the incident ---
        A("Incident_I204", RDF.type, SCIMA.HazMatSpill)
        A("Incident_I204", SCIMA.hasStatus, Literal("active"))

        # --- hop 1: the zone that contains it, the segment it affects ---
        A("Incident_I204", SCIMA.withinZone, "ControlZone_Z7")
        A("Incident_I204", SCIMA.affects, "RoadSegment_Main_St_NB")
        A("ControlZone_Z7", RDF.type, SCIMA.ControlZone)
        A("RoadSegment_Main_St_NB", RDF.type, SCIMA.RoadSegment)
        # The segment sits in the same zone, so it leads nowhere new.
        A("RoadSegment_Main_St_NB", SCIMA.withinZone, "ControlZone_Z7")

        # --- hop 2: the stations covering the zone, plus one neighbour ---
        for st in ("Riverside", "Crosstown", "Hilltop"):
            A("ControlZone_Z7", SCIMA.servedByStation, f"Station_{st}")
            A(f"Station_{st}", RDF.type, SCIMA.Station)
        A("ControlZone_Z7", SCIMA.adjacentTo, "ControlZone_Z4")
        A("ControlZone_Z4", RDF.type, SCIMA.ControlZone)

        # --- hop 3: the vehicles based at those stations ---
        fleet = [
            ("AMB_17", SCIMA.Ambulance, "Riverside", "available"),
            ("AMB_22", SCIMA.Ambulance, "Riverside", "out_of_service"),
            ("FIRE_12", SCIMA.FireTruck, "Riverside", "available"),
            ("AMB_04", SCIMA.Ambulance, "Crosstown", "en_route"),
            ("AMB_09", SCIMA.Ambulance, "Crosstown", "available"),
            ("AMB_31", SCIMA.Ambulance, "Hilltop", "available"),
        ]
        for local, cls, station, status in fleet:
            A(local, RDF.type, cls)
            A(local, SCIMA.stationedAt, f"Station_{station}")
            A(local, SCIMA.hasStatus, Literal(status))
            A(local, SCIMA.etaMinutes, Literal(ROAD_ETA_MIN[f"scima:{local}"]))

    # ---- graph access --------------------------------------------------
    @staticmethod
    def _qn(ref) -> str:
        s = str(ref)
        return "scima:" + s[len(str(SCIMA)):] if s.startswith(str(SCIMA)) else s

    def _ref(self, qname: str) -> URIRef:
        return SCIMA[qname.split(":", 1)[1]]

    def neighbors(self, qname: str,
                  via: list[str]) -> list[tuple[str, str, float]]:
        """Neighbours reachable by one of ``via``, in either direction.

        Returns ``(neighbour, predicate, edge_cost)``. Topology edges cost
        nothing; the edge onto a vehicle costs that vehicle's road ETA, so
        the accumulated cost of a path is the response time it implies.
        """
        ref = self._ref(qname)
        allowed = {self._ref(p) for p in via}
        out: list[tuple[str, str, float]] = []
        for _, p, o in self.g.triples((ref, None, None)):
            if p in allowed and isinstance(o, URIRef):
                out.append((self._qn(o), self._qn(p), self.edge_cost(self._qn(o))))
        for s, p, _ in self.g.triples((None, None, ref)):
            if p in allowed and isinstance(s, URIRef):
                out.append((self._qn(s), self._qn(p), self.edge_cost(self._qn(s))))
        return sorted(set(out))

    def edge_cost(self, target: str) -> float:
        return ROAD_ETA_MIN.get(target, 0.0)

    def is_instance_of(self, qname: str, cls: str) -> bool:
        """Type check that honours the class hierarchy.

        This is ``rdfs:subClassOf*`` in Python. Asking for
        ``scima:Ambulance`` matches an ``scima:AirAmbulance`` too, which is
        what keeps the query correct when a new subclass is declared.
        """
        target = self._ref(cls)
        for t in self.g.objects(self._ref(qname), RDF.type):
            if t == target:
                return True
            if target in self.g.transitive_objects(t, RDFS.subClassOf):
                return True
        return False

    def available(self, qname: str) -> bool:
        return self.g.value(self._ref(qname), SCIMA.hasStatus) == Literal("available")

    def straight_line_bound(self, qname: str) -> float:
        """Optimistic minutes still to go, from straight-line distance.

        Roads cannot be shorter than the line between their endpoints, so
        this never overestimates the true remaining cost. That is exactly
        the admissibility condition A* needs to prune a branch and stay
        correct.
        """
        km = STATION_KM.get(qname)
        if km is None:
            return 0.0
        return km / RESPONSE_SPEED_KMH * 60.0


# =========================================================================
# Section 3: the relation-guided beam walk
# =========================================================================
@dataclass
class WalkResult:
    best: str | None
    cost: float
    visited: int
    frontier_log: list[list[str]] = field(default_factory=list)

    def __str__(self) -> str:
        answer = f"{self.best} at {self.cost:.1f} min" if self.best else "no answer"
        return f"{answer}, {self.visited} nodes visited"


def guided_walk(scene: DispatchScene, start: str, goal_class: str,
                via: list[str], max_hops: int = 3, beam_width: int = 3,
                prune: bool = True) -> WalkResult:
    """Walk from ``start`` following only ``via``, keeping the ``beam_width``
    most promising nodes per hop.

    With ``prune`` on, a branch whose optimistic lower bound already loses
    to the best complete answer found so far is skipped without being
    expanded. Because the bound is admissible, that cannot discard the true
    best answer.

    On the I-204 scene this visits 12 nodes with ``prune=False`` and 11
    with ``prune=True``, returning AMB-17 at 4.2 minutes either way.
    """
    frontier = [(start, 0.0)]
    seen = {start}
    best: str | None = None
    best_cost = float("inf")
    visited = 0
    log: list[list[str]] = []

    for _ in range(max_hops):
        candidates: list[tuple[str, float]] = []
        for node, cost in frontier:
            # Prune the whole branch before paying to expand it.
            if prune and cost + scene.straight_line_bound(node) >= best_cost:
                continue
            for nbr, _pred, edge_cost in scene.neighbors(node, via):
                if nbr in seen:
                    continue
                seen.add(nbr)
                visited += 1
                new_cost = cost + edge_cost

                # The goal test uses the class hierarchy, so any subclass counts.
                if scene.is_instance_of(nbr, goal_class):
                    if scene.available(nbr) and new_cost < best_cost:
                        best, best_cost = nbr, new_cost
                    continue
                candidates.append((nbr, new_cost))

        # Rank by optimistic total cost, keep the beam, drop the rest.
        candidates.sort(key=lambda nc: (nc[1] + scene.straight_line_bound(nc[0]), nc[0]))
        frontier = candidates[:beam_width]
        log.append([n for n, _ in frontier])

    return WalkResult(best=best, cost=best_cost, visited=visited, frontier_log=log)


# =========================================================================
# Section 5: cache lifetimes read off each predicate's rate of change
# =========================================================================
# Half-life of each predicate's truth, measured from SCIMA's change logs.
HALF_LIFE_SEC: dict[str, float] = {
    "scima:hasStatus": 90,                  # available / busy flips fast
    "scima:hasLocation": 300,               # a moving vehicle's fix
    "scima:servedByStation": 30 * 86_400,   # zone service boundaries
    "scima:stationedAt": 90 * 86_400,       # roster, changes twice a year
    "scima:hasSpeedLimit": math.inf,        # static until the road is re-signed
}

QUERIES_PER_MIN = 8.0          # 480 routing queries an hour
TOPOLOGY_HOPS = 9              # of the 12, the part that only writes invalidate
STATUS_HOPS = 3                # of the 12, the part that expires on its own
RE_DISPATCH_COST_MIN = 3.5     # what sending an already-busy vehicle costs
# Roster writes run about twice a week, far rarer than the queries, so the
# topology cache is effectively cold once per hour and warm after that.
TOPOLOGY_FILLS_PER_HOUR = 1.0


# A multi-casualty surge drives status flips well above the rate the change
# logs recorded, because vehicles are assigned, reassigned, and released far
# faster than in ordinary operations. The logged half-life is an estimate of
# a quantity that itself moves, so every rate below takes a multiplier.
SURGE_RATE_MULTIPLIER = 3.0


def decay_rate(predicate: str, rate_multiplier: float = 1.0) -> float:
    """The decay rate lambda = ln 2 / h for this predicate, scaled by the
    regime actually in force.

    The half-lives above were measured over ordinary operations. Holding
    lambda constant is an assertion that the transition law is
    time-homogeneous, which is testable and often false, so the multiplier
    is an explicit argument rather than a hidden assumption.

    >>> round(decay_rate("scima:hasStatus"), 6)
    0.007702
    >>> round(decay_rate("scima:hasStatus", SURGE_RATE_MULTIPLIER), 6)
    0.023105
    """
    half_life = HALF_LIFE_SEC[predicate]
    if half_life == math.inf:
        return 0.0
    return rate_multiplier * math.log(2) / half_life


def belief_weight(predicate: str, age_sec: float,
                  rate_multiplier: float = 1.0) -> float:
    """How much of its original weight a stored value still carries after
    ``age_sec`` with no fresh evidence, w(d) = exp(-lambda * d).

    This is the closed form of the transition law applied for d steps with
    no observation, not an independent modelling choice. For a two-state
    chain with per-step probability q of leaving the current state,
    (1 - q)^d = exp(d * ln(1 - q)), so lambda = -ln(1 - q).

    >>> round(belief_weight("scima:hasStatus", 25), 3)
    0.825
    >>> round(belief_weight("scima:hasStatus", 60), 3)
    0.63
    """
    return math.exp(-decay_rate(predicate, rate_multiplier) * age_sec)


def ttl_for(predicate: str, confidence_floor: float = 0.80,
            rate_multiplier: float = 1.0) -> float:
    """Longest age at which a cached value still carries ``confidence_floor``
    of the weight it had when it was written.

    A value decaying as w(d) = exp(-lambda * d) reaches the floor at
    ln(1 / floor) / lambda. The lifetime is a consequence of the floor and
    the current rate, so it moves when the rate does.

    >>> round(ttl_for("scima:hasStatus", 0.80), 2)
    28.97
    >>> round(ttl_for("scima:hasStatus", 0.50), 2)
    90.0
    >>> round(ttl_for("scima:hasStatus", 0.80, SURGE_RATE_MULTIPLIER), 2)
    9.66
    """
    lam = decay_rate(predicate, rate_multiplier)
    if lam == 0.0:
        return math.inf                     # rely on write invalidation
    return math.log(1 / confidence_floor) / lam


def hit_rate(ttl_sec: float, queries_per_min: float = QUERIES_PER_MIN) -> float:
    """Fraction of queries a cache with this lifetime serves from memory.

    One fill serves ``queries_per_min * ttl / 60`` queries on average, so
    one query in that many pays for the fill.
    """
    if ttl_sec == math.inf:
        return 1.0
    per_fill = queries_per_min * ttl_sec / 60.0
    return 0.0 if per_fill <= 1 else 1 - 1 / per_fill


def fresh_hops_per_query(confidence_floor: float = 0.80,
                         rate_multiplier: float = 1.0) -> float:
    """Graph hops a query still has to walk once the caches are warm.

    Topology is invalidated only by writes and status expires on its own,
    so the two parts of the 12-hop walk get very different hit rates. Under
    a surge the status lifetime shortens, its hit rate collapses, and the
    cache saves much less. That is the correct behaviour, since a cache is
    a bet that the world is holding still.
    """
    topology_miss = TOPOLOGY_FILLS_PER_HOUR / (QUERIES_PER_MIN * 60)
    status_ttl = ttl_for("scima:hasStatus", confidence_floor, rate_multiplier)
    status_miss = 1 - hit_rate(status_ttl)
    return TOPOLOGY_HOPS * topology_miss + STATUS_HOPS * status_miss


def stale_dispatch_rate(confidence_floor: float = 0.80,
                        rate_multiplier: float = 1.0,
                        ttl_sec: float | None = None) -> float:
    """Fraction of dispatches that go to a vehicle which has already gone
    busy.

    A cached entry is on average half its lifetime old, and only flips from
    available to busy hurt, which is about half of all flips.

    Pass ``ttl_sec`` to model a lifetime frozen in a config file while the
    world speeds up underneath it. Leave it ``None`` and the lifetime is
    re-derived from the rate actually in force, in which case lambda
    cancels out of the algebra entirely::

        the average entry is half a lifetime old, and
        exp(-lambda * T / 2) = sqrt(exp(-lambda * T)) = sqrt(floor), so
        rate = (1 - sqrt(floor)) / 2

    So the confidence floor is the quantity that governs the error rate.
    The lifetime is a consequence, and freezing a consequence is what
    breaks under load.

    >>> round(100 * stale_dispatch_rate(0.80), 1)
    5.3
    >>> round(100 * stale_dispatch_rate(0.80, SURGE_RATE_MULTIPLIER), 1)
    5.3
    >>> frozen = ttl_for("scima:hasStatus", 0.80)
    >>> round(100 * stale_dispatch_rate(0.80, SURGE_RATE_MULTIPLIER, frozen), 1)
    14.2
    """
    if ttl_sec is None:
        return (1 - math.sqrt(confidence_floor)) / 2
    lam = decay_rate("scima:hasStatus", rate_multiplier)
    return (1 - math.exp(-lam * ttl_sec / 2)) / 2


def stale_answer_penalty(confidence_floor: float = 0.80,
                         rate_multiplier: float = 1.0,
                         ttl_sec: float | None = None) -> float:
    """Expected minutes added to a response by serving a stale availability.

    Every wrong answer costs one re-dispatch, so this is just the rate
    above priced at ``RE_DISPATCH_COST_MIN``.
    """
    rate = stale_dispatch_rate(confidence_floor, rate_multiplier, ttl_sec)
    return rate * RE_DISPATCH_COST_MIN


# =========================================================================
# Section 5: from an average penalty to a per-candidate one
# =========================================================================
# How old each vehicle's last status post is, in seconds, at the moment the
# question is asked. Freshness is a property of the evidence rather than of
# the vehicle, so it lives beside the scene instead of inside it.
STATUS_AGE_SEC: dict[str, float] = {
    "scima:AMB_17": 25.0,
    "scima:AMB_22": 12.0,
    "scima:FIRE_12": 40.0,
    "scima:AMB_04": 8.0,
    "scima:AMB_09": 31.0,
    "scima:AMB_31": 3.0,
}


def expected_time(eta_min: float, status_age_sec: float,
                  rate_multiplier: float = 1.0,
                  redispatch_min: float = RE_DISPATCH_COST_MIN) -> float:
    """Expected response time once the age of the availability evidence is
    priced in.

    With probability w the vehicle is still available and arrives in
    ``eta_min``. With the rest it has gone busy since the last post,
    dispatch finds out, and the incident pays one re-dispatch on top::

        E[T] = t + (1 - w) * redispatch

    A fleet-wide average penalty cannot rank one vehicle against another.
    This can, because each candidate carries the age of its own evidence.

    >>> round(expected_time(4.2, 25), 2)
    4.81
    >>> round(expected_time(5.0, 3), 2)
    5.08
    >>> round(expected_time(4.2, 60), 2)
    5.5
    """
    w = belief_weight("scima:hasStatus", status_age_sec, rate_multiplier)
    return eta_min + (1 - w) * redispatch_min


@dataclass
class Candidate:
    vehicle: str
    eta_min: float
    status_age_sec: float
    weight: float
    expected_min: float

    def __str__(self) -> str:
        return (f"{self.vehicle:14s} eta {self.eta_min:4.1f} min   "
                f"status {self.status_age_sec:5.1f} s old   "
                f"w {self.weight:.3f}   E[T] {self.expected_min:5.2f} min")


def rank_candidates(scene: "DispatchScene",
                    ages: dict[str, float] | None = None,
                    rate_multiplier: float = 1.0,
                    goal_class: str = "scima:Ambulance") -> list[Candidate]:
    """Every available vehicle of ``goal_class``, ranked by expected time.

    Raw ETA order and expected-time order are not the same list, and the
    difference is the whole point. Ranking on distance alone hides how much
    the graph deserves to be believed.
    """
    ages = STATUS_AGE_SEC if ages is None else ages
    out: list[Candidate] = []
    for vehicle, eta in ROAD_ETA_MIN.items():
        if not scene.is_instance_of(vehicle, goal_class):
            continue
        if not scene.available(vehicle):
            continue
        age = ages.get(vehicle, 0.0)
        out.append(Candidate(
            vehicle=vehicle,
            eta_min=eta,
            status_age_sec=age,
            weight=belief_weight("scima:hasStatus", age, rate_multiplier),
            expected_min=expected_time(eta, age, rate_multiplier),
        ))
    return sorted(out, key=lambda c: (c.expected_min, c.vehicle))


def path_confidence(ages: dict[str, float],
                    rate_multiplier: float = 1.0) -> float:
    """Joint confidence in a multi-hop answer, as the product over the live
    hops the walk crossed.

    Topology hops hold for days and contribute a factor of essentially one.
    The live hops are the ones that decide it. Independence is assumed here,
    which is worth saying out loud, because correlated staleness such as one
    silent radio would make this optimistic.

    >>> round(path_confidence({"scima:hasStatus": 25, "scima:hasLocation": 25}), 3)
    0.779
    """
    w = 1.0
    for predicate, age in ages.items():
        w *= belief_weight(predicate, age, rate_multiplier)
    return w


class PredicateCache:
    """A cache keyed on the walk that produced the answer, with per-predicate
    expiry and exact write invalidation.

    Two queries share an entry only if they would walk the same edges under
    the same freshness rule, which is what the key encodes. A write to one
    predicate drops exactly the entries whose walk depended on it and leaves
    everything else alone.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple, object] = {}
        self.index: dict[str, set[tuple]] = {}

    @staticmethod
    def key(focal: str, goal_class: str, predicates: list[str],
            max_hops: int, confidence_floor: float) -> tuple:
        return (focal, goal_class, tuple(sorted(predicates)),
                max_hops, confidence_floor)

    def put(self, key: tuple, value: object, predicates: list[str]) -> None:
        self.entries[key] = value
        for p in predicates:
            self.index.setdefault(p, set()).add(key)

    def get(self, key: tuple):
        return self.entries.get(key)

    def invalidate(self, predicate: str) -> int:
        """Drop every entry whose walk used ``predicate``. Returns how many."""
        keys = self.index.pop(predicate, set())
        dropped = 0
        for k in keys:
            if self.entries.pop(k, None) is not None:
                dropped += 1
        return dropped


# =========================================================================
# Sections 1 and 8: what each search design costs
# =========================================================================
@dataclass
class Cost:
    name: str
    touched: float
    calls: int
    input_tokens: int
    dollars: float
    seconds: float
    outcome: str

    def __str__(self) -> str:
        return (f"{self.name:38s} {self.touched:>9}  {self.calls:>4} call(s)  "
                f"{self.input_tokens:>10,} tok  ${self.dollars:>8.3f}  "
                f"{self.seconds:>7.2f} s   {self.outcome}")


class CostModel:
    """Token, dollar, and latency accounting for one dispatch question.

    The only subtle part is that a stateless model re-reads its whole
    transcript on every turn, so a loop of k tool calls costs
    sum over k of (framing + record * (k - 1)) input tokens, which grows
    with the square of k rather than linearly.
    """

    @staticmethod
    def looping_calls(n: int, record_tokens: int = RECORD_TOKENS) -> int:
        """Input tokens for an agent that makes ``n`` sequential tool calls."""
        return n * FRAMING_TOKENS + record_tokens * (n * (n - 1) // 2)

    @staticmethod
    def single_call(payload_tokens: int) -> int:
        """Input tokens for one call that carries its whole payload at once."""
        return FRAMING_TOKENS + payload_tokens

    @staticmethod
    def dollars(input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * PRICE_INPUT_PER_TOKEN
                + output_tokens * PRICE_OUTPUT_PER_TOKEN)

    def table(self) -> list[Cost]:
        """Every design in the article's comparison table, priced."""
        rows: list[Cost] = []

        # 1. Scan every vehicle, one tool call each.
        tok = self.looping_calls(FLEET_SIZE)
        rows.append(Cost(
            "scan every vehicle, one call each", FLEET_SIZE, FLEET_SIZE, tok,
            self.dollars(tok, FLEET_SIZE * REQUEST_TOKENS),
            FLEET_SIZE * ROUND_TRIP_SEC,
            "right answer, 10 minutes after it mattered"))

        # 2. Pull all 800 records in one call.
        tok = self.single_call(FLEET_SIZE * RECORD_TOKENS)
        rows.append(Cost(
            "pull all 800 records in one call", FLEET_SIZE, 1, tok,
            self.dollars(tok, ANSWER_TOKENS), ROUND_TRIP_SEC,
            "impossible at sensor scale, where 50,000 records overflow the window"))

        # 3. Vector search, top 40.
        tok = self.single_call(40 * RECORD_TOKENS)
        rows.append(Cost(
            "vector search, top 40", 40, 1, tok,
            self.dollars(tok, ANSWER_TOKENS), ROUND_TRIP_SEC + VECTOR_STAGE_SEC,
            "AMB-17 sits at rank 37; the model still applies the filter by hand"))

        # 4. Blind three-hop BFS.
        _, blind = blind_frontier(3)
        tok = self.single_call(blind * RECORD_TOKENS)
        rows.append(Cost(
            "blind three-hop BFS", blind, 1, tok,
            self.dollars(tok, ANSWER_TOKENS), ROUND_TRIP_SEC + blind * HOP_SEC,
            "correct, but 444 of 456 nodes could never have been the answer"))

        # 5. The right walk in the wrong place: guided, but agent-side.
        tok = self.looping_calls(12)
        rows.append(Cost(
            "relation-guided walk, agent-side", 12, 12, tok,
            self.dollars(tok, 12 * REQUEST_TOKENS), 12 * ROUND_TRIP_SEC,
            "the loop costs more than the walk saves"))

        # 6. The same walk, run by the engine.
        tok = self.single_call(90 + 10 * 84)     # query text plus ten rows
        rows.append(Cost(
            "relation-guided walk, in the engine", 12, 1, tok,
            self.dollars(tok, ANSWER_TOKENS), ROUND_TRIP_SEC + 12 * HOP_SEC,
            "AMB-17, 1.4 km, ETA 4.2 min, inside the 2-second budget"))

        # 7. The same again, warm cache.
        fresh = fresh_hops_per_query()
        rows.append(Cost(
            "relation-guided walk, warm cache", round(fresh, 2), 1, tok,
            self.dollars(tok, ANSWER_TOKENS), ROUND_TRIP_SEC + fresh * HOP_SEC,
            "same answer; the saving is graph time, which pays off in batch"))

        return rows


# =========================================================================
# CLI
# =========================================================================
def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cost-efficient agent search over SCIMA-KG (Article 10).")
    parser.add_argument("--predicates", action="store_true",
                        help="derive the walk's predicate set from the schema")
    parser.add_argument("--dispatch", metavar="INCIDENT",
                        help="run the guided walk for an incident, e.g. I-204")
    parser.add_argument("--cache", action="store_true",
                        help="show per-predicate TTLs, hit rates, and the stale-answer price")
    parser.add_argument("--rank", action="store_true",
                        help="rank candidates by expected time, not raw ETA")
    parser.add_argument("--surge", action="store_true",
                        help="re-price the cache when the flip rate triples")
    parser.add_argument("--costs", action="store_true",
                        help="price every search design in the comparison table")
    args = parser.parse_args(argv)

    onto = ScimaOntology.load("v1.0")
    s = onto.summary()
    print(f"Loaded SCIMA-OWL v1.0: {s.n_classes} classes, "
          f"{s.n_properties} properties, {s.n_axioms} axioms")

    routes = schema_routes(onto, "scima:Incident", "scima:EmergencyVehicle")
    via = routes.predicates

    if args.predicates:
        print("\nSchema search: paths from scima:Incident to "
              "scima:EmergencyVehicle within 4 hops")
        print(f"  {len(routes.classes)} of {s.n_classes} classes lie on a route")
        print(f"  {len(via)} of {s.n_properties} properties can lie on such a route "
              f"({100 * len(via) / s.n_properties:.0f} percent of the vocabulary)")
        for p in via:
            mark = "  <- carries data in this scene" if p in SCENE_PREDICATES else ""
            print(f"    {p}{mark}")

    if args.dispatch:
        scene = DispatchScene()
        loose = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance",
                            via, prune=False)
        tight = guided_walk(scene, "scima:Incident_I204", "scima:Ambulance",
                            via, prune=True)
        levels, blind = blind_frontier(3)
        print(f"\nDispatch {args.dispatch}: nearest available ambulance")
        print(f"  full scan            : {FLEET_SIZE} records")
        print(f"  blind 3-hop BFS      : {' + '.join(map(str, levels))} = {blind} nodes")
        print(f"  guided walk          : {loose}")
        print(f"  guided + bound (A*)  : {tight}")
        print("  frontier by hop      :")
        for i, f in enumerate(tight.frontier_log, start=1):
            print(f"    hop {i}: {', '.join(f) if f else '(exhausted)'}")

    if args.cache:
        print("\nCache lifetime per predicate, at a 0.80 confidence floor:")
        for p in HALF_LIFE_SEC:
            ttl = ttl_for(p)
            shown = "no expiry (invalidate on write)" if ttl == math.inf else (
                f"{ttl:,.0f} s" if ttl < 3600 else f"{ttl / 86400:.1f} days")
            print(f"  {p:24s} {shown}")
        for floor in (0.80, 0.50):
            fresh = fresh_hops_per_query(floor)
            print(f"\n  floor {floor:.2f}: status TTL "
                  f"{ttl_for('scima:hasStatus', floor):.1f} s, "
                  f"hit rate {100 * hit_rate(ttl_for('scima:hasStatus', floor)):.1f}%, "
                  f"{fresh:.2f} fresh hops of 12 ({12 / fresh:.0f}x less graph work), "
                  f"stale-answer penalty {stale_answer_penalty(floor) * 60:.0f} s")

    if args.rank:
        scene = DispatchScene()
        print("\nCandidates ranked by expected time, E[T] = eta + (1 - w) x "
              f"{RE_DISPATCH_COST_MIN} min:")
        for c in rank_candidates(scene):
            print(f"  {c}")
        stale = dict(STATUS_AGE_SEC)
        stale["scima:AMB_17"] = 60.0
        print("\n  Now let AMB-17's status evidence age to 60 seconds:")
        for c in rank_candidates(scene, stale):
            print(f"  {c}")
        print("  The ranking flips, and nothing about the road network moved.")

        live = {"scima:hasStatus": 25.0, "scima:hasLocation": 25.0}
        print(f"\n  Joint confidence over the live hops: "
              f"{' x '.join(f'{belief_weight(p, a):.3f}' for p, a in live.items())}"
              f" = {path_confidence(live):.3f}")
        print("  So the answer is AMB-17 at about 78 percent, not AMB-17 as a fact.")

    if args.surge:
        floor = 0.80
        frozen = ttl_for("scima:hasStatus", floor)
        m = SURGE_RATE_MULTIPLIER
        print(f"\nA surge multiplies the status flip rate by {m:.0f}, so the "
              f"half-life falls {HALF_LIFE_SEC['scima:hasStatus']:.0f} s -> "
              f"{HALF_LIFE_SEC['scima:hasStatus'] / m:.0f} s.")
        print(f"  lifetime frozen at {frozen:.1f} s : "
              f"{100 * stale_dispatch_rate(floor, m, frozen):.1f}% stale, "
              f"{stale_answer_penalty(floor, m, frozen) * 60:.0f} s penalty")
        print(f"  lifetime re-derived      : "
              f"{ttl_for('scima:hasStatus', floor, m):.1f} s -> "
              f"{100 * stale_dispatch_rate(floor, m):.1f}% stale, "
              f"{stale_answer_penalty(floor, m) * 60:.0f} s penalty")
        print(f"  the floor is the invariant: (1 - sqrt({floor:.2f})) / 2 = "
              f"{100 * (1 - math.sqrt(floor)) / 2:.1f}%, with lambda cancelled out")
        for label, mult in (("normal", 1.0), ("surge", m)):
            fresh = fresh_hops_per_query(floor, mult)
            hr = hit_rate(ttl_for("scima:hasStatus", floor, mult))
            print(f"  {label:6s}: status hit rate {100 * hr:4.1f}%, "
                  f"{fresh:.2f} fresh hops of 12 ({12 / fresh:.1f}x less graph work)")

    if args.costs:
        print("\nWhat one dispatch question costs, by search design:")
        rows = CostModel().table()
        for r in rows:
            print(f"  {r}")
        loop, engine = rows[0], rows[5]
        print(f"\n  loop vs engine: {loop.input_tokens / engine.input_tokens:,.0f}x tokens, "
              f"{loop.dollars / engine.dollars:,.0f}x dollars, "
              f"{loop.seconds / engine.seconds:,.0f}x latency")
        print(f"  decision budget {DECISION_BUDGET_SEC:.0f} s: "
              f"{sum(1 for r in rows if r.seconds <= DECISION_BUDGET_SEC)} of "
              f"{len(rows)} designs meet it")


if __name__ == "__main__":
    _cli()
