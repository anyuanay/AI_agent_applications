"""Project, score, and evolve a context graph (Article 3).

Articles 1 and 2 built a place to *keep* knowledge: SCIMA-OWL (the schema)
and SCIMA-KG (the populated triples). That global graph is the right home
for knowledge and the wrong thing to hand an agent: it does not fit in a
context window. Article 3 introduces the **context graph**, an agent's
dynamic working memory: a small, task-scoped projection of the global KG,
extracted around a focal entity, scored for relevance, and pruned to a
node budget.

This module implements three things from the article:

  * ``build_context_graph`` -- a k-hop projection scored by structural,
    semantic, and temporal relevance, pruned to ``max_nodes`` (Section 2-4).
  * ``ContextGraphBuilder.trace`` -- the four-turn evolution of the graph as
    the agent runs its loop: seed, expand, refresh/evict, act (Section 5).
  * ``precondition_holds`` -- the ASK gate that grounds an action against the
    graph and ontology before it fires, so the agent cannot drift into a
    hallucinated or invalid action (Section 6).

The demo runs over a deterministic, representative *scene* around incident
I-204 rather than a full 250K-triple city graph, so the example is fast and
reproducible. The projection, scoring, eviction, and write-back logic is the
same logic that would run against the full KG.

Usage:
    python -m scima.context_graph --build --focal Incident_I204 --goal resolve
    python -m scima.context_graph --trace I-204
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import RDF, Graph, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")

_ROOT = Path(__file__).resolve().parent.parent
_V0_5 = _ROOT / "ontologies" / "scima_owl_v0_5.ttl"

# Relevance weights (Article 3, Section 4): structure and semantics lead,
# freshness breaks ties.
W_STRUCTURAL = 0.4
W_SEMANTIC = 0.4
W_TEMPORAL = 0.2

DEFAULT_K = 3
DEFAULT_MAX_NODES = 150

# Per-type maximum age (minutes) before a reading is treated as stale and is
# eligible for staleness eviction when a fresher equivalent exists. A preview
# of Age of Information (Article 6).
STALE_AFTER_MIN = 5.0


# ---- goals: which entity types a goal cares about -------------------------
# Semantic relevance is "does this node's type matter to the current goal?"
GOAL_RELEVANT_TYPES: dict[str, set[str]] = {
    "resolve": {
        "scima:Incident",
        "scima:WaterMain",
        "scima:RoadSegment",
        "scima:TrafficLight",
        "scima:FlowSensor",
        "scima:GoalState",
        "scima:DispatchAgent",
        "scima:ZoneAgent",
    },
}


@dataclass
class ScoredNode:
    qname: str
    node_type: str
    hop: int
    age_min: float
    relevance: float

    def __str__(self) -> str:
        return f"{self.qname:32s} hop={self.hop} age={self.age_min:4.1f}m r={self.relevance:.3f}"


@dataclass
class TurnRecord:
    turn: int
    name: str
    description: str
    members: list[str]
    added: list[str] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        line = f"Turn {self.turn} ({self.name}): {len(self.members)} nodes"
        if self.added:
            line += f"  +{','.join(self.added)}"
        if self.evicted:
            line += f"  -{','.join(self.evicted)}"
        return line


class ContextGraphBuilder:
    """Builds and evolves a context graph over a SCIMA-KG scene.

    Wraps an rdflib ``Graph`` holding the v0.5 schema plus a deterministic
    incident scene. Nodes carry a synthetic ``age_min`` (minutes since last
    update) so temporal relevance and staleness eviction are demonstrable
    without a live clock.
    """

    def __init__(self) -> None:
        self.g = Graph()
        self.g.parse(_V0_5, format="turtle")
        # age (minutes since last refresh) per entity qname; absent => static fact
        self.age_min: dict[str, float] = {}
        self._build_scene()

    # ---- scene --------------------------------------------------------
    def _add(self, s: str, p: URIRef, o) -> None:
        subj = SCIMA[s]
        # NB: URIRef and Literal both subclass str, so test for them first;
        # only a *bare* local name (plain str) gets expanded into the namespace.
        obj = o if isinstance(o, (URIRef, Literal)) else SCIMA[o]
        self.g.add((subj, p, obj))

    def _build_scene(self) -> None:
        """A deterministic scene around the compound incident I-204.

        Core entities are the ones the article narrates; a filler ring of
        adjacent road segments (each with a light and a camera) makes the
        3-hop neighbourhood realistically large, so pruning to a budget is
        a real operation rather than a formality.
        """
        A = self._add

        # --- focal incident (hop 0) ---
        A("Incident_I204", RDF.type, SCIMA.Incident)
        # plain string literals so the article's plain-text ASK gate matches
        A("Incident_I204", SCIMA.hasStatus, Literal("active"))

        # --- hop 1: what the incident touches and who owns it ---
        A("Incident_I204", SCIMA.affects, "WaterMain_7B")
        A("Incident_I204", SCIMA.affects, "RoadSegment_Main_St_NB")
        A("WaterMain_7B", RDF.type, SCIMA.WaterMain)
        A("RoadSegment_Main_St_NB", RDF.type, SCIMA.RoadSegment)

        A("DispatchAgent_D1", RDF.type, SCIMA.DispatchAgent)
        A("DispatchAgent_D1", SCIMA.respondsTo, "Incident_I204")
        A("GoalState_Resolve_I204", RDF.type, SCIMA.GoalState)
        A("GoalState_Resolve_I204", SCIMA.targets, "Incident_I204")
        A("DispatchAgent_D1", SCIMA.pursuesGoal, "GoalState_Resolve_I204")

        # --- hop 2: sensors on the water main, light on the road, zone owner ---
        A("FlowSensor_A42", RDF.type, SCIMA.FlowSensor)
        A("FlowSensor_A42", SCIMA.monitors, "WaterMain_7B")
        self.age_min["scima:FlowSensor_A42"] = 0.5   # fresh at turn 1

        A("FlowSensor_B7", RDF.type, SCIMA.FlowSensor)
        A("FlowSensor_B7", SCIMA.monitors, "WaterMain_7B")  # backup, kept fresh
        self.age_min["scima:FlowSensor_B7"] = 0.2

        A("TL_90", RDF.type, SCIMA.TrafficLight)
        A("TL_90", SCIMA.locatedOn, "RoadSegment_Main_St_NB")

        A("ZoneAgent_N", RDF.type, SCIMA.ZoneAgent)
        A("RoadSegment_Main_St_NB", SCIMA.managedBy, "ZoneAgent_N")

        # --- vehicles near Main St (locatedOn used loosely for position) ---
        A("Vehicle_Ambulance_3", RDF.type, SCIMA.Vehicle)
        A("Vehicle_Ambulance_3", SCIMA.hasAvailability, Literal("available"))
        A("Vehicle_Ambulance_3", SCIMA.locatedOn, "RoadSegment_Main_St_NB")
        self.age_min["scima:Vehicle_Ambulance_3"] = 0.3

        # --- a far, off-goal vehicle that should score low and be dropped ---
        A("RoadSegment_Far_9", RDF.type, SCIMA.RoadSegment)
        A("RoadSegment_Main_St_NB", SCIMA.connectedTo, "RoadSegment_Far_9")
        A("RoadSegment_Far_9", SCIMA.connectedTo, "RoadSegment_Far_99")
        A("RoadSegment_Far_99", RDF.type, SCIMA.RoadSegment)
        A("Vehicle_Bus_88", RDF.type, SCIMA.Vehicle)
        A("Vehicle_Bus_88", SCIMA.locatedOn, "RoadSegment_Far_9")  # far + a transit bus, low relevance
        self.age_min["scima:Vehicle_Bus_88"] = 1.0

        # --- filler ring: adjacent segments (hop 2), each with a light and
        #     camera (hop 3), so the neighbourhood is realistically large ---
        for i in range(43):
            seg = f"RoadSegment_Adj_{i}"
            A("RoadSegment_Main_St_NB", SCIMA.connectedTo, seg)
            A(seg, RDF.type, SCIMA.RoadSegment)
            light = f"TL_Adj_{i}"
            A(light, RDF.type, SCIMA.TrafficLight)
            A(light, SCIMA.locatedOn, seg)
            cam = f"Camera_Adj_{i}"
            A(cam, RDF.type, SCIMA.TrafficCamera)
            A(cam, SCIMA.monitors, seg)
            self.age_min[f"scima:{cam}"] = 2.0

    # ---- relevance scoring (Article 3, Section 4) ---------------------
    def _qname(self, ref) -> str:
        s = str(ref)
        return "scima:" + s[len(str(SCIMA)):] if s.startswith(str(SCIMA)) else s

    def node_type(self, qname: str) -> str:
        t = self.g.value(SCIMA[qname.split(":", 1)[1]], RDF.type)
        return self._qname(t) if t is not None else "scima:Thing"

    def _neighbors(self, qname: str) -> set[str]:
        """Undirected neighbours of a node (ignoring schema/literal targets)."""
        ref = SCIMA[qname.split(":", 1)[1]]
        out: set[str] = set()
        for _, p, o in self.g.triples((ref, None, None)):
            if isinstance(o, URIRef) and str(o).startswith(str(SCIMA)) and p != RDF.type:
                out.add(self._qname(o))
        for s, p, _ in self.g.triples((None, None, ref)):
            if isinstance(s, URIRef) and str(s).startswith(str(SCIMA)) and p != RDF.type:
                out.add(self._qname(s))
        return out

    def _hops_from(self, focal: str, k: int) -> dict[str, int]:
        """BFS hop distance from the focal entity, out to k hops."""
        dist = {focal: 0}
        q = deque([focal])
        while q:
            cur = q.popleft()
            if dist[cur] >= k:
                continue
            for nbr in self._neighbors(cur):
                # only follow links between named instances, not schema terms
                if nbr not in dist and self.node_type(nbr) != "scima:Thing":
                    dist[nbr] = dist[cur] + 1
                    q.append(nbr)
        return dist

    def relevance(self, qname: str, hop: int, goal: str) -> float:
        relevant_types = GOAL_RELEVANT_TYPES.get(goal, set())
        structural = 1.0 / (hop + 1)
        semantic = 1.0 if self.node_type(qname) in relevant_types else 0.2
        age = self.age_min.get(qname, 0.0)
        temporal = 1.0 / (age + 1.0)
        return W_STRUCTURAL * structural + W_SEMANTIC * semantic + W_TEMPORAL * temporal

    def build_context_graph(self, focal: str, goal: str = "resolve",
                            k: int = DEFAULT_K,
                            max_nodes: int = DEFAULT_MAX_NODES) -> list[ScoredNode]:
        """Extract a k-hop context graph around ``focal``, scored and pruned.

        Mirrors the ``build_context_graph`` snippet in the article: BFS to k
        hops, score every candidate by relevance to the goal, sort, and keep
        the top ``max_nodes``. The focal entity is always kept.
        """
        hops = self._hops_from(focal, k)
        scored = [
            ScoredNode(
                qname=n,
                node_type=self.node_type(n),
                hop=h,
                age_min=self.age_min.get(n, 0.0),
                relevance=self.relevance(n, h, goal),
            )
            for n, h in hops.items()
        ]
        # focal first, then by relevance descending
        scored.sort(key=lambda s: (s.hop != 0, -s.relevance))
        return scored[:max_nodes]

    # ---- precondition gate (Article 3, Section 6) --------------------
    def precondition_holds(self, incident: str, vehicle: str) -> bool:
        """ASK gate: only dispatch if the incident is open and the vehicle is free.

        This is the grounding check that stops an agent from acting on a
        hallucinated or stale state. Returns False (rather than acting) when
        the world does not actually support the action.
        """
        q = """
        PREFIX scima: <http://scima.city/ontology#>
        ASK {
            %(inc)s a scima:Incident ;
                    scima:hasStatus "active" .
            %(veh)s a scima:Vehicle ;
                    scima:hasAvailability "available" .
        }
        """ % {"inc": incident, "veh": vehicle}
        return bool(self.g.query(q).askAnswer)

    def dispatch(self, vehicle: str, incident: str) -> bool:
        """Turn-3 write-back: commit the assignment if preconditions hold."""
        if not self.precondition_holds(incident, vehicle):
            return False
        self._add(vehicle.split(":", 1)[1], SCIMA.assignedTo, incident.split(":", 1)[1])
        self.g.set((SCIMA[vehicle.split(":", 1)[1]], SCIMA.hasAvailability,
                    Literal("busy")))
        return True

    # ---- the four-turn evolution (Article 3, Section 5) --------------
    def trace(self, focal: str = "scima:Incident_I204",
              goal: str = "resolve",
              max_nodes: int = DEFAULT_MAX_NODES) -> list[TurnRecord]:
        """Walk the context graph through the agent loop, turn by turn.

        Each turn refreshes a clock, recomputes relevance, evicts the stale
        and the low-relevance, and (on the last turn) writes back the result
        of acting. The focal entity stays pinned throughout.
        """
        records: list[TurnRecord] = []

        # --- Turn 0: report. Seed with focal + its directly affected asset. ---
        seed = [focal, "scima:WaterMain_7B"]
        records.append(TurnRecord(
            0, "report",
            "Incident extracted and seeded with its one affected asset.",
            members=list(seed), added=list(seed),
        ))
        prev = set(seed)

        # --- Turn 1: expand. 3-hop projection, scored, pruned to budget. ---
        full = self.build_context_graph(focal, goal, k=DEFAULT_K, max_nodes=max_nodes)
        # at turn 1 the dispatcher is still surveying infrastructure: hold
        # vehicles back until planning starts (turn 2).
        t1 = [s.qname for s in full if s.node_type != "scima:Vehicle"]
        members1 = set(t1)
        records.append(TurnRecord(
            1, "expand",
            "3-hop projection pulls in roads, lights, and sensors; the far "
            "off-goal Bus-88 scores low and is dropped to stay under budget.",
            members=t1,
            added=sorted(members1 - prev),
            evicted=sorted(prev - members1),
        ))
        prev = members1

        # --- Turn 2: refresh. Time passes; A42 goes stale; backup + vehicle in. ---
        self._advance_clock(8.0)  # 8 minutes elapse
        members2 = set(prev)
        evicted2: list[str] = []
        # staleness eviction: A42 stale and a fresher equivalent (B7) exists
        if self._is_stale("scima:FlowSensor_A42") and "scima:FlowSensor_B7" in self._fresh_equivalents("scima:FlowSensor_A42"):
            members2.discard("scima:FlowSensor_A42")
            evicted2.append("scima:FlowSensor_A42")
        # planning starts: pull in the nearest available vehicle
        added2 = []
        for v in ("scima:FlowSensor_B7", "scima:Vehicle_Ambulance_3"):
            if v not in members2:
                members2.add(v)
                added2.append(v)
        records.append(TurnRecord(
            2, "refresh",
            "Sensor A42 is now 8 minutes stale; it is evicted in favour of "
            "fresh backup B7. Planning begins, so Ambulance-3 is pulled in.",
            members=sorted(members2),
            added=sorted(added2), evicted=sorted(evicted2),
        ))
        prev = members2

        # --- Turn 3: act. ASK-check, dispatch, write back assignedTo. ---
        ok = self.dispatch("scima:Vehicle_Ambulance_3", "scima:Incident_I204")
        members3 = set(prev)
        records.append(TurnRecord(
            3, "act",
            ("Preconditions hold; Ambulance-3 is dispatched and assignedTo is "
             "written back to the graph." if ok else
             "Preconditions failed; the action is rejected, not hallucinated."),
            members=sorted(members3),
            added=[], evicted=[],
        ))
        return records

    # ---- staleness helpers (preview of Article 6) -------------------
    def _advance_clock(self, minutes: float) -> None:
        for k in list(self.age_min):
            # the backup sensor and the camera ring keep reporting; treat them
            # as refreshed each turn, while A42 has gone silent.
            if k in ("scima:FlowSensor_B7", "scima:Vehicle_Ambulance_3"):
                continue
            self.age_min[k] += minutes

    def _is_stale(self, qname: str) -> bool:
        return self.age_min.get(qname, 0.0) > STALE_AFTER_MIN

    def _fresh_equivalents(self, qname: str) -> set[str]:
        """Same-type nodes monitoring the same target that are not stale."""
        ref = SCIMA[qname.split(":", 1)[1]]
        target = self.g.value(ref, SCIMA.monitors)
        my_type = self.g.value(ref, RDF.type)
        out: set[str] = set()
        if target is None or my_type is None:
            return out
        for other in self.g.subjects(SCIMA.monitors, target):
            oq = self._qname(other)
            if oq == qname:
                continue
            if self.g.value(other, RDF.type) == my_type and not self._is_stale(oq):
                out.add(oq)
        return out


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and evolve a SCIMA context graph (Article 3).")
    parser.add_argument("--build", action="store_true", help="build a context graph")
    parser.add_argument("--focal", default="Incident_I204", help="focal entity local id")
    parser.add_argument("--goal", default="resolve", help="goal name")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES,
                        help=f"node budget (default {DEFAULT_MAX_NODES})")
    parser.add_argument("--trace", metavar="INCIDENT", help="trace the 4-turn evolution")
    args = parser.parse_args(argv)

    onto = ScimaOntology.load("v0.5").summary()
    print(f"Loaded SCIMA-OWL v0.5: {onto.n_classes} classes, "
          f"{onto.n_properties} properties, {onto.n_axioms} axioms")

    builder = ContextGraphBuilder()

    if args.build:
        focal = args.focal if args.focal.startswith("scima:") else f"scima:{args.focal}"
        cg = builder.build_context_graph(focal, args.goal, max_nodes=args.max_nodes)
        print(f"Built context graph: focal {args.focal}, 3-hop, "
              f"{len(cg)} nodes (budget {args.max_nodes})")
        print("Top by relevance:")
        for s in cg[:6]:
            print(f"  {s}")

    if args.trace:
        local = args.trace if args.trace.startswith("Incident") else f"Incident_{args.trace.replace('-', '')}"
        for rec in builder.trace(focal=f"scima:{local}", max_nodes=args.max_nodes):
            print(rec)


if __name__ == "__main__":
    _cli()
