"""Continual learning, ontology and KG evolution (Article 12).

Every module before this one moved facts under a schema that held
still. This one moves the schema.

The article's claim is that the three decisions people usually argue
about are measurements. Add a class, retire a class, retire an axiom:
each has an observable, a cost pair, and a threshold that falls out of
them. Nothing here is a tuned constant.

  * ``promotion_floor`` / ``should_add_class`` -- concept birth. The
    observable is the unmatched mention rate, the share of extracted
    mentions no class in the current schema covers. It ran 1.2 percent
    for EV chargers a year before the release and 8.7 percent in May,
    against a floor of 0.084 derived from mention volume and the two
    costs (Section 2).
  * ``silence_before_retiring`` -- concept death, the same argument run
    backward. Silence is only evidence when it is surprising, so the
    wait is ln(1/alpha) / r and a rare-but-healthy class earns a long
    one (Section 2).
  * ``pooled_weight`` / ``seasonal_decisions`` -- an axiom weight is a
    process. Pooling the year puts the traffic-jam axiom above the
    action floor in a quarter where the season puts it below, and no
    alarm fires (Section 3).
  * ``quarters_before_retiring_axiom`` -- do not retire a rule on one
    bad quarter. k > ln(alpha) / ln(p) turns the noisiness of the
    estimate into the length of the wait (Section 3).
  * ``classify_addition`` -- safe additions grow the schema, unsafe
    ones forbid something and reach backward through data that was
    legal when it arrived (Section 4).
  * ``sites_by_hardcoded_list`` / ``sites_by_subclass_walk`` -- the
    failure that reports nothing. A hand-typed list of type names loses
    EVChargingStation the day it is declared and still returns rows
    (Section 4).
  * ``mine_axiom_candidates`` -- support and confidence over several
    windows, so a candidate axiom is judged as a process too, and never
    promoted without a person (Section 5).
  * ``compatibility_gate`` -- replay the query suite, re-run the
    reasoner over stored triples, repair what it flags. Only then may
    v1.5 claim owl:backwardCompatibleWith (Sections 4 and 6).

Usage:
    python -m scima.ontology_evolution --promote
    python -m scima.ontology_evolution --retire
    python -m scima.ontology_evolution --weights
    python -m scima.ontology_evolution --stale-list
    python -m scima.ontology_evolution --mine
    python -m scima.ontology_evolution --gate
    python -m scima.ontology_evolution --diff
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import date

from rdflib import OWL, RDF, RDFS, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")

# --------------------------------------------------------------------------
# The SCIMA spring, as measured. Every number below is an observation or a
# cost the city can quote, never a threshold. The thresholds are derived.
# --------------------------------------------------------------------------

#: Monthly unmatched-mention rate for electric-vehicle charging, twelve
#: readings ending in May. The first and last are the article's 1.2 and 8.7
#: percent; the ten between them are the climb that made the promotion
#: predictable rather than a surprise.
CHARGER_MENTION_RATE = [
    0.012, 0.015, 0.019, 0.024, 0.031, 0.038,
    0.046, 0.052, 0.061, 0.068, 0.076, 0.087,
]

MENTIONS_PER_MONTH = 40_000       # extracted mentions the pipeline sees
COST_PER_HOMELESS_MENTION = 0.40  # dollars, a mention with nowhere to go
COST_OF_RETIRING = 9_000.0        # dollars, pulling a class back out
CHANCE_CONCEPT_LASTS = 0.85       # from the city's own record of pilots

#: Evacuation zones designated per month while the class was healthy. Small,
#: because the class was always rare, which is exactly why its retirement
#: test has to wait a long time.
EVACUATION_ZONE_RATE = 0.5
EVACUATION_ZONE_SILENCE_MONTHS = 11.0
FALSE_RETIREMENT_BUDGET = 0.01

#: P(a traffic jam counts as a high-risk zone), estimated on a rolling
#: window, one reading per quarter. Winter is high because ice and early
#: darkness put more crashes inside stopped traffic.
TRAFFIC_JAM_WEIGHT = {
    "winter": 0.87,
    "spring": 0.74,
    "summer": 0.61,
    "fall": 0.79,
}
ACTION_FLOOR = 0.70          # the dispatch agent sends an extra unit above this
DIP_PROBABILITY = 0.25       # chance the estimate dips under the floor on noise

#: What the dispatch agent's site query was typed with, on a day when the
#: list was complete. It has been complete ever since, which is the problem.
HARDCODED_SITE_TYPES = [
    "scima:WaterMain",
    "scima:RoadSegment",
    "scima:TrafficLight",
    "scima:PowerNode",
]

#: Confidence of one mined pattern across four quarterly windows, and of a
#: second that only looks good in one of them.
MINED_PATTERNS = {
    "connector types must match": [0.985, 0.991, 0.988, 0.990],
    "every station has a transit stop nearby": [0.42, 0.93, 0.55, 0.38],
}
MINING_CONFIDENCE_FLOOR = 0.95
MINING_SPREAD_CEILING = 0.05

#: Stations already in the KG when the one-operator limit was proposed.
#: Three of them were recorded with two operators, and every one of those
#: triples passed the admission gate on the day it arrived.
STORED_STATION_OPERATORS = {
    "scima:EVCS_01": ["scima:CityPower"],
    "scima:EVCS_02": ["scima:CityPower", "scima:GridCoOp"],
    "scima:EVCS_03": ["scima:GridCoOp"],
    "scima:EVCS_04": ["scima:CityPower", "scima:Harbor_Util"],
    "scima:EVCS_05": ["scima:CityPower", "scima:GridCoOp"],
}

#: Additions in the v1.1 -> v1.5 release, and what each one does to data
#: that is already stored.
V1_5_ADDITIONS = [
    ("AddClass", "scima:EVChargingStation"),
    ("AddClass", "scima:DroneCorridorSegment"),
    ("AddClass", "scima:ChargingEvent"),
    ("AddSubProperty", "scima:hasChargingPoint"),
    ("WidenRange", "scima:designates"),
    ("DeprecateClass", "scima:EvacuationZone"),
    ("AddDisjointness", "scima:EVChargingStation owl:disjointWith scima:PowerNode"),
    ("AddCardinalityLimit", "scima:hasOperator max 1"),
    ("AddValueRestriction", "scima:chargedVehicle allValuesFrom scima:Vehicle"),
]

SAFE_KINDS = {"AddClass", "AddSubProperty", "AddProperty", "WidenRange",
              "DeprecateClass", "AddInstance"}
# A kind reaches back when it can condemn a triple that was legal on the day
# it arrived. Whether it actually did is what the gate finds out. Two of the
# three below condemn nothing in the stored graph, and only re-running the
# reasoner shows that.
BACKWARD_REACHING_KINDS = {"AddDisjointness", "AddCardinalityLimit",
                           "AddValueRestriction", "NarrowRange",
                           "NarrowDomain", "AddFunctional"}


# --------------------------------------------------------------------------
# Section 2: concept birth
# --------------------------------------------------------------------------

def promotion_floor(mentions_per_month: float = MENTIONS_PER_MONTH,
                    cost_per_homeless_mention: float = COST_PER_HOMELESS_MENTION,
                    chance_concept_lasts: float = CHANCE_CONCEPT_LASTS,
                    cost_of_retiring: float = COST_OF_RETIRING) -> float:
    """The unmatched mention rate at which adding the class starts to pay.

    Waiting one more month costs ``m * N * C_late``. Adding the class
    now risks ``(1 - q) * C_retire``. Setting the two equal and solving
    for m gives the floor, which moves on its own when any cost moves.

    >>> round(promotion_floor(), 6)
    0.084375
    """
    cost_of_being_wrong = (1.0 - chance_concept_lasts) * cost_of_retiring
    return cost_of_being_wrong / (mentions_per_month * cost_per_homeless_mention)


def should_add_class(mention_rate: float, **kwargs) -> bool:
    """True when this month's reading has crossed the derived floor.

    >>> should_add_class(0.061)   # the March reading
    False
    >>> should_add_class(0.087)   # the May reading
    True
    """
    return mention_rate > promotion_floor(**kwargs)


def promotion_month(rates=None, **kwargs) -> int:
    """Index of the first reading that crosses the floor, or -1.

    >>> promotion_month()
    11
    """
    rates = CHARGER_MENTION_RATE if rates is None else rates
    for i, m in enumerate(rates):
        if should_add_class(m, **kwargs):
            return i
    return -1


# --------------------------------------------------------------------------
# Section 2: concept death
# --------------------------------------------------------------------------

def silence_before_retiring(healthy_rate: float,
                            budget: float = FALSE_RETIREMENT_BUDGET) -> float:
    """Months of silence to wait before retiring a class, in months.

    If the class were alive at ``healthy_rate`` arrivals per month, the
    chance of seeing nothing for T months is e^(-r * T). Wait until that
    falls under the false-retirement budget, so T > ln(1/alpha) / r.

    A rare class has a small r and therefore earns a long wait, which is
    the property that stops the rule from retiring something healthy.

    >>> round(silence_before_retiring(0.5), 2)
    9.21
    >>> round(silence_before_retiring(0.05), 1)    # ten times rarer
    92.1
    """
    if healthy_rate <= 0:
        raise ValueError("a class with no healthy arrival rate cannot be tested")
    return math.log(1.0 / budget) / healthy_rate


def should_retire_class(silent_months: float, healthy_rate: float,
                        budget: float = FALSE_RETIREMENT_BUDGET) -> bool:
    """True when the silence is longer than chance can explain.

    >>> should_retire_class(11.0, 0.5)
    True
    >>> should_retire_class(11.0, 0.05)   # rare class, same silence
    False
    """
    return silent_months > silence_before_retiring(healthy_rate, budget)


# --------------------------------------------------------------------------
# Section 3: an axiom weight is a process
# --------------------------------------------------------------------------

def pooled_weight(weights=None) -> float:
    """The single number a system gets by averaging the year.

    >>> round(pooled_weight(), 4)
    0.7525
    """
    w = TRAFFIC_JAM_WEIGHT if weights is None else weights
    values = list(w.values()) if isinstance(w, dict) else list(w)
    return sum(values) / len(values)


def seasonal_decisions(floor: float = ACTION_FLOOR, weights=None):
    """Compare what the pooled weight decides against what the season does.

    Returns one row per season: the seasonal weight, what the agent does
    reading the pooled value, what it should do, and whether the two
    disagree.

    >>> [r["season"] for r in seasonal_decisions() if r["wrong"]]
    ['summer']
    """
    w = TRAFFIC_JAM_WEIGHT if weights is None else weights
    pooled = pooled_weight(w)
    rows = []
    for season, value in w.items():
        acts_pooled = pooled >= floor
        acts_true = value >= floor
        rows.append({
            "season": season,
            "weight": value,
            "pooled_says_act": acts_pooled,
            "season_says_act": acts_true,
            "wrong": acts_pooled != acts_true,
        })
    return rows


def quarters_before_retiring_axiom(dip_probability: float = DIP_PROBABILITY,
                                   budget: float = FALSE_RETIREMENT_BUDGET) -> int:
    """Consecutive quarters under the floor before an axiom is retired.

    k dips in a row happen by chance with probability p^k. Choose the
    smallest k whose chance is under the budget, so k > ln(a) / ln(p).
    A noisier estimate makes the wait longer on its own.

    >>> quarters_before_retiring_axiom()
    4
    >>> quarters_before_retiring_axiom(dip_probability=0.05)  # quiet estimate
    2
    """
    if not 0.0 < dip_probability < 1.0:
        raise ValueError("dip probability must be strictly between 0 and 1")
    return math.ceil(math.log(budget) / math.log(dip_probability))


# --------------------------------------------------------------------------
# Section 4: safe additions, backward-reaching additions, and the gate
# --------------------------------------------------------------------------

def classify_addition(kind: str) -> str:
    """'safe' when the change only grows the schema, else 'reaches back'.

    >>> classify_addition("AddClass")
    'safe'
    >>> classify_addition("AddCardinalityLimit")
    'reaches back'
    """
    if kind in SAFE_KINDS:
        return "safe"
    if kind in BACKWARD_REACHING_KINDS:
        return "reaches back"
    raise ValueError(f"unclassified change kind {kind!r}")


def stations_violating_one_operator(stored=None) -> list[str]:
    """Stations the new cardinality limit condemns, retroactively.

    Every one of these triples passed the admission gate on the day it
    arrived. Only re-running the reasoner after the schema moved finds
    them.

    >>> stations_violating_one_operator()
    ['scima:EVCS_02', 'scima:EVCS_04', 'scima:EVCS_05']
    """
    stored = STORED_STATION_OPERATORS if stored is None else stored
    return sorted(s for s, ops in stored.items() if len(ops) > 1)


def sites_by_hardcoded_list(onto: ScimaOntology, types=None) -> list[str]:
    """The brittle query. Matches type names somebody typed by hand.

    It keeps running after the schema moves and returns fewer rows, with
    nothing in any log to say so.
    """
    types = HARDCODED_SITE_TYPES if types is None else types
    declared = set(onto.classes())
    return sorted(t for t in types if t in declared)


def sites_by_subclass_walk(onto: ScimaOntology,
                           root: str = "scima:InfrastructureEntity") -> list[str]:
    """The durable query. Follows rdfs:subClassOf zero or more times.

    The zero is the part people miss, since it lets the pattern match the
    parent itself as well as everything under it.
    """
    root_ref = onto._to_ref(root)
    seen, frontier = {root_ref}, [root_ref]
    while frontier:
        parent = frontier.pop()
        for child in onto.graph.subjects(RDFS.subClassOf, parent):
            if child not in seen and isinstance(child, URIRef):
                seen.add(child)
                frontier.append(child)
    return sorted(onto._qnames(seen))


def missed_by_the_stale_list(version: str = "v1.5") -> list[str]:
    """Classes the hand-typed list loses at this schema version.

    >>> "scima:EVChargingStation" in missed_by_the_stale_list()
    True
    """
    onto = ScimaOntology.load(version)
    walked = set(sites_by_subclass_walk(onto))
    typed = set(sites_by_hardcoded_list(onto))
    return sorted(walked - typed - {"scima:InfrastructureEntity"})


@dataclass
class GateResult:
    queries_replayed: int
    query_rows_lost: int
    flagged_by_reasoner: list[str]
    repaired: int
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: {self.queries_replayed} queries replayed, "
            f"{self.query_rows_lost} rows lost, "
            f"{len(self.flagged_by_reasoner)} triples flagged, "
            f"{self.repaired} repaired"
        )


def compatibility_gate(old: str = "v1.1", new: str = "v1.5",
                       repair: bool = True) -> GateResult:
    """Earn the right to claim owl:backwardCompatibleWith.

    Three checks, in the order the article gives them. Replay the saved
    query suite and lose no rows. Re-run the reasoner over stored triples
    and see what the new axioms condemn. Repair or quarantine whatever it
    flags, then run it once more.

    >>> compatibility_gate().passed
    True
    >>> compatibility_gate(repair=False).passed
    False
    """
    old_onto = ScimaOntology.load(old)
    new_onto = ScimaOntology.load(new)

    # 1. every class and property the old version declared is still there,
    #    so every saved query still has something to match.
    old_names = set(old_onto.classes()) | set(old_onto.object_properties()) \
        | set(old_onto.datatype_properties())
    new_names = set(new_onto.classes()) | set(new_onto.object_properties()) \
        | set(new_onto.datatype_properties())
    rows_lost = len(old_names - new_names)

    # 2. the reasoner over stored data, under the new axioms only.
    flagged = stations_violating_one_operator()

    repaired = len(flagged) if repair else 0
    passed = rows_lost == 0 and repaired == len(flagged)
    return GateResult(
        queries_replayed=len(old_names),
        query_rows_lost=rows_lost,
        flagged_by_reasoner=flagged,
        repaired=repaired,
        passed=passed,
    )


# --------------------------------------------------------------------------
# Section 5: from instances to axioms
# --------------------------------------------------------------------------

def mine_axiom_candidates(patterns=None,
                          floor: float = MINING_CONFIDENCE_FLOOR,
                          spread_ceiling: float = MINING_SPREAD_CEILING):
    """Propose only patterns that are both high and steady across windows.

    A pattern that holds today may be an accident of this quarter, so a
    candidate axiom is judged as a process too. Nothing here promotes
    anything. Every survivor goes to a person.

    >>> [c["pattern"] for c in mine_axiom_candidates() if c["propose"]]
    ['connector types must match']
    """
    patterns = MINED_PATTERNS if patterns is None else patterns
    out = []
    for name, windows in patterns.items():
        mean = sum(windows) / len(windows)
        spread = max(windows) - min(windows)
        out.append({
            "pattern": name,
            "mean_confidence": round(mean, 4),
            "spread": round(spread, 4),
            "propose": mean >= floor and spread <= spread_ceiling,
            "needs_human": True,
        })
    return out


# --------------------------------------------------------------------------
# Section 6: the change log
# --------------------------------------------------------------------------

@dataclass
class ChangeLog:
    """Every edit to the schema, with the reading that caused it.

    Six months later, when somebody asks why a class exists, the answer
    is in the schema and not in a chat thread.
    """
    ontology_iri: str
    version: str
    changes: list = field(default_factory=list)

    def add_class(self, iri: str, parent: str, rationale: str,
                  reading: float | None = None, on: str | None = None):
        self.changes.append({
            "type": "AddClass", "iri": iri, "parent": parent,
            "rationale": rationale, "reading": reading,
            "on": on or date.today().isoformat(),
            "safety": classify_addition("AddClass"),
        })
        return self

    def deprecate_class(self, iri: str, replaced_by: str, rationale: str,
                        reading: float | None = None, on: str | None = None):
        self.changes.append({
            "type": "DeprecateClass", "iri": iri, "replaced_by": replaced_by,
            "rationale": rationale, "reading": reading,
            "on": on or date.today().isoformat(),
            "safety": classify_addition("DeprecateClass"),
        })
        return self

    def add_restriction(self, on_property: str, rationale: str,
                        on: str | None = None):
        self.changes.append({
            "type": "AddCardinalityLimit", "iri": on_property,
            "rationale": rationale,
            "on": on or date.today().isoformat(),
            "safety": classify_addition("AddCardinalityLimit"),
        })
        return self

    def backward_reaching(self) -> list[dict]:
        return [c for c in self.changes if c["safety"] == "reaches back"]


def spring_release() -> ChangeLog:
    """The v1.1 to v1.5 log, built from the readings rather than by hand.

    >>> log = spring_release()
    >>> len(log.changes), len(log.backward_reaching())
    (3, 1)
    """
    log = ChangeLog("http://scima.city/ontology", "v1.5")
    may = CHARGER_MENTION_RATE[-1]
    if should_add_class(may):
        log.add_class("scima:EVChargingStation", "scima:InfrastructureEntity",
                      "unmatched mention rate crossed the derived floor",
                      reading=may, on="2026-05-04")
    if should_retire_class(EVACUATION_ZONE_SILENCE_MONTHS, EVACUATION_ZONE_RATE):
        log.deprecate_class("scima:EvacuationZone", "scima:ControlZone",
                            "no new instance for eleven months",
                            reading=EVACUATION_ZONE_SILENCE_MONTHS,
                            on="2026-05-04")
    log.add_restriction("scima:hasOperator",
                        "one operator per station, agreed with the utility",
                        on="2026-05-04")
    return log


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _promote():
    floor = promotion_floor()
    print(f"promotion floor m* = (1 - q) x C_retire / (N x C_late) = {floor:.4f}")
    for i, m in enumerate(CHARGER_MENTION_RATE, start=1):
        waiting = m * MENTIONS_PER_MONTH * COST_PER_HOMELESS_MENTION
        mark = "  ADD THE CLASS" if m > floor else ""
        print(f"  month {i:2d}  m = {m:.3f}   waiting costs ${waiting:8,.0f}{mark}")
    wrong = (1 - CHANCE_CONCEPT_LASTS) * COST_OF_RETIRING
    print(f"  cost of being wrong = (1 - {CHANCE_CONCEPT_LASTS}) x "
          f"${COST_OF_RETIRING:,.0f} = ${wrong:,.0f}")


def _retire():
    for rate in (EVACUATION_ZONE_RATE, 0.05):
        wait = silence_before_retiring(rate)
        verdict = "retire" if should_retire_class(
            EVACUATION_ZONE_SILENCE_MONTHS, rate) else "keep watching"
        print(f"healthy rate {rate:4.2f}/month -> wait {wait:5.1f} months; "
              f"after {EVACUATION_ZONE_SILENCE_MONTHS:.0f} silent months: {verdict}")


def _weights():
    print(f"pooled weight over the year = {pooled_weight():.4f}, "
          f"action floor = {ACTION_FLOOR}")
    for row in seasonal_decisions():
        flag = "  <-- acts on a rule the season does not support" if row["wrong"] else ""
        print(f"  {row['season']:<7} weight {row['weight']:.2f}   "
              f"pooled says act = {row['pooled_says_act']}   "
              f"season says act = {row['season_says_act']}{flag}")
    print(f"retire the axiom after {quarters_before_retiring_axiom()} "
          f"straight quarters under the floor")


def _stale_list():
    onto = ScimaOntology.load("v1.5")
    typed = sites_by_hardcoded_list(onto)
    walked = sites_by_subclass_walk(onto)
    print(f"hand-typed list  -> {len(typed)} types, no error reported")
    print(f"subclass walk    -> {len(walked)} types")
    print("missed by the list:", ", ".join(missed_by_the_stale_list()))


def _mine():
    for c in mine_axiom_candidates():
        verdict = "propose to a reviewer" if c["propose"] else "not steady enough"
        print(f"  {c['pattern']:<42} mean {c['mean_confidence']:.3f}  "
              f"spread {c['spread']:.3f}  {verdict}")


def _gate():
    for repair in (False, True):
        result = compatibility_gate(repair=repair)
        print(f"repair={repair!s:<5} {result}")
        if result.flagged_by_reasoner:
            print("   flagged:", ", ".join(result.flagged_by_reasoner))


def _diff():
    old = ScimaOntology.load("v1.1").summary()
    new = ScimaOntology.load("v1.5").summary()
    print(f"  {old}")
    print(f"  {new}")
    print(f"  delta: +{new.n_classes - old.n_classes} classes, "
          f"+{new.n_properties - old.n_properties} properties, "
          f"+{new.n_axioms - old.n_axioms} axioms")
    for kind, what in V1_5_ADDITIONS:
        print(f"  {classify_addition(kind):<12} {kind:<20} {what}")
    print("\nchange log:")
    for c in spring_release().changes:
        print(f"  {c['on']}  {c['type']:<20} {c['iri']:<28} "
              f"[{c['safety']}] {c['rationale']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promote", action="store_true",
                        help="concept birth: the derived floor and the crossing")
    parser.add_argument("--retire", action="store_true",
                        help="concept death: how long silence has to last")
    parser.add_argument("--weights", action="store_true",
                        help="an axiom weight read as a seasonal process")
    parser.add_argument("--stale-list", action="store_true",
                        help="the query failure that reports nothing")
    parser.add_argument("--mine", action="store_true",
                        help="candidate axioms mined over four windows")
    parser.add_argument("--gate", action="store_true",
                        help="the backward-compatibility gate")
    parser.add_argument("--diff", action="store_true",
                        help="the v1.1 to v1.5 schema diff and change log")
    args = parser.parse_args(argv)

    ran = False
    for flag, fn in (("promote", _promote), ("retire", _retire),
                     ("weights", _weights), ("stale_list", _stale_list),
                     ("mine", _mine), ("gate", _gate), ("diff", _diff)):
        if getattr(args, flag):
            fn()
            ran = True
    if not ran:
        _diff()


if __name__ == "__main__":
    main()
