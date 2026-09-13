"""Article 13's memory exchange, adapted from SOLIR's SAKD notes.

Offline teaching model. One binary claim has a fixed state during its interval.
Distinct reports must be conditionally independent given that state. Likelihoods
are supplied assumptions, not source authentication or measured reliability.
No LLM, training, world-state transition, or general graph merge is implemented.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import argparse
import json
import math
from pathlib import Path


def probability(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities must be finite and between zero and one")
    return value


def bayesian_open(prior: float, report_given_open: float,
                  report_given_closed: float) -> float:
    """Update the binary state using conditional report likelihoods.

    >>> round(bayesian_open(0.8, 0.1, 0.9), 3)
    0.308
    """
    open_weight = probability(prior) * probability(report_given_open)
    closed_weight = (1 - prior) * probability(report_given_closed)
    total = open_weight + closed_weight
    if total == 0:
        raise ValueError("report has zero probability under this model")
    return open_weight / total


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps need a timezone")
    return parsed


@dataclass(frozen=True)
class Scope:
    entity: str
    direction: str
    vehicle: str
    start: str
    end: str

    def __post_init__(self):
        if not all((self.entity, self.direction, self.vehicle)):
            raise ValueError("entity, direction, and vehicle are required")
        if instant(self.start) >= instant(self.end):
            raise ValueError("scope needs a positive interval")


@dataclass(frozen=True)
class Report:
    observation_id: str
    source: str
    scope: Scope
    observed_at: str
    given_open: float
    given_closed: float

    def __post_init__(self):
        if not self.observation_id or not self.source:
            raise ValueError("observation identity and source are required")
        probability(self.given_open)
        probability(self.given_closed)
        if not instant(self.scope.start) <= instant(self.observed_at) < instant(self.scope.end):
            raise ValueError("observation falls outside the claim interval")


@dataclass(frozen=True)
class Change:
    version: int
    observation_id: str
    prior_open: float
    posterior_open: float
    assessed_at: str
    validation: str = "accepted"


@dataclass(frozen=True)
class Memory:
    holder: str
    claim_id: str
    scope: Scope
    open_probability: float
    assessed_at: str
    source_snapshot: str
    inherited_evidence: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]
    reports: tuple[Report, ...] = ()
    history: tuple[Change, ...] = ()
    version: int = 0

    def __post_init__(self):
        probability(self.open_probability)
        instant(self.assessed_at)
        if not all((self.holder, self.claim_id, self.source_snapshot)):
            raise ValueError("holder, claim ID, and source snapshot are required")

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(self.inherited_evidence) | frozenset(
            report.observation_id for report in self.reports)

    def receive(self, report: Report, assessed_at: str) -> 'Memory':
        """Validate and accept evidence. A duplicate returns the same snapshot.

        Identity collisions require review. Distinct report IDs do not prove
        statistical independence. Callers must establish that assumption.
        """
        if report.scope != self.scope:
            raise ValueError("claim scope mismatch requires review")
        for existing in self.reports:
            if existing.observation_id == report.observation_id:
                if existing != report:
                    raise ValueError("observation ID has conflicting contents")
                return self
        if report.observation_id in self.inherited_evidence:
            raise ValueError("report ID belongs to inherited evidence")
        if instant(assessed_at) < max(instant(self.assessed_at), instant(report.observed_at)):
            raise ValueError("assessment precedes memory or observation")
        posterior = bayesian_open(self.open_probability, report.given_open,
                                  report.given_closed)
        change = Change(self.version + 1, report.observation_id,
                        self.open_probability, posterior, assessed_at)
        return replace(self, open_probability=posterior, assessed_at=assessed_at,
                       reports=self.reports + (report,), history=self.history + (change,),
                       version=change.version)

    def context(self) -> dict:
        """Select the claim and its graph links for a route request."""
        return {"holder": self.holder, "memory_version": self.version,
                "source_snapshot": self.source_snapshot, "claim_id": self.claim_id,
                "scope": asdict(self.scope), "open": self.open_probability,
                "closed": 1 - self.open_probability,
                "assessed_at": self.assessed_at,
                "evidence_ids": sorted(self.evidence_ids),
                "reports": [asdict(report) for report in self.reports],
                "triples": [list(triple) for triple in self.triples]}

    def save(self, path: Path) -> None:
        """Write a snapshot through an atomic file replacement."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> 'Memory':
        data = json.loads(Path(path).read_text())
        data['scope'] = Scope(**data['scope'])
        data['inherited_evidence'] = tuple(data['inherited_evidence'])
        data['triples'] = tuple(tuple(triple) for triple in data['triples'])
        data['reports'] = tuple(Report(**{**r, 'scope': Scope(**r['scope'])})
                                for r in data['reports'])
        data['history'] = tuple(Change(**c) for c in data['history'])
        return cls(**data)


def seed(source: Memory, holder: str, role_entity: str) -> Memory:
    """Copy common claim structure and the selected role's graph links."""
    selected = tuple(t for t in source.triples
                     if t[0] in {source.scope.entity, source.claim_id, role_entity})
    return replace(source, holder=holder, triples=selected,
                   source_snapshot=f"{source.holder}:v{source.version}",
                   history=(), version=0)


@dataclass(frozen=True)
class Probe:
    holder: str
    claim_id: str
    scope: Scope
    version: int
    open_probability: float
    evidence_ids: frozenset[str]


def probe(actor: Memory) -> Probe:
    """Read one stored claim. This does not test retrieval or action quality."""
    return Probe(actor.holder, actor.claim_id, actor.scope, actor.version,
                 actor.open_probability, actor.evidence_ids)


def targeted_cast(source: Memory, response: Probe, report_budget: int) -> tuple[Report, ...]:
    """Select missing reports in source order under a report-count budget."""
    if not isinstance(report_budget, int) or report_budget < 0:
        raise ValueError("report budget must be a nonnegative integer")
    if response.claim_id != source.claim_id or response.scope != source.scope:
        raise ValueError("probe concerns a different claim")
    return tuple(r for r in source.reports
                 if r.observation_id not in response.evidence_ids)[:report_budget]


def route_decision(memory: Memory, minimum_open: float = 0.7) -> dict:
    """Record a toy threshold policy and the exact supporting context."""
    probability(minimum_open)
    return {"action": "cross" if memory.open_probability >= minimum_open else "detour",
            "minimum_open": minimum_open, "dependencies": [memory.claim_id],
            "context": memory.context()}


def initial_memories() -> tuple[Memory, Memory, Memory]:
    scope = Scope("RiverBridge", "northbound", "ambulance",
                  "2026-09-12T09:00:00-04:00", "2026-09-12T10:00:00-04:00")
    source = Memory("source", "bridge-open", scope, 0.8, scope.start,
                    "source:v0", ("inspection-initial",), (
                        ("RiverBridge", "type", "Bridge"),
                        ("RiverBridge", "connectsTo", "NorthRoad"),
                        ("bridge-open", "about", "RiverBridge"),
                        ("bridge-open", "vehicle", "ambulance"),
                        ("North", "inspects", "RiverBridge"),
                        ("Dispatch", "routeUses", "RiverBridge"),
                        ("Dispatch", "destination", "Hospital")))
    return source, seed(source, "north", "North"), seed(source, "dispatch", "Dispatch")


def walkthrough() -> list[tuple[str, tuple[Memory, Memory, Memory]]]:
    source, north, dispatch = initial_memories()
    report = Report("crew-closure-001", "north-crew", source.scope,
                    "2026-09-12T09:05:00-04:00", 0.1, 0.9)
    stages = [("initialization", (source, north, dispatch))]
    north = north.receive(report, "2026-09-12T09:06:00-04:00")
    stages.append(("local report", (source, north, dispatch)))
    source = source.receive(north.reports[0], "2026-09-12T09:07:00-04:00")
    stages.append(("manager review", (source, north, dispatch)))
    response = probe(dispatch)
    stages.append(("probe dispatch", (source, north, dispatch)))
    for update in targeted_cast(source, response, report_budget=1):
        dispatch = dispatch.receive(update, "2026-09-12T09:08:00-04:00")
        north = north.receive(update, "2026-09-12T09:08:00-04:00")
    stages.append(("targeted cast", (source, north, dispatch)))
    dispatch = dispatch.receive(north.reports[0], "2026-09-12T09:09:00-04:00")
    stages.append(("duplicate peer report", (source, north, dispatch)))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save each stage and decision as JSON")
    args = parser.parse_args()
    for index, (name, memories) in enumerate(walkthrough()):
        values = " ".join(f"{m.holder}={m.open_probability:.3f}" for m in memories)
        decision = route_decision(memories[2])
        print(f"{index}. {name}: {values} dispatch={decision['action']}")
        if args.output:
            directory = args.output / f"{index:02d}"
            for memory in memories:
                memory.save(directory / f"{memory.holder}.json")
            (directory / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__":
    main()
