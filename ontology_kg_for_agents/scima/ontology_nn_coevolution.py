"""Article 15: an offline, reviewed ontology/model update cycle.

No neural model is trained or called. Adapter records describe candidates only.
Review inputs are caller-supplied attestations, not authentication or independent
verification. Distinct observation IDs alone do not prove independence.
"""

from dataclasses import asdict, dataclass, replace
import argparse
import json
import math
from pathlib import Path

from .source_actor_memory import Memory, Report, Scope, initial_memories


@dataclass(frozen=True)
class StormEpisode:
    observation_id: str
    regime: str
    ends_closed: bool

    def __post_init__(self):
        if not self.observation_id or not self.regime:
            raise ValueError("observation ID and regime are required")
        if type(self.ends_closed) is not bool:
            raise ValueError("ends_closed must be a boolean")


@dataclass(frozen=True)
class ClosureRate:
    """One-minute endpoint transitions from open in a declared storm regime.

    Episodes require independent, accurately labeled observations. The prior
    parameters are support counts, not observations. No state filtering or LLM
    training occurs here.
    """
    alpha: float = 2.0
    beta: float = 8.0
    regime: str = "storm-demo"
    episodes: tuple[StormEpisode, ...] = ()

    def __post_init__(self):
        if not self.regime or any(not math.isfinite(x) or x <= 0
                                  for x in (self.alpha, self.beta)):
            raise ValueError("positive finite prior parameters and regime required")
        ids = [e.observation_id for e in self.episodes]
        if len(ids) != len(set(ids)) or any(e.regime != self.regime for e in self.episodes):
            raise ValueError("episodes need unique IDs and matching regimes")

    @property
    def posterior(self) -> tuple[float, float]:
        closed = sum(e.ends_closed for e in self.episodes)
        return self.alpha + closed, self.beta + len(self.episodes) - closed

    @property
    def mean(self) -> float:
        a, b = self.posterior
        return a / (a + b)

    def observe(self, episode: StormEpisode) -> 'ClosureRate':
        if episode.regime != self.regime:
            raise ValueError("regime mismatch requires model review")
        for known in self.episodes:
            if known.observation_id == episode.observation_id:
                if known != episode:
                    raise ValueError("observation ID has conflicting contents")
                return self
        return replace(self, episodes=self.episodes + (episode,))


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    target: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSet:
    version: str
    memory_version: int
    ontology_version: int
    evidence_ids: tuple[str, ...]
    examples: tuple[TrainingExample, ...]


@dataclass(frozen=True)
class Candidate:
    version: str
    base_model: str
    training_set: str
    status: str = "tests_pending"
    weights_trained: bool = False


@dataclass(frozen=True)
class Proposal:
    version: str
    statement: str
    scope: Scope
    model_version: str
    derived_from: tuple[str, ...]
    status: str = "pending"
    supporting_observation: str | None = None
    supporting_origins: tuple[str, ...] = ()
    reviewer: str | None = None


@dataclass(frozen=True)
class Review:
    """A caller's externally checked observation, scoped to one proposal.

    origins must identify original observations, including those underlying
    copied reports. This record cannot authenticate those assertions.
    """
    proposal_version: str
    observation_id: str
    origins: tuple[str, ...]
    scope: Scope
    reviewer: str
    supports: bool

    def __post_init__(self):
        if not all((self.proposal_version, self.observation_id, self.reviewer)) or not self.origins:
            raise ValueError("review needs proposal, observation, origins, and reviewer")
        if any(not origin for origin in self.origins) or type(self.supports) is not bool:
            raise ValueError("review needs valid origins and a boolean finding")


@dataclass(frozen=True)
class Cycle:
    memory: Memory
    base_model: str = "W0"
    ontology_version: int = 0
    accepted_changes: tuple[Proposal, ...] = ()
    training: TrainingSet | None = None
    candidate: Candidate | None = None
    proposal: Proposal | None = None

    def receive(self, report: Report, assessed_at: str) -> 'Cycle':
        memory = self.memory.receive(report, assessed_at)
        if memory is self.memory:
            return self
        # Preserve artifacts with their pinned versions. prepare() replaces them.
        return replace(self, memory=memory)

    def prepare(self, version: str) -> 'Cycle':
        """Create deterministic, checked-example fixtures from accepted records.

        This is a training manifest, not a tokenizer, training run, or evaluator.
        The fixed-state Bayesian memory model is inherited from Article 13.
        """
        if not version or not self.memory.reports:
            raise ValueError("training needs a version and accepted report evidence")
        if self.training and version == self.training.version:
            raise ValueError("use a new training version")
        evidence = tuple(sorted(self.memory.evidence_ids | {
            origin for p in self.accepted_changes
            for origin in (p.supporting_observation, *p.supporting_origins)
            if origin is not None}))
        scope = self.memory.scope
        examples = (
            TrainingExample(
                f"Does the report establish closure of {scope.entity}?",
                "It supplies evidence, not certainty. Retrieve the scoped assessment.",
                evidence),
            TrainingExample(
                f"Does a general-traffic restriction establish access for {scope.vehicle}?",
                "Request vehicle scope. Missing information does not establish access.",
                evidence),
        ) + tuple(TrainingExample(
            f"What reviewed distinction concerns {scope.entity}?", p.statement,
            tuple(sorted(set(p.derived_from) | set(p.supporting_origins) | {p.supporting_observation})))
            for p in self.accepted_changes)
        manifest = TrainingSet(version, self.memory.version, self.ontology_version,
                               evidence, examples)
        return replace(self, training=manifest, candidate=None, proposal=None)

    def record_candidate(self, version: str) -> 'Cycle':
        if not version or version == self.base_model:
            raise ValueError("candidate needs a distinct version")
        if not self.training or (self.training.memory_version, self.training.ontology_version) != (
                self.memory.version, self.ontology_version):
            raise ValueError("a current training manifest is required")
        if self.candidate is not None:
            raise ValueError("candidate already recorded for this manifest")
        return replace(self, candidate=Candidate(version, self.base_model, self.training.version))

    def propose(self, version: str, statement: str) -> 'Cycle':
        if not version or not statement or not self.candidate or not self.training:
            raise ValueError("proposal needs a candidate, version, and statement")
        if self.proposal is not None:
            raise ValueError("resolve the existing proposal before another cycle")
        if (self.training.memory_version, self.training.ontology_version) != (
                self.memory.version, self.ontology_version):
            raise ValueError("candidate targets an outdated graph")
        return replace(self, proposal=Proposal(version, statement, self.memory.scope,
                       self.candidate.version, self.training.evidence_ids))

    def review(self, evidence: Review) -> 'Cycle':
        p = self.proposal
        if p is None or p.status != "pending" or evidence.proposal_version != p.version:
            raise ValueError("review must target the pending proposal")
        if evidence.scope != p.scope:
            raise ValueError("review scope mismatch")
        known = set(p.derived_from) | self.memory.evidence_ids
        for accepted in self.accepted_changes:
            known.add(accepted.supporting_observation)
            known.update(accepted.supporting_origins)
        if evidence.observation_id in known or known.intersection(evidence.origins):
            raise ValueError("copied or training evidence is not an independent review")
        reviewed = replace(p, status="accepted" if evidence.supports else "rejected",
                           supporting_observation=evidence.observation_id,
                           supporting_origins=evidence.origins, reviewer=evidence.reviewer)
        accepted = self.accepted_changes + (reviewed,) if evidence.supports else self.accepted_changes
        return replace(self, proposal=reviewed, accepted_changes=accepted,
                       ontology_version=self.ontology_version + int(evidence.supports))


def walkthrough() -> list[tuple[str, Cycle]]:
    """Fixed-state bridge memory, followed by a hypothetical reviewed change."""
    _, north, _ = initial_memories()
    cycle = Cycle(north)
    report = Report("R17", "north-crew", north.scope,
                    "2026-09-12T09:05:00-04:00", .1, .9)
    stages = [("inherited memory", cycle)]
    cycle = cycle.receive(report, "2026-09-12T09:06:00-04:00")
    stages.append(("accepted report", cycle))
    cycle = cycle.prepare("T1")
    stages.append(("checked examples", cycle))
    cycle = cycle.record_candidate("W1-candidate")
    stages.append(("candidate record, no training", cycle))
    cycle = cycle.propose("P1", "Vehicle-specific restrictions need an explicit vehicle scope.")
    stages.append(("pending model proposal", cycle))
    cycle = cycle.review(Review("P1", "inspection-independent", ("inspection-independent",),
                               north.scope, "teaching-reviewer", True))
    stages.append(("reviewed ontology change", cycle))
    cycle = cycle.prepare("T2")
    stages.append(("next training manifest", cycle))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON snapshots to this directory")
    args = parser.parse_args()
    rate = ClosureRate()
    for i in range(10):
        rate = rate.observe(StormEpisode(f"episode-{i + 1}", rate.regime, i < 7))
    result = {"implementation": "offline teaching records, no model training",
              "closure_rate": {"prior": [2, 8], "posterior": rate.posterior, "mean": rate.mean},
              "stages": [{"label": label, "cycle": asdict(cycle)} for label, cycle in walkthrough()]}
    print(f"Storm transition: Beta{rate.posterior}, mean={rate.mean:.2f}")
    for stage in result["stages"]:
        c = stage['cycle']
        print(f"{stage['label']}: memory={c['memory']['version']} ontology={c['ontology_version']} "
              f"model={c['candidate']['version'] if c['candidate'] else c['base_model']}")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
        for i, stage in enumerate(result["stages"]):
            (args.output / f"{i:02d}.json").write_text(json.dumps(stage, indent=2) + "\n")


if __name__ == "__main__":
    main()
