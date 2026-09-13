"""Supplemental River Bridge case from the earlier Article 13 treatment. Run with python -m scima.belief_decisions."""

from dataclasses import dataclass
import math

from scima.agent_beliefs import aged_belief, publication_floor


@dataclass(frozen=True)
class PendingPlan:
    name: str
    dependencies: frozenset[str]


def assessment(age_minutes: float = 30, disruption: bool = False) -> dict:
    """Use a symmetric two-state model and illustrative common-scale losses."""
    if not math.isfinite(age_minutes) or age_minutes < 0:
        raise ValueError("age must be finite and nonnegative")
    half_life = 15 if disruption else 30
    probability = aged_belief(0.9, age_minutes * 60, half_life * 60)
    publish_loss = (1 - probability) * 80
    withhold_loss = probability * 20
    return {
        "probability": probability,
        "threshold": publication_floor(80, 20),
        "publish_loss": publish_loss,
        "withhold_loss": withhold_loss,
        "decision": "publish" if publish_loss < withhold_loss else "withhold",
    }


def affected_plans(changed_claim: str, operator: str, plans,
                   derived_from: dict[str, frozenset[str]]) -> list[str]:
    """Recheck pending plans under either operator, following derived claims.

    This does not retract historical decisions or judge the new evidence.
    Dependency edges run from a derived claim to its immediate sources.
    """
    if operator not in {"update", "revision"}:
        raise ValueError("operator must be update or revision")
    affected = {changed_claim}
    while True:
        additions = {claim for claim, sources in derived_from.items()
                     if sources & affected} - affected
        if not additions:
            break
        affected.update(additions)
    return [plan.name for plan in plans if plan.dependencies & affected]


def source_groups(copies: dict[str, str]) -> dict[str, list[str]]:
    """Group message IDs by stable observation ID. Do not fuse probabilities."""
    groups: dict[str, list[str]] = {}
    for message, observation in copies.items():
        groups.setdefault(observation, []).append(message)
    return groups


PLANS = (
    PendingPlan("ambulance across River Bridge", frozenset({"route-clear"})),
    PendingPlan("shuttle on Hill Road", frozenset({"hill-road-open"})),
)
DEPENDENCIES = {"route-clear": frozenset({"bridge-open"})}
COPIES = {"north-message": "inspection-report",
          "dispatcher-copy": "inspection-report",
          "south-copy": "inspection-report"}


def main():
    for name, age, disruption in (("Fresh", 0, False), ("Base", 30, False),
                                  ("Disruption", 30, True)):
        result = assessment(age, disruption)
        print(f"{name}: p={result['probability']:.3f}, "
              f"publish loss={result['publish_loss']:.2f}, "
              f"withhold loss={result['withhold_loss']:.2f}, {result['decision']}")
    print(f"Observation groups: {len(source_groups(COPIES))}")
    for operator in ("update", "revision"):
        print(f"{operator}: recheck "
              f"{affected_plans('bridge-open', operator, PLANS, DEPENDENCIES)}")


if __name__ == "__main__":
    main()
