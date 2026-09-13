"""Check the new article's arithmetic, provenance grouping, and plan repair."""

import pytest

from scima.belief_decisions import (
    COPIES, DEPENDENCIES, PLANS, PendingPlan, affected_plans, assessment, source_groups,
)


def test_transition_and_publication_examples():
    assert assessment(0)["decision"] == "publish"
    base = assessment(30)
    assert base["probability"] == pytest.approx(0.9 * 0.75 + 0.1 * 0.25)
    assert base["threshold"] == pytest.approx(0.8)
    assert base["publish_loss"] == pytest.approx(24)
    assert base["withhold_loss"] == pytest.approx(14)
    assert base["decision"] == "withhold"
    assert assessment(30, True)["probability"] == pytest.approx(0.6)


def test_age_changes_probability_without_changing_source():
    probabilities = [assessment(age)["probability"] for age in range(61)]
    assert all(a > b for a, b in zip(probabilities, probabilities[1:]))
    assert all(0.5 < p <= 0.9 for p in probabilities)
    assert len(source_groups(COPIES)) == 1
    assert len(source_groups({**COPIES, "crew-message": "crew-notice"})) == 2


@pytest.mark.parametrize("operator", ["update", "revision"])
def test_both_operators_recheck_dependent_pending_plans(operator):
    assert affected_plans("bridge-open", operator, PLANS, DEPENDENCIES) == [
        "ambulance across River Bridge"]
    assert affected_plans("unrelated", operator, PLANS, DEPENDENCIES) == []


def test_dependency_chain_and_cycle_terminate():
    graph = {"route": frozenset({"clear"}), "clear": frozenset({"bridge", "route"})}
    plans = [PendingPlan("ambulance", frozenset({"route"}))]
    assert affected_plans("bridge", "update", plans, graph) == ["ambulance"]


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf")])
def test_invalid_age_is_rejected(age):
    with pytest.raises(ValueError):
        assessment(age)
