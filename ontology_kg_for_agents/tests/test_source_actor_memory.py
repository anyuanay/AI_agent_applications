"""Checks for the revised Article 13 memory example."""
from dataclasses import replace
import math
import pytest
from scima.source_actor_memory import (
    Memory, Report, bayesian_open, initial_memories, probe, route_decision,
    seed, targeted_cast, walkthrough,
)


@pytest.mark.parametrize('prior,lo,lc', [(-.1, .1, .9), (1.1, .1, .9),
                                       (.8, math.nan, .9), (.8, .1, math.inf),
                                       (.8, 0, 0), (1, 0, .9)])
def test_invalid_bayes_inputs(prior, lo, lc):
    with pytest.raises(ValueError):
        bayesian_open(prior, lo, lc)


def test_article_six_stages_and_decisions():
    stages = walkthrough()
    p = .08 / .26
    expected = [(.8, .8, .8), (.8, p, .8), (p, p, .8),
                (p, p, .8), (p, p, p), (p, p, p)]
    assert len(stages) == 6
    for (_, memories), values in zip(stages, expected):
        assert tuple(m.open_probability for m in memories) == pytest.approx(values)
    assert stages[2][1] == stages[3][1]  # probing adds no observation
    assert stages[4][1] == stages[5][1]  # peer duplicate has no effect
    assert route_decision(stages[0][1][2])['action'] == 'cross'
    decision = route_decision(stages[-1][1][2])
    assert decision['action'] == 'detour'
    assert decision['dependencies'] == ['bridge-open']
    assert decision['context']['memory_version'] == 1
    assert decision['context']['evidence_ids'] == ['crew-closure-001', 'inspection-initial']


def test_seed_preserves_origin_scope_evidence_and_role_links():
    source, north, dispatch = initial_memories()
    assert north.source_snapshot == dispatch.source_snapshot == 'source:v0'
    assert source.scope == north.scope == dispatch.scope
    assert source.evidence_ids == north.evidence_ids == dispatch.evidence_ids
    assert ('North', 'inspects', 'RiverBridge') in north.triples
    assert ('North', 'inspects', 'RiverBridge') not in dispatch.triples
    assert ('Dispatch', 'destination', 'Hospital') in dispatch.triples
    assert ('RiverBridge', 'connectsTo', 'NorthRoad') in dispatch.triples
    updated = walkthrough()[2][1][0]
    late = seed(updated, 'late', 'Dispatch')
    assert late.source_snapshot == 'source:v1'
    assert late.receive(updated.reports[0], updated.assessed_at) is late


def test_roundtrip_and_duplicate_survive_restart(tmp_path):
    memory = walkthrough()[-1][1][2]
    path = tmp_path / 'memory.json'
    memory.save(path)
    loaded = Memory.load(path)
    assert loaded == memory
    assert loaded.receive(memory.reports[0], memory.assessed_at) is loaded
    assert loaded.history[0].prior_open == .8
    assert loaded.history[0].validation == 'accepted'


def test_review_rejects_wrong_scope_time_and_identity_collision():
    source = walkthrough()[2][1][0]
    report = source.reports[0]
    with pytest.raises(ValueError, match='scope mismatch'):
        source.receive(replace(report, scope=replace(report.scope, entity='OtherBridge')),
                       source.assessed_at)
    with pytest.raises(ValueError, match='conflicting contents'):
        source.receive(replace(report, given_open=.2), source.assessed_at)
    with pytest.raises(ValueError, match='precedes'):
        initial_memories()[0].receive(report, report.scope.start)
    with pytest.raises(ValueError, match='outside'):
        replace(report, observed_at=report.scope.end)
    with pytest.raises(ValueError, match='timezone'):
        replace(report, observed_at='2026-09-12T09:05:00')
    assert source.version == 1


def test_probe_budget_and_reconciliation_preserve_local_evidence():
    source = walkthrough()[2][1][0]
    dispatch = initial_memories()[2]
    local = Report('local-002', 'dispatch-sensor', dispatch.scope,
                   '2026-09-12T09:06:00-04:00', .7, .3)
    dispatch = dispatch.receive(local, '2026-09-12T09:07:00-04:00')
    response = probe(dispatch)
    assert targeted_cast(source, response, 0) == ()
    reports = targeted_cast(source, response, 1)
    assert reports == source.reports
    merged = dispatch.receive(reports[0], '2026-09-12T09:08:00-04:00')
    assert merged.evidence_ids == {'inspection-initial', 'local-002', 'crew-closure-001'}
    assert merged.open_probability == pytest.approx(bayesian_open(.8, .07, .27))
    assert merged.open_probability != pytest.approx(source.open_probability)
    assert targeted_cast(source, probe(merged), 1) == ()
    with pytest.raises(ValueError):
        targeted_cast(source, response, -1)
    with pytest.raises(ValueError, match='different claim'):
        targeted_cast(source, replace(response, claim_id='other'), 1)
