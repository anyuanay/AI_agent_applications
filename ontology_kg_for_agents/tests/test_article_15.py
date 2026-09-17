"""Behavioral checks for the Article 15 offline teaching companion."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from scima.ontology_nn_coevolution import ClosureRate, Cycle, Review, StormEpisode, walkthrough
from scima.source_actor_memory import initial_memories


def test_beta_posterior_preserves_counts_and_deduplicates():
    rate = ClosureRate()
    assert rate.mean == .2
    for i in range(10):
        episode = StormEpisode(str(i), rate.regime, i < 7)
        rate = rate.observe(episode)
        assert rate.observe(episode) is rate
    assert rate.posterior == (9, 11)
    assert rate.mean == .45
    with pytest.raises(ValueError, match='conflicting'):
        rate.observe(StormEpisode('0', rate.regime, False))
    with pytest.raises(ValueError, match='regime'):
        rate.observe(StormEpisode('new', 'another-storm', True))


@pytest.mark.parametrize('a,b', [(0, 2), (-1, 2), (2, float('nan')), (float('inf'), 2)])
def test_invalid_prior_rejected(a, b):
    with pytest.raises(ValueError):
        ClosureRate(a, b)


def test_rate_requires_boolean_labels_and_unique_observations():
    with pytest.raises(ValueError):
        StormEpisode('x', 'storm-demo', 'closed')
    episode = StormEpisode('x', 'storm-demo', True)
    with pytest.raises(ValueError):
        ClosureRate(episodes=(episode, episode))


def test_memory_change_does_not_train_or_accept_a_proposal():
    stages = walkthrough()
    initial, updated = stages[0][1], stages[1][1]
    assert initial.memory.version == 0
    assert initial.memory.open_probability == .8
    assert updated.memory.version == 1
    # This companion uses the fixed-state Article 13 calculation, not Article 14's prediction.
    assert updated.memory.open_probability == pytest.approx(4 / 13)
    assert updated.base_model == 'W0' and updated.candidate is None
    candidate = stages[3][1]
    assert candidate.candidate.weights_trained is False
    assert candidate.candidate.status == 'tests_pending'
    pending = stages[4][1]
    assert pending.ontology_version == 0 and pending.accepted_changes == ()
    assert pending.proposal.status == 'pending'
    assert pending.memory == updated.memory


def test_duplicate_report_cannot_change_training_or_candidate():
    pending = walkthrough()[4][1]
    report = pending.memory.reports[0]
    assert pending.receive(report, pending.memory.assessed_at) is pending
    assert pending.training.evidence_ids == ('R17', 'inspection-initial')
    with pytest.raises(ValueError, match='conflicting'):
        pending.receive(replace(report, given_open=.2), pending.memory.assessed_at)


def test_new_evidence_makes_existing_manifest_stale():
    prepared = walkthrough()[2][1]
    report = replace(prepared.memory.reports[0], observation_id='another-observation')
    changed = prepared.receive(report, prepared.memory.assessed_at)
    with pytest.raises(ValueError, match='current training'):
        changed.record_candidate('W1')
    pending_candidate = walkthrough()[3][1].receive(report, prepared.memory.assessed_at)
    with pytest.raises(ValueError, match='outdated graph'):
        pending_candidate.propose('new-proposal', 'hypothesis')
    fresh = changed.prepare('T-fresh').record_candidate('W-fresh')
    assert 'another-observation' in fresh.training.evidence_ids


def test_training_and_proposals_require_prior_stages():
    cycle = Cycle(initial_memories()[1])
    with pytest.raises(ValueError):
        cycle.prepare('T1')
    with pytest.raises(ValueError):
        cycle.record_candidate('W1')
    with pytest.raises(ValueError):
        cycle.propose('P1', 'hypothesis')
    with pytest.raises(ValueError):
        walkthrough()[3][1].record_candidate('W2')


def review_for(cycle, **kwargs):
    return Review(**{'proposal_version': cycle.proposal.version,
                     'observation_id': 'external-review', 'origins': ('independent-inspection',),
                     'scope': cycle.memory.scope, 'reviewer': 'reviewer', 'supports': True,
                     **kwargs})


@pytest.mark.parametrize('change', [
    {'observation_id': 'R17'},
    {'origins': ('R17',)},
    {'origins': ('inspection-initial',)},
    {'proposal_version': 'wrong'},
])
def test_copies_and_wrong_proposal_cannot_supply_review(change):
    pending = walkthrough()[4][1]
    with pytest.raises(ValueError):
        pending.review(review_for(pending, **change))
    assert pending.ontology_version == 0


def test_review_scope_and_rejected_findings_do_not_change_ontology():
    pending = walkthrough()[4][1]
    with pytest.raises(ValueError, match='scope'):
        pending.review(review_for(pending, scope=replace(pending.memory.scope, vehicle='truck')))
    rejected = pending.review(review_for(pending, supports=False))
    assert rejected.proposal.status == 'rejected'
    assert rejected.ontology_version == 0 and rejected.accepted_changes == ()
    with pytest.raises(ValueError, match='pending'):
        rejected.review(review_for(pending))


def test_reviewed_change_drives_next_cycle_with_original_evidence():
    pending = walkthrough()[4][1]
    accepted = pending.review(review_for(pending))
    assert accepted.ontology_version == 1
    assert accepted.memory == pending.memory
    assert accepted.proposal.supporting_origins == ('independent-inspection',)
    next_cycle = accepted.prepare('T2')
    assert next_cycle.candidate is None
    assert next_cycle.training.ontology_version == 1
    assert {'independent-inspection', 'external-review'} <= set(next_cycle.training.evidence_ids)
    assert next_cycle.training.examples[-1].target == accepted.proposal.statement
    proposal = next_cycle.record_candidate('W2').propose('P2', 'next change')
    with pytest.raises(ValueError):
        proposal.review(review_for(proposal, observation_id='a-copy', origins=('independent-inspection',)))


def test_cli_saves_reloadable_records_without_model_training(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scima.ontology_nn_coevolution',
                             '--output', str(tmp_path)], capture_output=True, text=True, check=True)
    trace = json.loads((tmp_path / 'trace.json').read_text())
    assert trace['closure_rate']['posterior'] == [9, 11]
    assert trace['closure_rate']['mean'] == .45
    assert len(trace['stages']) == 7
    assert len(list(tmp_path.glob('[0-9][0-9].json'))) == 7
    for stage in trace['stages']:
        candidate = stage['cycle']['candidate']
        assert candidate is None or candidate['weights_trained'] is False
    assert 'candidate record, no training' in result.stdout
