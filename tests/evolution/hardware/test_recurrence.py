"""Recurrence plan/proof tests (PR-EVO-HW-5, §Phase 8)."""

from __future__ import annotations

import pytest

from rosclaw.evolution.hardware.recurrence import (
    RecurrenceBlockedError,
    evaluate_recurrence,
    load_promoted_rule,
    plan_recurrence,
    rule_hash,
)
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore

COLLECTION = "evo_promoted_rules"


def _store_with_rule(changes=None, status="active", sessions=None):
    store = InMemoryKnowledgeStore()
    store.connect()
    store.insert(
        COLLECTION,
        {
            "id": "rule_cand_x",
            "rule_id": "rule_cand_x",
            "candidate_id": "cand_x",
            "changes": changes if changes is not None else {"inter_round_cooldown_sec": 2.0},
            "scope": "operator_approved_recurrence",
            "status": status,
            "canary_sessions": sessions or ["prac_1", "prac_2"],
        },
    )
    return store


def test_no_promoted_rule_is_an_honest_block() -> None:
    store = InMemoryKnowledgeStore()
    store.connect()
    with pytest.raises(RecurrenceBlockedError, match="no promoted rule"):
        plan_recurrence(store, None)


def test_inactive_rule_does_not_count() -> None:
    store = _store_with_rule(status="superseded")
    with pytest.raises(RecurrenceBlockedError):
        plan_recurrence(store, None)


def test_plan_carries_hash_and_sessions() -> None:
    store = _store_with_rule()
    plan = plan_recurrence(store, None)
    assert plan.rule_id == "rule_cand_x"
    assert plan.changes == {"inter_round_cooldown_sec": 2.0}
    assert plan.canary_sessions == ["prac_1", "prac_2"]
    assert plan.rule_hash


def test_registry_drift_empties_hash() -> None:
    store = _store_with_rule()
    drifted_row = {"candidate_id": "cand_x", "changes": {"inter_round_cooldown_sec": 4.0}}
    plan = plan_recurrence(store, drifted_row)
    assert plan.rule_hash == ""  # drift detected → application refused
    proof = evaluate_recurrence(plan=plan, session_summary={}, baseline_invalid=0.2)
    assert proof.hash_match is False
    assert proof.improved is None


def test_multiple_active_rules_is_a_violation() -> None:
    store = _store_with_rule()
    store.insert(
        COLLECTION,
        {
            "id": "rule_cand_y", "rule_id": "rule_cand_y", "candidate_id": "cand_y",
            "changes": {}, "status": "active",
        },
    )
    with pytest.raises(RecurrenceBlockedError, match="one-at-a-time"):
        load_promoted_rule(store)


def test_evaluate_recurrence_judges_improvement() -> None:
    store = _store_with_rule()
    plan = plan_recurrence(store, None)
    summary = {
        "invalid_rate": 0.10,
        "verified_rate": 0.90,
        "peak_temperature": 47,
        "candidate_lifecycle": {"cooldown_applied": True, "cooldown_total_s": 78.0},
    }
    proof = evaluate_recurrence(plan=plan, session_summary=summary, baseline_invalid=0.20)
    assert proof.hash_match is True
    assert proof.improved is True
    assert proof.apply_round == 1
    assert proof.after_metrics["lifecycle"]["cooldown_total_s"] == 78.0
    worse = evaluate_recurrence(
        plan=plan,
        session_summary={**summary, "invalid_rate": 0.30},
        baseline_invalid=0.20,
    )
    assert worse.improved is False


def test_rule_hash_stable() -> None:
    rule = {"candidate_id": "cand_x", "changes": {"inter_round_cooldown_sec": 2.0}}
    assert rule_hash(rule) == rule_hash(dict(rule))
    assert rule_hash(rule) != rule_hash({"candidate_id": "cand_x", "changes": {"inter_round_cooldown_sec": 4.0}})
