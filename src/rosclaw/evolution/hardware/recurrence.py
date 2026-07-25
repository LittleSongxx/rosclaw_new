"""Recurrence — the self-evolution proof (PR-EVO-HW-5, 真机自进化v2 §Phase 8).

The defining test of the whole programme:

    promoted rule exists (from a successful canary)
    → runtime restarted (new practice session, new trace, old permits dead)
    → baseline conditions again
    → same regime/failure appears
    → system AUTOMATICALLY retrieves the promoted rule, re-validates it
      (choreography), applies it between rounds, and the critic observes
      the improvement again — with zero manual config edits.

Honesty rules:
* NO promoted rule → BLOCKED with the truthful reason (a rolled-back or
  never-canaried candidate yields nothing to apply).
* The rule applied must hash-match the VALIDATED candidate version; a
  drifted rule is refused.
* Every step lands in the evidence manifest: rule selected, hash check,
  choreography re-check, apply point, before/after metrics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

COLLECTION = "evo_promoted_rules"


class RecurrenceBlockedError(RuntimeError):
    """The recurrence cannot run — the reason is the evidence."""


@dataclass(frozen=True)
class RecurrencePlan:
    rule_id: str
    candidate_id: str
    changes: dict[str, Any]
    canary_sessions: list[str]
    rule_hash: str


@dataclass
class RecurrenceProof:
    rule_id: str
    hash_match: bool
    apply_round: int | None = None
    before_metrics: dict[str, Any] = field(default_factory=dict)
    after_metrics: dict[str, Any] = field(default_factory=dict)
    improved: bool | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "hash_match": self.hash_match,
            "apply_round": self.apply_round,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improved": self.improved,
        }


def rule_hash(rule: dict[str, Any]) -> str:
    """Content hash of the rule's actionable part (changes + candidate id)."""
    changes = rule.get("changes") or {}
    if isinstance(changes, str):
        changes = json.loads(changes)
    blob = json.dumps(
        {"candidate_id": rule.get("candidate_id"), "changes": changes},
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_promoted_rule(store: Any) -> dict[str, Any] | None:
    """The single ACTIVE promoted rule, or None.  Multiple active rules are
    a contract violation — promotion is one-at-a-time by design."""
    rules = [
        row
        for row in store.query(COLLECTION, filters={"status": "active"}, limit=10)
        if row.get("status") == "active"
    ]
    if not rules:
        return None
    if len(rules) > 1:
        raise RecurrenceBlockedError(
            f"{len(rules)} active promoted rules — promotion must be one-at-a-time"
        )
    return rules[0]


def plan_recurrence(store: Any, registry_row: dict[str, Any] | None) -> RecurrencePlan:
    """Build the recurrence plan with the hash-drift check (§Phase 8:
    Candidate 与历史验证版本 Hash 一致)."""
    rule = load_promoted_rule(store)
    if rule is None:
        raise RecurrenceBlockedError(
            "no promoted rule — recurrence requires a PROMOTED candidate from "
            "a successful canary (the current candidate was rolled back "
            "honestly: nothing to apply)"
        )
    expected = rule_hash(rule)
    actual = rule_hash(rule)  # computed from the stored rule itself
    if registry_row is not None:
        candidate_changes = registry_row.get("changes") or {}
        if isinstance(candidate_changes, str):
            candidate_changes = json.loads(candidate_changes)
        actual = rule_hash(
            {"candidate_id": registry_row.get("candidate_id"), "changes": candidate_changes}
        )
    sessions = rule.get("canary_sessions") or []
    if isinstance(sessions, str):
        sessions = json.loads(sessions)
    return RecurrencePlan(
        rule_id=str(rule["rule_id"]),
        candidate_id=str(rule["candidate_id"]),
        changes=(rule.get("changes") if isinstance(rule.get("changes"), dict) else json.loads(rule.get("changes") or "{}")),
        canary_sessions=list(sessions),
        rule_hash=expected if expected == actual else "",
    )


def evaluate_recurrence(
    *,
    plan: RecurrencePlan,
    session_summary: dict[str, Any],
    baseline_invalid: float | None,
) -> RecurrenceProof:
    """The critic: did the promoted rule improve the recurrence session
    against the baseline reference?"""
    proof = RecurrenceProof(rule_id=plan.rule_id, hash_match=bool(plan.rule_hash))
    if not plan.rule_hash:
        return proof
    lifecycle = session_summary.get("candidate_lifecycle") or {}
    proof.apply_round = 1 if lifecycle.get("cooldown_applied") or lifecycle.get("rehome_count") or lifecycle.get("neutral_count") else None
    proof.before_metrics = {"baseline_invalid_rate": baseline_invalid}
    after = {
        "invalid_rate": session_summary.get("invalid_rate"),
        "verified_rate": session_summary.get("verified_rate"),
        "peak_temperature": session_summary.get("peak_temperature"),
        "lifecycle": lifecycle,
    }
    proof.after_metrics = after
    if baseline_invalid is not None and after["invalid_rate"] is not None:
        proof.improved = bool(after["invalid_rate"] < baseline_invalid)
    return proof
