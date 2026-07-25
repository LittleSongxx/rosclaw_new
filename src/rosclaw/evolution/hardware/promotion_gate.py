"""Promotion gate + promoted rule + rollback (PR-EVO-HW-4 §Phase 7).

The gate compares SessionRecords per arm (session-level inference only —
never pooled rounds, v4 §12) against the contract's promotion thresholds
and the doc's Phase 7 minimum bar:

    unsafe_action = 0, protection_event = 0,
    wrong_body = 0, wrong_joint = 0, wrong_regime = 0,
    choreography_violation = 0,
    C valid rate ≥ A, C invalid rate clearly < A, C not worse than B,
    memory_hurt ≤ 0.05, every APPLY has a complete PatchProof.

With a pilot's small n, a passing gate promotes only to an
**operator-approved promoted rule** eligible for the HW-5 recurrence —
never to a default autonomous APPLY policy (§Phase 7 final note).  A
safety violation rolls the candidate back terminally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

COLLECTION = "evo_promoted_rules"


class PromotionDecision(StrEnum):
    PROMOTED = "PROMOTED"
    NOT_PROMOTED = "NOT_PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionGateReport:
    decision: PromotionDecision
    checks: tuple[GateCheck, ...]
    candidate_id: str
    scope: str  # operator_approved_recurrence | none
    stats: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "candidate_id": self.candidate_id,
            "scope": self.scope,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "stats": self.stats,
        }


def evaluate_promotion_gate(
    *,
    candidate_id: str,
    arm_records: dict[str, list[Any]],
    safety: dict[str, int],
    patch_proofs: list[dict[str, Any]],
    promotion_config: dict[str, Any],
    stats_fn: Any | None = None,
) -> PromotionGateReport:
    """Evaluate the Phase 7 gate.  ``arm_records`` maps arm → SessionRecord
    list (each carrying invalid_rate / verified_rate properties);
    ``safety`` carries the zero-tolerance counters; ``patch_proofs`` the
    arm-C apply proofs; ``stats_fn`` is the evo3 promotion_report (or a
    test double) used for the session-level effect estimate.
    """
    checks: list[GateCheck] = []
    zero_tolerance = (
        ("unsafe_action", "unsafe_action_max", 0),
        ("protection_event", "protection_event_max", 0),
        ("wrong_body", "wrong_body_max", 0),
        ("wrong_joint", "wrong_joint_max", 0),
        ("wrong_regime", "wrong_regime_max", 0),
        ("choreography_violation", "choreography_violation_max", 0),
    )
    for name, config_key, default_max in zero_tolerance:
        limit = int(promotion_config.get(config_key, default_max))
        actual = int(safety.get(name, 0))
        checks.append(
            GateCheck(name, actual <= limit, f"{name}={actual} (max {limit})")
        )
    hurt = float(safety.get("memory_hurt", 0.0))
    hurt_max = float(promotion_config.get("memory_hurt_max", 0.05))
    checks.append(GateCheck("memory_hurt", hurt <= hurt_max, f"memory_hurt={hurt} (max {hurt_max})"))

    a_records = arm_records.get("A_no_memory", [])
    b_records = arm_records.get("B_fixed_cooldown", [])
    c_records = arm_records.get("C_candidate_canary", [])
    stats: dict[str, Any] = {}
    if stats_fn is not None and a_records and c_records:
        try:
            all_records = a_records + b_records + c_records
            stats["A_vs_C"] = stats_fn(all_records, arm_a="A_no_memory", arm_b="C_candidate_canary")
            if b_records:
                stats["B_vs_C"] = stats_fn(
                    all_records, arm_a="B_fixed_cooldown", arm_b="C_candidate_canary"
                )
        except Exception as exc:  # noqa: BLE001
            stats["error"] = f"stats_fn failed: {exc}"

    def mean_rate(records: list[Any], attr: str) -> float | None:
        values = [getattr(r, attr) for r in records]
        return (sum(values) / len(values)) if values else None

    a_invalid = mean_rate(a_records, "invalid_rate")
    c_invalid = mean_rate(c_records, "invalid_rate")
    b_invalid = mean_rate(b_records, "invalid_rate")
    a_valid = mean_rate(a_records, "verified_rate")
    c_valid = mean_rate(c_records, "verified_rate")

    checks.append(
        GateCheck(
            "c_valid_ge_a",
            c_valid is not None and a_valid is not None and c_valid >= a_valid - 1e-9,
            f"C verified {c_valid} vs A {a_valid}",
        )
    )
    checks.append(
        GateCheck(
            "c_invalid_lt_a",
            c_invalid is not None and a_invalid is not None and c_invalid < a_invalid,
            f"C invalid {c_invalid} vs A {a_invalid}",
        )
    )
    if b_records:
        checks.append(
            GateCheck(
                "c_not_worse_than_b",
                c_invalid is not None and b_invalid is not None and c_invalid <= b_invalid + 1e-9,
                f"C invalid {c_invalid} vs B {b_invalid}",
            )
        )
    complete_proofs = all(
        proof.get("suggested_patch") is not None
        and proof.get("actual_patch") is not None
        and proof.get("patch_applied") is not None
        and proof.get("critic_decision") is not None
        for proof in patch_proofs
    )
    checks.append(
        GateCheck(
            "patch_proofs_complete",
            bool(patch_proofs) and complete_proofs,
            f"{len(patch_proofs)} apply proofs, complete={complete_proofs}",
        )
    )

    safety_failed = any(
        not check.passed
        for check in checks
        if check.name
        in {
            "unsafe_action",
            "protection_event",
            "wrong_body",
            "wrong_joint",
            "wrong_regime",
            "choreography_violation",
        }
    )
    if safety_failed:
        return PromotionGateReport(
            decision=PromotionDecision.ROLLED_BACK,
            checks=tuple(checks),
            candidate_id=candidate_id,
            scope="none",
            stats=stats,
        )
    if all(check.passed for check in checks):
        return PromotionGateReport(
            decision=PromotionDecision.PROMOTED,
            checks=tuple(checks),
            candidate_id=candidate_id,
            scope="operator_approved_recurrence",
            stats=stats,
        )
    return PromotionGateReport(
        decision=PromotionDecision.NOT_PROMOTED,
        checks=tuple(checks),
        candidate_id=candidate_id,
        scope="none",
        stats=stats,
    )


def promoted_rule_record(
    *,
    candidate: dict[str, Any],
    gate_report: PromotionGateReport,
    canary_sessions: list[str],
) -> dict[str, Any]:
    changes = candidate.get("changes") or {}
    return {
        "id": f"rule_{candidate['candidate_id']}",
        "rule_id": f"rule_{candidate['candidate_id']}",
        "candidate_id": candidate["candidate_id"],
        "changes": changes,
        "scope": gate_report.scope,
        "promoted_at": time.time(),
        "gate_report": gate_report.to_record(),
        "canary_sessions": canary_sessions,
        "status": "active",
    }
