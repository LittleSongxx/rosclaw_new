"""Evidence-bound self-evolution gate for the G1 Feedback Plane.

This module closes the development loop without granting an adaptation any
hardware authority.  It reloads the exact ILC artifact, rederives the active
controller snapshot, aggregates independent validation reports, and emits a
content-addressed candidate plus an explicit activation/rollback decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import (
    ControllerSnapshot,
    FallbackMode,
    canonical_hash,
)
from rosclaw.feedback.ilc import ILCFeedforward
from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.simforge.g1_ilc_validation import G1ILCFeedforwardCandidate

_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class FeedbackEvolutionDecision(StrEnum):
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


@dataclass(frozen=True)
class FeedbackEvidenceRef:
    kind: str
    schema_version: str
    content_hash: str
    path: str

    def bound_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
        }

    def to_dict(self) -> dict[str, str]:
        return {**self.bound_dict(), "path": self.path}


@dataclass(frozen=True)
class FeedbackControllerCandidate:
    """Composite L1 controller + C2 ILC candidate with rollback lineage."""

    body_hash: str
    kick_prior_hash: str
    regime_hash: str
    controller_hash: str
    controller_snapshot_hash: str
    loop_spec_hash: str
    ilc_candidate_hash: str
    feedforward_hash: str
    immutable_safety_kernel_hash: str
    parent_snapshot_hash: str
    rollback_target_hash: str
    source_evidence: tuple[FeedbackEvidenceRef, ...]
    tracking_error_reduction: float
    accepted_update_count: int
    rollback_count: int
    evidence_domain: str = "SIM"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.feedback.controller_candidate.v1"

    @property
    def candidate_hash(self) -> str:
        return canonical_hash(self._bound_content())

    def _bound_content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "regime_hash": self.regime_hash,
            "controller_hash": self.controller_hash,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "loop_spec_hash": self.loop_spec_hash,
            "ilc_candidate_hash": self.ilc_candidate_hash,
            "feedforward_hash": self.feedforward_hash,
            "immutable_safety_kernel_hash": self.immutable_safety_kernel_hash,
            "parent_snapshot_hash": self.parent_snapshot_hash,
            "rollback_target_hash": self.rollback_target_hash,
            "source_evidence": [item.bound_dict() for item in self.source_evidence],
            "tracking_error_reduction": self.tracking_error_reduction,
            "accepted_update_count": self.accepted_update_count,
            "rollback_count": self.rollback_count,
            "evidence_domain": self.evidence_domain,
            "activation_ceiling": self.activation_ceiling,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._bound_content(),
            "candidate_hash": self.candidate_hash,
            "source_evidence": [item.to_dict() for item in self.source_evidence],
        }


@dataclass(frozen=True)
class FeedbackPromotionCheck:
    gate: str
    passed: bool
    missing: bool
    safety_critical: bool
    detail: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeedbackEvolutionResult:
    candidate: FeedbackControllerCandidate
    decision: FeedbackEvolutionDecision
    checks: tuple[FeedbackPromotionCheck, ...]
    candidate_artifact_verified: bool
    activated: bool = False
    registry_mutated: bool = False
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.feedback.evolution_result.v1"

    @property
    def result_hash(self) -> str:
        return canonical_hash(self.to_dict(include_result_hash=False))

    def to_dict(self, *, include_result_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_dict(),
            "decision": self.decision.value,
            "checks": [asdict(check) for check in self.checks],
            "candidate_artifact_verified": self.candidate_artifact_verified,
            "activation": {
                "activated": self.activated,
                "registry_mutated": self.registry_mutated,
                "hardware_command_sent": self.hardware_command_sent,
                "rollback_target_hash": self.candidate.rollback_target_hash,
                "reason": (
                    "all gates passed"
                    if self.activated
                    else "candidate retained offline until every promotion gate passes"
                ),
            },
            "claims": {
                "evidence_domain": "SIM",
                "real_hardware": False,
                "self_evolution_scope": "bounded C2 trial-to-trial feedforward",
                "unrestricted_online_weight_updates": False,
            },
        }
        if include_result_hash:
            value["result_hash"] = canonical_hash(value)
        return value


class FeedbackPromotionGate:
    """Apply F1-F15 with missing evidence distinct from a measured failure."""

    def evaluate(
        self,
        *,
        candidate: FeedbackControllerCandidate,
        checks: tuple[FeedbackPromotionCheck, ...],
        candidate_artifact_verified: bool,
    ) -> FeedbackEvolutionResult:
        expected = tuple(f"F{index}" for index in range(1, 16))
        if tuple(check.gate for check in checks) != expected:
            raise ValueError("Feedback Promotion requires ordered F1-F15 checks")
        measured_failures = [check for check in checks if not check.passed and not check.missing]
        if measured_failures:
            decision = FeedbackEvolutionDecision.REJECTED
        elif any(check.missing for check in checks):
            decision = FeedbackEvolutionDecision.NEED_MORE_EVIDENCE
        else:
            decision = FeedbackEvolutionDecision.SIM_CHAMPION
        if not candidate_artifact_verified:
            decision = FeedbackEvolutionDecision.REJECTED
        # Evaluation never activates a candidate.  Registry mutation is a
        # separate, authorized transaction after the complete gate.
        return FeedbackEvolutionResult(
            candidate=candidate,
            decision=decision,
            checks=checks,
            candidate_artifact_verified=candidate_artifact_verified,
        )


def run_g1_feedback_evolution(
    *,
    feedback_path: Path,
    holdout_path: Path,
    ilc_path: Path,
    chaos_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> FeedbackEvolutionResult:
    """Build and gate one offline candidate from immutable Phase 6 evidence."""

    checkout = source_checkout.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("Feedback evolution evidence must be outside the source checkout")

    feedback, feedback_ref = _read_evidence(
        feedback_path,
        kind="feedback_ab",
        expected_schema="rosclaw.g1_feedback.validation.v1",
    )
    holdout, holdout_ref = _read_evidence(
        holdout_path,
        kind="feedback_holdout",
        expected_schema="rosclaw.g1_feedback.holdout.v1",
    )
    ilc, ilc_ref = _read_evidence(
        ilc_path,
        kind="feedback_ilc",
        expected_schema="rosclaw.g1_ilc.validation.v2",
    )
    chaos, chaos_ref = _read_chaos(chaos_path)
    evidence_refs = (feedback_ref, holdout_ref, ilc_ref, chaos_ref)

    body_hash = _common_hash((feedback, holdout, ilc), "body_hash")
    kick_prior_hash = _common_hash((feedback, holdout, ilc), "kick_prior_hash")
    _require_simulation_only(feedback, holdout, ilc, chaos)
    feedforward, ilc_candidate = _load_ilc_candidate(
        ilc,
        source_checkout=checkout,
    )
    if feedforward.body_hash != body_hash:
        raise ValueError("ILC candidate Body hash does not match feedback evidence")

    runtime = build_g1_balance_runtime(body_hash=body_hash)
    snapshot = ControllerSnapshot(
        controller_id=runtime.spec.loop_id + ":controller",
        controller_type=type(runtime.controller).__name__,
        body_hash=body_hash,
        loop_spec_hash=runtime.spec.spec_hash,
        config=runtime.controller.config_dict(),
    )
    receipts = tuple(case["feedback_receipt"] for case in _object_list(feedback, "cases"))
    if not receipts:
        raise ValueError("feedback A/B evidence contains no receipts")
    receipt_snapshot_hashes = {str(receipt.get("controller_snapshot_hash")) for receipt in receipts}
    receipt_controller_hashes = {str(receipt.get("controller_hash")) for receipt in receipts}
    receipt_loop_hashes = {str(receipt.get("loop_spec_hash")) for receipt in receipts}

    stale_fails_closed, bounds_verified, projector_verified = _runtime_fault_probes(runtime)
    trials = _object_list(ilc, "trials")
    accepted_updates = sum(bool(trial.get("update_accepted")) for trial in trials)
    rollback_count = sum(
        int(trial.get("trial", 0)) > 1 and not bool(trial.get("update_accepted"))
        for trial in trials
    )
    safety_kernel_hash = canonical_hash(
        {
            "loop_spec_hash": runtime.spec.spec_hash,
            "output_limits": dict(sorted(runtime.spec.output_limits.items())),
            "fallback_stale_observation": runtime.spec.fallback_stale_observation.value,
            "fallback_deadline_miss": runtime.spec.fallback_deadline_miss.value,
            "fallback_unsafe_projection": runtime.spec.fallback_unsafe_projection.value,
        }
    )
    candidate = FeedbackControllerCandidate(
        body_hash=body_hash,
        kick_prior_hash=kick_prior_hash,
        regime_hash=feedforward.regime_hash,
        controller_hash=runtime.spec.controller_hash,
        controller_snapshot_hash=snapshot.snapshot_hash,
        loop_spec_hash=runtime.spec.spec_hash,
        ilc_candidate_hash=ilc_candidate.candidate_hash,
        feedforward_hash=feedforward.trajectory_hash,
        immutable_safety_kernel_hash=safety_kernel_hash,
        parent_snapshot_hash=snapshot.snapshot_hash,
        rollback_target_hash=snapshot.snapshot_hash,
        source_evidence=evidence_refs,
        tracking_error_reduction=float(ilc.get("error_reduction", 0.0)),
        accepted_update_count=accepted_updates,
        rollback_count=rollback_count,
    )

    checks = _build_checks(
        feedback=feedback,
        holdout=holdout,
        ilc=ilc,
        chaos=chaos,
        receipts=receipts,
        runtime=runtime,
        snapshot=snapshot,
        receipt_snapshot_hashes=receipt_snapshot_hashes,
        receipt_controller_hashes=receipt_controller_hashes,
        receipt_loop_hashes=receipt_loop_hashes,
        stale_fails_closed=stale_fails_closed,
        bounds_verified=bounds_verified,
        projector_verified=projector_verified,
        ilc_candidate=ilc_candidate,
        evidence_refs=evidence_refs,
    )
    result = FeedbackPromotionGate().evaluate(
        candidate=candidate,
        checks=checks,
        candidate_artifact_verified=True,
    )
    _atomic_json(destination, result.to_dict())
    return result


def _build_checks(
    *,
    feedback: dict[str, Any],
    holdout: dict[str, Any],
    ilc: dict[str, Any],
    chaos: dict[str, Any],
    receipts: tuple[dict[str, Any], ...],
    runtime: Any,
    snapshot: ControllerSnapshot,
    receipt_snapshot_hashes: set[str],
    receipt_controller_hashes: set[str],
    receipt_loop_hashes: set[str],
    stale_fails_closed: bool,
    bounds_verified: bool,
    projector_verified: bool,
    ilc_candidate: G1ILCFeedforwardCandidate,
    evidence_refs: tuple[FeedbackEvidenceRef, ...],
) -> tuple[FeedbackPromotionCheck, ...]:
    ref = {item.kind: item.content_hash for item in evidence_refs}
    fixed_rate = bool(
        runtime.spec.rate_hz == 250.0
        and all(float(receipt.get("jitter_p99_ms", float("inf"))) <= 0.05 for receipt in receipts)
        and all(int(receipt.get("dropped_frame_count", -1)) == 0 for receipt in receipts)
    )
    deadline_ok = bool(
        float(feedback.get("deadline_compliance_rate", 0.0)) >= 0.999
        and int(holdout.get("deadline_miss_count", -1)) == 0
        and all(int(receipt.get("deadline_miss_count", -1)) == 0 for receipt in receipts)
        and all(
            int(trial.get("deadline_miss_count", -1)) == 0 for trial in _object_list(ilc, "trials")
        )
    )
    local_corrections = [receipt for receipt in receipts if bool(receipt.get("correction_applied"))]
    local_tracking_improved = bool(
        local_corrections
        and all(bool(receipt.get("tracking_improved")) for receipt in local_corrections)
    )
    ilc_tracking_improved = bool(
        ilc.get("passed") is True
        and ilc.get("monotonic_error") is True
        and float(ilc.get("error_reduction", 0.0)) >= 0.01
    )
    ab_cases = _object_list(feedback, "cases")
    holdout_cases = _object_list(holdout, "holdout_cases")
    fall_ok = _rate_not_increased(
        ab_cases, "baseline", "feedback", "post_kick_fall"
    ) and _flat_rate_not_increased(
        holdout_cases,
        "baseline_fall",
        "feedback_fall",
    )
    limit_ok = (
        _rate_not_increased(ab_cases, "baseline", "feedback", "joint_limit_violation")
        and _rate_not_increased(ab_cases, "baseline", "feedback", "torque_limit_violation")
        and _flat_rate_not_increased(
            holdout_cases,
            "baseline_joint_violation",
            "feedback_joint_violation",
        )
        and _flat_rate_not_increased(
            holdout_cases,
            "baseline_torque_violation",
            "feedback_torque_violation",
        )
    )
    strict_replay = bool(
        all(case.get("trajectory_strict_replay") is True for case in ab_cases)
        and all(case.get("strict_replay") is True for case in holdout_cases)
        and ilc.get("strict_replay") is True
    )
    snapshot_complete = bool(
        receipt_snapshot_hashes == {snapshot.snapshot_hash}
        and receipt_controller_hashes == {runtime.spec.controller_hash}
        and receipt_loop_hashes == {runtime.spec.spec_hash}
    )
    dds = _object(chaos, "dds")
    executor = _object(chaos, "executor")
    dds_chaos = bool(
        chaos.get("passed") is True
        and dds.get("passed") is True
        and executor.get("passed") is True
        and dds.get("real_hardware_opened") is False
        and executor.get("real_hardware_opened") is False
        and int(executor.get("old_trigger_replay_count", -1)) == 0
        and int(executor.get("stale_task_verified_count", -1)) == 0
    )
    # There is intentionally no boolean caller override for F15.  A future
    # canary must be bound to the candidate and independently attested.
    canary_present = False
    return (
        _check(
            "F1",
            fixed_rate,
            False,
            True,
            "250 Hz loop contract and zero dropped frames",
            ref["feedback_ab"],
        ),
        _check(
            "F2",
            deadline_ok,
            False,
            True,
            "deadline miss rate stays below the declared threshold",
            ref["feedback_ab"],
            ref["feedback_holdout"],
            ref["feedback_ilc"],
        ),
        _check(
            "F3",
            stale_fails_closed,
            False,
            True,
            "stale observation removes the residual through fallback",
            snapshot.snapshot_hash,
        ),
        _check(
            "F4",
            bounds_verified and ilc_candidate.residual_peak <= ilc_candidate.residual_limit,
            False,
            True,
            "runtime and learned feedforward remain inside immutable residual bounds",
            ilc_candidate.candidate_hash,
        ),
        _check(
            "F5",
            projector_verified,
            False,
            True,
            "the synchronous runtime routes residual output through SafetyProjector",
            snapshot.snapshot_hash,
        ),
        _check(
            "F6",
            local_tracking_improved and ilc_tracking_improved,
            ilc_tracking_improved and not local_tracking_improved,
            False,
            "ILC tracking improves, but reflex-local correction-window error still needs causal improvement evidence",
            ref["feedback_ab"],
            ref["feedback_ilc"],
        ),
        _check(
            "F7",
            fall_ok,
            False,
            True,
            "fall rate does not increase in A/B or holdout",
            ref["feedback_ab"],
            ref["feedback_holdout"],
        ),
        _check(
            "F8",
            limit_ok,
            False,
            True,
            "torque and joint-limit violations do not regress",
            ref["feedback_ab"],
            ref["feedback_holdout"],
        ),
        _check(
            "F9",
            holdout.get("holdout_passed") is True,
            False,
            False,
            "multi-regime disturbance holdout passes",
            ref["feedback_holdout"],
        ),
        _check(
            "F10",
            ilc.get("wrong_regime_rejected") is True,
            False,
            True,
            "wrong-regime feedforward is rejected",
            ref["feedback_ilc"],
        ),
        _check(
            "F11",
            strict_replay,
            False,
            True,
            "A/B, holdout, and selected ILC trajectory replay exactly",
            ref["feedback_ab"],
            ref["feedback_holdout"],
            ref["feedback_ilc"],
        ),
        _check(
            "F12",
            snapshot_complete,
            False,
            True,
            "controller, loop, and snapshot hashes rederive from current code",
            snapshot.snapshot_hash,
        ),
        _check(
            "F13",
            holdout.get("historical_regression_passed") is True,
            False,
            True,
            "historical Phase 4 motion corpus has no physical regression",
            ref["feedback_holdout"],
        ),
        _check(
            "F14",
            dds_chaos,
            False,
            True,
            "isolated canonical DDS loopback and executor chaos pass",
            ref["dds_chaos"],
        ),
        _check(
            "F15",
            canary_present,
            True,
            True,
            "candidate-bound real G1 Canary is not authorized or present",
        ),
    )


def _runtime_fault_probes(runtime: Any) -> tuple[bool, bool, bool]:
    spec = runtime.spec
    base_action = dict.fromkeys(spec.output_limits, 0.0)
    reference = dict.fromkeys(spec.reference_signals, 0.0)
    actual = {
        "torso_roll": 3.0,
        "torso_pitch": -3.0,
        "com_y_relative": 0.5,
        "support_slip_m": 0.2,
    }
    stale = runtime.tick(
        timestamp_ns=3 * spec.period_ns,
        observation_timestamp_ns=0,
        phase=0.40,
        reference=reference,
        actual=actual,
        base_action=base_action,
    )
    stale_fails_closed = bool(
        stale.projected == {}
        and stale.fallback is spec.fallback_stale_observation
        and stale.fallback is FallbackMode.BASE_POLICY_ONLY
    )
    runtime.reset()
    command = runtime.tick(
        timestamp_ns=1_000_000_000,
        observation_timestamp_ns=1_000_000_000,
        phase=0.40,
        reference=reference,
        actual=actual,
        base_action=base_action,
    )
    bounds_verified = bool(
        command.projected
        and all(
            abs(value) <= spec.output_limits[name] + 1e-12
            for name, value in command.projected.items()
        )
    )
    projector_verified = bool(
        command.fallback is None
        and set(command.projected).issubset(spec.output_limits)
        and command.command_hash.startswith("sha256:")
    )
    return stale_fails_closed, bounds_verified, projector_verified


def _load_ilc_candidate(
    ilc: dict[str, Any],
    *,
    source_checkout: Path,
) -> tuple[ILCFeedforward, G1ILCFeedforwardCandidate]:
    raw = _object(ilc, "candidate_feedforward")
    artifact = Path(str(raw.get("artifact_path", ""))).expanduser().resolve()
    if artifact == source_checkout or source_checkout in artifact.parents:
        raise ValueError("ILC candidate artifact must be outside the source checkout")
    if not artifact.is_file() or artifact.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("ILC candidate artifact is missing or too large")
    if _sha256_file(artifact) != raw.get("artifact_hash"):
        raise ValueError("ILC candidate artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as archive:
        if archive.files != ["feedforward_residual"]:
            raise ValueError("ILC candidate artifact has unexpected arrays")
        values = np.asarray(archive["feedforward_residual"], dtype=np.float64)
    feedforward = ILCFeedforward(
        body_hash=str(raw.get("body_hash", "")),
        regime_hash=str(raw.get("regime_hash", "")),
        joint_names=tuple(str(value) for value in raw.get("joint_names", ())),
        values=values,
        residual_limit=float(raw.get("residual_limit", 0.0)),
        trial=int(raw.get("trial", -1)),
        source_receipt_hashes=tuple(str(value) for value in raw.get("source_receipt_hashes", ())),
    )
    manifest = feedforward.to_manifest()
    candidate = G1ILCFeedforwardCandidate(
        trajectory_hash=feedforward.trajectory_hash,
        body_hash=feedforward.body_hash,
        regime_hash=feedforward.regime_hash,
        joint_names=feedforward.joint_names,
        shape=tuple(int(value) for value in values.shape),
        residual_limit=feedforward.residual_limit,
        residual_peak=float(manifest["residual_peak"]),
        trial=feedforward.trial,
        selected_campaign_trial=int(raw.get("selected_campaign_trial", -1)),
        source_receipt_hashes=feedforward.source_receipt_hashes,
        value_hash=str(manifest["value_hash"]),
        artifact_path=str(artifact),
        artifact_hash=_sha256_file(artifact),
    )
    for field in (
        "trajectory_hash",
        "value_hash",
        "candidate_hash",
        "artifact_hash",
    ):
        expected = candidate.to_dict()[field]
        if raw.get(field) != expected:
            raise ValueError(f"ILC candidate {field} mismatch")
    if list(values.shape) != raw.get("shape"):
        raise ValueError("ILC candidate shape mismatch")
    trials = _object_list(ilc, "trials")
    if not trials or trials[-1].get("feedforward_hash") != candidate.trajectory_hash:
        raise ValueError("ILC selected trial does not bind the candidate feedforward")
    return feedforward, candidate


def _read_evidence(
    path: Path,
    *,
    kind: str,
    expected_schema: str,
) -> tuple[dict[str, Any], FeedbackEvidenceRef]:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"{kind} evidence schema mismatch")
    return value, FeedbackEvidenceRef(
        kind=kind,
        schema_version=expected_schema,
        content_hash=canonical_hash(value),
        path=str(resolved),
    )


def _read_chaos(path: Path) -> tuple[dict[str, Any], FeedbackEvidenceRef]:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    if _object(value, "dds").get("schema_version") != "rosclaw.g1.dds_loopback_receipt.v1":
        raise ValueError("DDS loopback evidence schema mismatch")
    if _object(value, "executor").get("schema_version") != "rosclaw.g1.chaos.v1":
        raise ValueError("executor chaos evidence schema mismatch")
    return value, FeedbackEvidenceRef(
        kind="dds_chaos",
        schema_version="rosclaw.g1.feedback_dds_chaos.aggregate.v1",
        content_hash=canonical_hash(value),
        path=str(resolved),
    )


def _require_simulation_only(*reports: dict[str, Any]) -> None:
    serialized = json.dumps(reports, sort_keys=True, allow_nan=False)
    if '"real_hardware": true' in serialized or '"real_hardware_opened": true' in serialized:
        raise ValueError("Feedback evolution received unexpected real-hardware evidence")
    claims = [_object(report, "claims") for report in reports[:3]]
    if any(claim.get("evidence_domain") != "SIM" for claim in claims):
        raise ValueError("Feedback evolution requires SIM evidence domains")


def _common_hash(reports: tuple[dict[str, Any], ...], field: str) -> str:
    values = {str(report.get(field, "")) for report in reports}
    if len(values) != 1 or not next(iter(values)).startswith("sha256:"):
        raise ValueError(f"feedback evidence {field} values do not match")
    return next(iter(values))


def _object(value: dict[str, Any], field: str) -> dict[str, Any]:
    item = value.get(field)
    if not isinstance(item, dict):
        raise ValueError(f"feedback evidence field {field!r} is not an object")
    return item


def _object_list(value: dict[str, Any], field: str) -> tuple[dict[str, Any], ...]:
    items = value.get(field)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"feedback evidence field {field!r} is not an object list")
    return tuple(items)


def _rate_not_increased(
    rows: tuple[dict[str, Any], ...],
    baseline_field: str,
    candidate_field: str,
    metric: str,
) -> bool:
    if not rows:
        return False
    return sum(bool(_object(row, candidate_field).get(metric)) for row in rows) <= sum(
        bool(_object(row, baseline_field).get(metric)) for row in rows
    )


def _flat_rate_not_increased(
    rows: tuple[dict[str, Any], ...], baseline_field: str, candidate_field: str
) -> bool:
    if not rows:
        return False
    return sum(bool(row.get(candidate_field)) for row in rows) <= sum(
        bool(row.get(baseline_field)) for row in rows
    )


def _check(
    gate: str,
    passed: bool,
    missing: bool,
    safety_critical: bool,
    detail: str,
    *evidence_refs: str,
) -> FeedbackPromotionCheck:
    return FeedbackPromotionCheck(
        gate=gate,
        passed=bool(passed),
        missing=bool(missing),
        safety_critical=safety_critical,
        detail=detail,
        evidence_refs=tuple(evidence_refs),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"feedback evidence is missing or too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"feedback evidence is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "FeedbackControllerCandidate",
    "FeedbackEvidenceRef",
    "FeedbackEvolutionDecision",
    "FeedbackEvolutionResult",
    "FeedbackPromotionCheck",
    "FeedbackPromotionGate",
    "run_g1_feedback_evolution",
]
