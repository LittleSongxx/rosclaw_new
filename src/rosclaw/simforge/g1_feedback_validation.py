"""Same-scenario GoalForge A/B validation for the Phase 6 Feedback Plane."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_VALIDATION_SECRET = b"rosclaw-feedback-phase6-validation"


@dataclass(frozen=True)
class G1FeedbackMetrics:
    status: str
    success: bool
    target_error_m: float
    support_foot_slip_m: float
    com_margin_min_m: float
    torso_roll_peak_rad: float
    torso_pitch_peak_rad: float
    post_kick_fall: bool
    joint_limit_violation: bool
    torque_limit_violation: bool
    robustness: float

    @classmethod
    def from_episode(cls, episode: GoalForgeEpisode) -> G1FeedbackMetrics:
        result = episode.result
        return cls(
            status=result.status.value,
            success=result.success,
            target_error_m=result.target_error_m,
            support_foot_slip_m=result.support_foot_slip_m,
            com_margin_min_m=result.com_margin_min_m,
            torso_roll_peak_rad=result.torso_roll_peak_rad,
            torso_pitch_peak_rad=result.torso_pitch_peak_rad,
            post_kick_fall=result.post_kick_fall,
            joint_limit_violation=result.joint_limit_violation,
            torque_limit_violation=result.torque_limit_violation,
            robustness=result.robustness,
        )


@dataclass(frozen=True)
class G1FeedbackABCase:
    scenario_id: str
    scenario_commitment: str
    disturbance_n: float
    baseline: G1FeedbackMetrics
    feedback: G1FeedbackMetrics
    feedback_receipt: dict[str, Any]
    trajectory_strict_replay: bool

    @property
    def passed(self) -> bool:
        no_safety_regression = (
            not self.feedback.post_kick_fall
            and not self.feedback.joint_limit_violation
            and not self.feedback.torque_limit_violation
        )
        return self.feedback.success and no_safety_regression and self.trajectory_strict_replay

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class G1FeedbackValidation:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    cases: tuple[G1FeedbackABCase, ...]
    deadline_compliance_rate: float
    baseline_success_rate: float
    feedback_success_rate: float
    rescue_count: int
    nominal_no_regression: bool
    schema_version: str = "rosclaw.g1_feedback.validation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.cases
            and all(case.passed for case in self.cases)
            and self.deadline_compliance_rate >= 0.999
            and self.feedback_success_rate >= self.baseline_success_rate
            and self.rescue_count >= 1
            and self.nominal_no_regression
        )

    def to_dict(self) -> dict[str, Any]:
        promotion_blockers = (
            "F6 receipt-local tracking error does not yet improve monotonically",
            "F9 broad disturbance Holdout is not yet complete",
            "F13 historical motion regression is not yet complete",
            "F14 canonical DDS chaos is not yet complete",
            "F15 real-body Canary is not authorized or complete",
        )
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "cases": [case.to_dict() for case in self.cases],
            "deadline_compliance_rate": self.deadline_compliance_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "feedback_success_rate": self.feedback_success_rate,
            "rescue_count": self.rescue_count,
            "nominal_no_regression": self.nominal_no_regression,
            "passed": self.passed,
            "promotion_assessment": {
                "status": "NEED_MORE_EVIDENCE",
                "eligible": False,
                "blockers": list(promotion_blockers),
            },
            "claims": {
                "evidence_domain": "SIM",
                "real_hardware": False,
                "learned_policy": False,
                "controller_role": "bounded residual around qualified kick prior",
            },
        }


def run_g1_feedback_validation(
    *,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    disturbances_n: tuple[float, ...] = (0.0, 35.0, 65.0, 80.0),
) -> G1FeedbackValidation:
    """Run deterministic off/on pairs and persist raw evidence outside source."""

    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("Feedback validation evidence must be outside the source checkout")
    if not disturbances_n or any(not 0.0 <= value <= 80.0 for value in disturbances_n):
        raise ValueError("disturbances_n must be non-empty values in [0, 80]")
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=5)
    ledger = SeedLedger(task_id="g1_penalty_kick", secret=_VALIDATION_SECRET)
    nominal = generate_goalforge_scenarios(
        ledger=ledger,
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]
    cases: list[G1FeedbackABCase] = []
    total_ticks = 0
    deadline_misses = 0
    for disturbance_n in disturbances_n:
        scenario = replace(
            nominal,
            scenario_id=f"g1-feedback-ab-{int(disturbance_n):03d}n",
            disturbance_n=disturbance_n,
        )
        parameters = ShotParameters()
        baseline = backend.run(scenario, parameters)
        runtime = build_g1_balance_runtime(body_hash=backend.qualification.body_hash)
        feedback = backend.run(scenario, parameters, feedback_runtime=runtime)
        latencies = tuple(record.latency_ns for record in runtime.records)
        replay_clock = RecordedLatencyClock(latencies)
        replay_runtime = build_g1_balance_runtime(
            body_hash=backend.qualification.body_hash,
            compute_clock_ns=replay_clock,
        )
        replay = backend.run(scenario, parameters, feedback_runtime=replay_runtime)
        strict_replay = bool(
            replay.result.summary_dict() == feedback.result.summary_dict()
            and trajectory_digest(replay.trajectory) == trajectory_digest(feedback.trajectory)
        )
        receipt = runtime.build_receipt(
            action_id=f"g1-feedback-ab-{int(disturbance_n):03d}n",
            strict_replay=strict_replay,
            evidence_domain="SIM",
        )
        total_ticks += receipt.samples
        deadline_misses += receipt.deadline_miss_count
        cases.append(
            G1FeedbackABCase(
                scenario_id=scenario.scenario_id,
                scenario_commitment=scenario.scenario_commitment,
                disturbance_n=disturbance_n,
                baseline=G1FeedbackMetrics.from_episode(baseline),
                feedback=G1FeedbackMetrics.from_episode(feedback),
                feedback_receipt=receipt.to_dict(),
                trajectory_strict_replay=strict_replay,
            )
        )
    baseline_success_rate = sum(case.baseline.success for case in cases) / len(cases)
    feedback_success_rate = sum(case.feedback.success for case in cases) / len(cases)
    nominal_case = min(cases, key=lambda item: item.disturbance_n)
    nominal_no_regression = bool(
        nominal_case.feedback.success
        and nominal_case.feedback.target_error_m <= nominal_case.baseline.target_error_m + 0.02
        and nominal_case.feedback.torso_roll_peak_rad
        <= nominal_case.baseline.torso_roll_peak_rad + 0.02
    )
    result = G1FeedbackValidation(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        backend_commit=backend.qualification.backend_commit,
        cases=tuple(cases),
        deadline_compliance_rate=(total_ticks - deadline_misses) / total_ticks,
        baseline_success_rate=baseline_success_rate,
        feedback_success_rate=feedback_success_rate,
        rescue_count=sum(not case.baseline.success and case.feedback.success for case in cases),
        nominal_no_regression=nominal_no_regression,
    )
    _atomic_json(destination, result.to_dict())
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "G1FeedbackABCase",
    "G1FeedbackMetrics",
    "G1FeedbackValidation",
    "run_g1_feedback_validation",
]
