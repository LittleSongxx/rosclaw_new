"""Counterfactual causal-window validation for the G1 balance reflex.

This validator is intentionally narrower than a promotion suite.  It pins one
qualified MuJoCo scenario, compares feedback off/on, proves that the prefix is
identical before the reflex activates, and measures the post-disturbance
attitude response.  A pass supports a local simulation causal claim only.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_feedback_validation import G1FeedbackMetrics
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_CAUSAL_SECRET = b"rosclaw-feedback-phase6-validation"
_DISTURBANCE_START_SEC = 4.6


@dataclass(frozen=True)
class AttitudeWindowMetrics:
    start_sec: float
    end_sec: float
    integrated_error_rad_sec: float
    peak_error_rad: float
    tail_mean_error_rad: float


@dataclass(frozen=True)
class G1CausalFeedbackValidation:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    scenario_id: str
    scenario_commitment: str
    disturbance_n: float
    baseline: G1FeedbackMetrics
    feedback: G1FeedbackMetrics
    baseline_window: AttitudeWindowMetrics
    feedback_window: AttitudeWindowMetrics
    baseline_early_window: AttitudeWindowMetrics
    feedback_early_window: AttitudeWindowMetrics
    activation_start_sec: float
    activation_end_sec: float
    active_trace_samples: int
    max_projected_residual_rad: float
    identical_pre_activation_prefix: bool
    counterfactual_diverged_after_activation: bool
    strict_replay: bool
    deadline_compliance_rate: float
    feedback_receipt: dict[str, Any]
    schema_version: str = "rosclaw.g1_feedback.causal_validation.v1"

    @property
    def integrated_error_improvement(self) -> float:
        baseline = self.baseline_window.integrated_error_rad_sec
        return (baseline - self.feedback_window.integrated_error_rad_sec) / max(baseline, 1e-12)

    @property
    def peak_error_improvement(self) -> float:
        baseline = self.baseline_window.peak_error_rad
        return (baseline - self.feedback_window.peak_error_rad) / max(baseline, 1e-12)

    @property
    def early_transient_regression(self) -> bool:
        return (
            self.feedback_early_window.integrated_error_rad_sec
            > self.baseline_early_window.integrated_error_rad_sec + 1e-12
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.feedback.success
            and not self.feedback.post_kick_fall
            and not self.feedback.joint_limit_violation
            and not self.feedback.torque_limit_violation
            and self.active_trace_samples > 0
            and self.max_projected_residual_rad > 0.0
            and self.identical_pre_activation_prefix
            and self.counterfactual_diverged_after_activation
            and self.strict_replay
            and self.deadline_compliance_rate >= 0.999
            and self.integrated_error_improvement >= 0.05
            and self.peak_error_improvement >= 0.05
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "scenario_id": self.scenario_id,
            "scenario_commitment": self.scenario_commitment,
            "disturbance_n": self.disturbance_n,
            "baseline": asdict(self.baseline),
            "feedback": asdict(self.feedback),
            "baseline_window": asdict(self.baseline_window),
            "feedback_window": asdict(self.feedback_window),
            "baseline_early_window": asdict(self.baseline_early_window),
            "feedback_early_window": asdict(self.feedback_early_window),
            "activation_start_sec": self.activation_start_sec,
            "activation_end_sec": self.activation_end_sec,
            "active_trace_samples": self.active_trace_samples,
            "max_projected_residual_rad": self.max_projected_residual_rad,
            "identical_pre_activation_prefix": self.identical_pre_activation_prefix,
            "counterfactual_diverged_after_activation": (
                self.counterfactual_diverged_after_activation
            ),
            "strict_replay": self.strict_replay,
            "deadline_compliance_rate": self.deadline_compliance_rate,
            "integrated_error_improvement": self.integrated_error_improvement,
            "peak_error_improvement": self.peak_error_improvement,
            "early_transient_regression": self.early_transient_regression,
            "feedback_receipt": self.feedback_receipt,
            "passed": self.passed,
            "promotion_assessment": {
                "status": "NEED_MORE_EVIDENCE",
                "eligible": False,
                "blockers": [
                    "single deterministic SIM scenario is not a disturbance distribution",
                    "early active-window attitude error is not required to improve monotonically",
                    "historical Anchor and multi-seed retention gates remain mandatory",
                    "real-body Canary is not authorized or complete",
                ],
            },
            "claims": {
                "evidence_domain": "SIM",
                "causal_scope": "matched feedback-off/on counterfactual in one pinned scenario",
                "real_hardware": False,
                "learned_policy": False,
                "hardware_authorized": False,
            },
        }


def run_g1_causal_feedback_validation(
    *,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    disturbance_n: float = 80.0,
) -> G1CausalFeedbackValidation:
    """Run a strict-replay off/on counterfactual and persist external evidence."""

    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("Causal feedback evidence must be outside the source checkout")
    if not 0.0 < disturbance_n <= 80.0:
        raise ValueError("disturbance_n must be in (0, 80]")

    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    nominal = generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_CAUSAL_SECRET),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]
    scenario = replace(
        nominal,
        scenario_id=f"g1-feedback-causal-{int(disturbance_n):03d}n",
        disturbance_n=disturbance_n,
    )
    parameters = ShotParameters()
    baseline = backend.run(scenario, parameters)
    runtime = build_g1_balance_runtime(body_hash=backend.qualification.body_hash)
    feedback = backend.run(scenario, parameters, feedback_runtime=runtime)
    replay_runtime = build_g1_balance_runtime(
        body_hash=backend.qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock(
            tuple(record.latency_ns for record in runtime.records)
        ),
    )
    replay = backend.run(scenario, parameters, feedback_runtime=replay_runtime)
    strict_replay = bool(
        replay.result.summary_dict() == feedback.result.summary_dict()
        and trajectory_digest(replay.trajectory) == trajectory_digest(feedback.trajectory)
    )
    receipt = runtime.build_receipt(
        action_id=scenario.scenario_id,
        strict_replay=strict_replay,
        evidence_domain="SIM",
    )

    time = np.asarray(feedback.trajectory["time"], dtype=np.float64)
    active = np.asarray(feedback.trajectory["feedback_active"], dtype=bool)
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise RuntimeError("Pinned causal scenario did not activate the feedback controller")
    first_active = int(active_indices[0])
    last_active = int(active_indices[-1])
    activation_start = float(time[first_active])
    activation_end = float(time[last_active])
    baseline_error = _attitude_error(baseline.trajectory["torso_quaternion"])
    feedback_error = _attitude_error(feedback.trajectory["torso_quaternion"])
    window_end = float(time[-1])
    early_end = min(window_end, activation_end)
    baseline_window = _window_metrics(time, baseline_error, _DISTURBANCE_START_SEC, window_end)
    feedback_window = _window_metrics(time, feedback_error, _DISTURBANCE_START_SEC, window_end)
    baseline_early = _window_metrics(time, baseline_error, activation_start, early_end)
    feedback_early = _window_metrics(time, feedback_error, activation_start, early_end)
    prefix_equal = all(
        np.array_equal(
            baseline.trajectory[name][:first_active],
            feedback.trajectory[name][:first_active],
        )
        for name in ("joint_position", "torso_quaternion", "ball_pose")
    )
    divergence = (
        float(
            np.max(
                np.abs(
                    baseline.trajectory["torso_quaternion"][first_active:]
                    - feedback.trajectory["torso_quaternion"][first_active:]
                )
            )
        )
        > 1e-9
    )
    residual = np.asarray(feedback.trajectory["feedback_residual"], dtype=np.float64)
    result = G1CausalFeedbackValidation(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        backend_commit=backend.qualification.backend_commit,
        scenario_id=scenario.scenario_id,
        scenario_commitment=scenario.scenario_commitment,
        disturbance_n=disturbance_n,
        baseline=G1FeedbackMetrics.from_episode(baseline),
        feedback=G1FeedbackMetrics.from_episode(feedback),
        baseline_window=baseline_window,
        feedback_window=feedback_window,
        baseline_early_window=baseline_early,
        feedback_early_window=feedback_early,
        activation_start_sec=activation_start,
        activation_end_sec=activation_end,
        active_trace_samples=int(active_indices.size),
        max_projected_residual_rad=float(np.max(np.abs(residual))),
        identical_pre_activation_prefix=prefix_equal,
        counterfactual_diverged_after_activation=divergence,
        strict_replay=strict_replay,
        deadline_compliance_rate=(receipt.samples - receipt.deadline_miss_count) / receipt.samples,
        feedback_receipt=receipt.to_dict(),
    )
    _atomic_json(destination, result.to_dict())
    return result


def _attitude_error(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("torso quaternion trace must have shape [N, 4]")
    w, x, y, z = (quaternion[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.sqrt(roll * roll + pitch * pitch)


def _window_metrics(
    time: np.ndarray,
    error: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> AttitudeWindowMetrics:
    selected = (time >= start_sec) & (time <= end_sec)
    if np.count_nonzero(selected) < 2:
        raise ValueError("causal attitude window must contain at least two trace samples")
    window_time = time[selected]
    window_error = error[selected]
    tail_start = max(start_sec, end_sec - 0.5)
    tail = error[(time >= tail_start) & (time <= end_sec)]
    return AttitudeWindowMetrics(
        start_sec=float(window_time[0]),
        end_sec=float(window_time[-1]),
        integrated_error_rad_sec=_trapezoid(window_error, window_time),
        peak_error_rad=float(np.max(window_error)),
        tail_mean_error_rad=float(np.mean(tail)),
    )


def _trapezoid(value: np.ndarray, coordinate: np.ndarray) -> float:
    """NumPy 1.24-compatible trapezoidal integration without deprecation warnings."""

    return float(np.sum(0.5 * (value[:-1] + value[1:]) * np.diff(coordinate)))


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
    "AttitudeWindowMetrics",
    "G1CausalFeedbackValidation",
    "run_g1_causal_feedback_validation",
]
