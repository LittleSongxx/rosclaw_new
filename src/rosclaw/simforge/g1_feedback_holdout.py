"""Multi-regime Holdout and historical-motion regression for G1 feedback."""

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
from rosclaw.simforge.phase4_run import historical_goalforge_scenarios
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_HOLDOUT_SECRET = b"rosclaw-phase6-feedback-holdout"
_LATENCY_NS = 100_000
_CLOCK_CAPACITY = 4_000


@dataclass(frozen=True)
class FeedbackHoldoutRegime:
    name: str
    support_friction: float
    latency_ms: float
    joint_zero_bias_rad: float
    disturbance_n: float


@dataclass(frozen=True)
class FeedbackHoldoutCase:
    regime: FeedbackHoldoutRegime
    scenario_commitment: str
    baseline_status: str
    feedback_status: str
    baseline_success: bool
    feedback_success: bool
    baseline_roll_peak_rad: float
    feedback_roll_peak_rad: float
    baseline_fall: bool
    feedback_fall: bool
    baseline_joint_violation: bool
    feedback_joint_violation: bool
    baseline_torque_violation: bool
    feedback_torque_violation: bool
    correction_applied: bool
    transparent_when_inactive: bool
    strict_replay: bool

    @property
    def passed(self) -> bool:
        return bool(
            (not self.baseline_success or self.feedback_success)
            and (not self.feedback_fall or self.baseline_fall)
            and (not self.feedback_joint_violation or self.baseline_joint_violation)
            and (not self.feedback_torque_violation or self.baseline_torque_violation)
            and self.transparent_when_inactive
            and self.strict_replay
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class HistoricalMotionCase:
    scenario_id: str
    scenario_commitment: str
    baseline_status: str
    feedback_status: str
    correction_applied: bool
    result_exact: bool
    physical_trajectory_exact: bool

    @property
    def passed(self) -> bool:
        return bool(
            not self.correction_applied and self.result_exact and self.physical_trajectory_exact
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class G1FeedbackHoldoutValidation:
    body_hash: str
    kick_prior_hash: str
    holdout_cases: tuple[FeedbackHoldoutCase, ...]
    historical_cases: tuple[HistoricalMotionCase, ...]
    baseline_success_rate: float
    feedback_success_rate: float
    rescue_count: int
    deadline_miss_count: int
    simulation_episode_count: int
    schema_version: str = "rosclaw.g1_feedback.holdout.v1"

    @property
    def holdout_passed(self) -> bool:
        return bool(
            self.holdout_cases
            and all(case.passed for case in self.holdout_cases)
            and self.feedback_success_rate >= self.baseline_success_rate
            and self.rescue_count >= 1
            and self.deadline_miss_count == 0
        )

    @property
    def historical_regression_passed(self) -> bool:
        return bool(self.historical_cases and all(case.passed for case in self.historical_cases))

    @property
    def passed(self) -> bool:
        return self.holdout_passed and self.historical_regression_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "holdout_cases": [case.to_dict() for case in self.holdout_cases],
            "historical_cases": [case.to_dict() for case in self.historical_cases],
            "baseline_success_rate": self.baseline_success_rate,
            "feedback_success_rate": self.feedback_success_rate,
            "rescue_count": self.rescue_count,
            "deadline_miss_count": self.deadline_miss_count,
            "simulation_episode_count": self.simulation_episode_count,
            "holdout_passed": self.holdout_passed,
            "historical_regression_passed": self.historical_regression_passed,
            "passed": self.passed,
            "claims": {
                "evidence_domain": "SIM",
                "real_hardware": False,
                "holdout_axes": [
                    "support_friction",
                    "control_latency",
                    "joint_zero_bias",
                    "lateral_disturbance",
                ],
                "disturbance_timing": "fixed backend window 4.6-4.8 seconds",
            },
        }


def run_g1_feedback_holdout(
    *,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1FeedbackHoldoutValidation:
    """Evaluate a frozen controller on private regimes and Phase 4 motions."""

    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("Feedback Holdout evidence must be outside the source checkout")
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=5)
    nominal = generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_HOLDOUT_SECRET),
        partition=Partition.HOLDOUT,
        count=1,
        generation=0,
    )[0]
    parameters = ShotParameters()
    cases: list[FeedbackHoldoutCase] = []
    deadline_misses = 0
    for index, regime in enumerate(_regimes()):
        scenario = replace(
            nominal,
            scenario_id=f"g1-feedback-holdout-{index:02d}-{regime.name}",
            support_ground_friction=regime.support_friction,
            control_latency_ms=regime.latency_ms,
            joint_zero_bias_rad=regime.joint_zero_bias_rad,
            disturbance_n=regime.disturbance_n,
        )
        baseline = backend.run(scenario, parameters)
        feedback = _run_feedback(backend, scenario, parameters)
        replay = _run_feedback(backend, scenario, parameters)
        receipt = feedback.feedback_receipt
        assert receipt is not None
        deadline_misses += receipt.deadline_miss_count
        correction = receipt.correction_applied
        cases.append(
            FeedbackHoldoutCase(
                regime=regime,
                scenario_commitment=scenario.scenario_commitment,
                baseline_status=baseline.result.status.value,
                feedback_status=feedback.result.status.value,
                baseline_success=baseline.result.success,
                feedback_success=feedback.result.success,
                baseline_roll_peak_rad=baseline.result.torso_roll_peak_rad,
                feedback_roll_peak_rad=feedback.result.torso_roll_peak_rad,
                baseline_fall=baseline.result.post_kick_fall,
                feedback_fall=feedback.result.post_kick_fall,
                baseline_joint_violation=baseline.result.joint_limit_violation,
                feedback_joint_violation=feedback.result.joint_limit_violation,
                baseline_torque_violation=baseline.result.torque_limit_violation,
                feedback_torque_violation=feedback.result.torque_limit_violation,
                correction_applied=correction,
                transparent_when_inactive=(
                    correction or _physical_digest(baseline) == _physical_digest(feedback)
                ),
                strict_replay=(
                    replay.result.summary_dict() == feedback.result.summary_dict()
                    and trajectory_digest(replay.trajectory)
                    == trajectory_digest(feedback.trajectory)
                ),
            )
        )

    historical: list[HistoricalMotionCase] = []
    for scenario in historical_goalforge_scenarios():
        baseline = backend.run(scenario, parameters)
        feedback = _run_feedback(backend, scenario, parameters)
        receipt = feedback.feedback_receipt
        assert receipt is not None
        deadline_misses += receipt.deadline_miss_count
        historical.append(
            HistoricalMotionCase(
                scenario_id=scenario.scenario_id,
                scenario_commitment=scenario.scenario_commitment,
                baseline_status=baseline.result.status.value,
                feedback_status=feedback.result.status.value,
                correction_applied=receipt.correction_applied,
                result_exact=(baseline.result.summary_dict() == feedback.result.summary_dict()),
                physical_trajectory_exact=(
                    _physical_digest(baseline) == _physical_digest(feedback)
                ),
            )
        )

    baseline_rate = sum(case.baseline_success for case in cases) / len(cases)
    feedback_rate = sum(case.feedback_success for case in cases) / len(cases)
    result = G1FeedbackHoldoutValidation(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        holdout_cases=tuple(cases),
        historical_cases=tuple(historical),
        baseline_success_rate=baseline_rate,
        feedback_success_rate=feedback_rate,
        rescue_count=sum(not case.baseline_success and case.feedback_success for case in cases),
        deadline_miss_count=deadline_misses,
        simulation_episode_count=3 * len(cases) + 2 * len(historical),
    )
    _atomic_json(destination, result.to_dict())
    return result


def _regimes() -> tuple[FeedbackHoldoutRegime, ...]:
    return (
        FeedbackHoldoutRegime("nominal-0n", 1.00, 0.0, 0.000, 0.0),
        FeedbackHoldoutRegime("nominal-65n", 1.00, 0.0, 0.000, 65.0),
        FeedbackHoldoutRegime("nominal-80n", 1.00, 0.0, 0.000, 80.0),
        FeedbackHoldoutRegime("friction-075", 0.75, 0.0, 0.000, 80.0),
        FeedbackHoldoutRegime("friction-055", 0.55, 0.0, 0.000, 80.0),
        FeedbackHoldoutRegime("latency-20ms", 1.00, 20.0, 0.000, 80.0),
        FeedbackHoldoutRegime("latency-40ms", 1.00, 40.0, 0.000, 80.0),
        FeedbackHoldoutRegime("bias-plus", 1.00, 0.0, 0.015, 80.0),
        FeedbackHoldoutRegime("bias-minus", 1.00, 0.0, -0.015, 80.0),
        FeedbackHoldoutRegime("mixed-a", 0.75, 20.0, 0.010, 80.0),
        FeedbackHoldoutRegime("mixed-b", 0.60, 40.0, -0.010, 80.0),
    )


def _run_feedback(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
) -> GoalForgeEpisode:
    runtime = build_g1_balance_runtime(
        body_hash=backend.qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )
    return backend.run(scenario, parameters, feedback_runtime=runtime)


def _physical_digest(episode: GoalForgeEpisode) -> str:
    physical = {
        name: values
        for name, values in episode.trajectory.items()
        if not name.startswith("feedback_")
        and not name.startswith("feedforward_")
        and not name.startswith("combined_")
    }
    return trajectory_digest(physical)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
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
    "FeedbackHoldoutCase",
    "FeedbackHoldoutRegime",
    "G1FeedbackHoldoutValidation",
    "HistoricalMotionCase",
    "run_g1_feedback_holdout",
]
