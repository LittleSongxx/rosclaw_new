"""Matched MuJoCo evaluation for a loaded Phase 7 residual candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.continual.contracts import PolicyVersion
from rosclaw.continual.g1_goalforge import build_g1_policy_lineage
from rosclaw.continual.inference import (
    ResidualCandidateArtifact,
    build_g1_candidate_runtime,
    load_residual_candidate,
)
from rosclaw.feedback.contracts import FeedbackFrame, FeedbackLoopSpec, canonical_hash
from rosclaw.feedback.controllers.base import ZeroResidualController
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.feedback.runtime import FeedbackRuntime
from rosclaw.simforge.backends.unitree_mujoco_backend import G1MuJoCoBackend, GoalForgeEpisode
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_SUITE_SECRET = b"rosclaw-phase7.1-candidate-evaluation-v1"
_FAILURE_TARGET_ERROR_M = 2.0
_LATENCY_NS = 100_000
_CLOCK_CAPACITY = 5000
_ACTION_OUTPUTS = {
    "waist_roll_residual": "joint:waist_roll_joint",
    "right_hip_roll_residual": "joint:right_hip_roll_joint",
    "right_hip_yaw_residual": "joint:right_hip_yaw_joint",
    "kick_phase_rate": "skill:kick_phase_rate",
}


@dataclass(frozen=True)
class CandidateEvaluationCounts:
    recent: int = 50
    anchor: int = 50
    boundary: int = 100
    self_partition: int = 50

    def __post_init__(self) -> None:
        if min(self.recent, self.anchor, self.boundary, self.self_partition) < 0:
            raise ValueError("candidate evaluation partition counts must be non-negative")
        if self.total <= 0:
            raise ValueError("candidate evaluation requires at least one scenario")
        if self.total > 1000:
            raise ValueError("candidate evaluation is limited to 1000 matched scenarios")

    @property
    def total(self) -> int:
        return self.recent + self.anchor + self.boundary + self.self_partition

    def to_dict(self) -> dict[str, int]:
        return {
            "recent": self.recent,
            "anchor": self.anchor,
            "boundary": self.boundary,
            "self": self.self_partition,
            "total": self.total,
        }


@dataclass(frozen=True)
class CandidateEvaluationRow:
    scenario_id: str
    scenario_commitment: str
    replay_partition: str
    arm: str
    policy_version: int | None
    policy_version_hash: str
    status: str
    success: bool
    contact: bool
    goal_crossed: bool
    penalized_target_error_m: float
    conditional_target_error_m: float | None
    ball_speed_mps: float
    fall: bool
    joint_violation: bool
    torque_violation: bool
    actuator_saturation: bool
    com_margin_min_m: float
    support_slip_m: float
    tracking_rms_rad: float
    energy_proxy: float
    action_drift_rms: float
    trajectory_hash: str
    inference_receipt_hash: str | None
    version_switch_count: int

    def __post_init__(self) -> None:
        metrics = {
            "penalized_target_error_m": self.penalized_target_error_m,
            "ball_speed_mps": self.ball_speed_mps,
            "com_margin_min_m": self.com_margin_min_m,
            "support_slip_m": self.support_slip_m,
            "tracking_rms_rad": self.tracking_rms_rad,
            "energy_proxy": self.energy_proxy,
            "action_drift_rms": self.action_drift_rms,
        }
        if self.conditional_target_error_m is not None:
            metrics["conditional_target_error_m"] = self.conditional_target_error_m
        invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(
                "candidate evaluation row contains non-finite metrics: " + ", ".join(invalid)
            )


@dataclass(frozen=True)
class CandidateMatchedEvaluation:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    candidate_policy: dict[str, Any]
    parent_policy: dict[str, Any]
    counts: dict[str, int]
    training_seed_count: int
    rows: tuple[CandidateEvaluationRow, ...]
    aggregates: dict[str, Any]
    paired_statistics: dict[str, Any]
    gate: dict[str, Any]
    replay_checks: dict[str, bool]
    schema_version: str = "rosclaw.continual.g1_candidate_matched_evaluation.v1"

    @property
    def passed(self) -> bool:
        return bool(self.gate["candidate_motion_effect_proven"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "candidate_policy": self.candidate_policy,
            "parent_policy": self.parent_policy,
            "counts": self.counts,
            "training_seed_count": self.training_seed_count,
            "arms": [
                "frozen_prior",
                "active_parent_v2",
                "candidate_v3",
                "fresh_network",
            ],
            "rows": [asdict(row) for row in self.rows],
            "aggregates": self.aggregates,
            "paired_statistics": self.paired_statistics,
            "gate": self.gate,
            "replay_checks": self.replay_checks,
            "passed": self.passed,
            "claims": {
                "evidence_domain": "SIM",
                "same_scenario_physics": True,
                "candidate_artifact_executed": True,
                "candidate_activated": False,
                "registry_mutated": False,
                "dds_opened": False,
                "hardware_authorized": False,
            },
        }


def run_g1_candidate_matched_evaluation(
    *,
    asset_root: Path,
    candidate_artifact_path: Path,
    candidate_policy: PolicyVersion,
    output_path: Path,
    source_checkout: Path,
    counts: CandidateEvaluationCounts | None = None,
    training_seed_count: int = 1,
    suite_shard: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> CandidateMatchedEvaluation:
    """Run all four arms with identical scenario, physics, and shot parameters."""

    counts = counts or CandidateEvaluationCounts()
    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("candidate evaluation evidence must be outside the source checkout")
    if destination.exists():
        raise FileExistsError(f"candidate evaluation output already exists: {destination}")
    if training_seed_count <= 0:
        raise ValueError("training_seed_count must be positive")
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=5)
    qualification = backend.qualification
    lineage = build_g1_policy_lineage(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        motion_hash=qualification.motion_hash,
        backend_commit=qualification.backend_commit,
        torque_guard_scale=backend.torque_guard_scale,
        through_version=2,
    )
    parent = lineage.policy(2)
    artifact = load_residual_candidate(
        candidate_artifact_path,
        policy=candidate_policy,
        parent=parent,
        expected_body_hash=qualification.body_hash,
    )
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in suite_shard):
        raise ValueError(
            "suite shard may contain only lowercase letters, numbers, dash, underscore"
        )
    scenarios = _evaluation_scenarios(counts, suite_shard=suite_shard)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination.with_name(destination.name + ".checkpoint")
    checkpoint_identity = canonical_hash(
        {
            "candidate_policy_hash": candidate_policy.version_hash,
            "counts": counts.to_dict(),
            "suite_shard": suite_shard,
        }
    )
    rows, replay_checks = _load_checkpoint(
        checkpoint_path,
        expected_identity=checkpoint_identity,
    )
    completed = {row.scenario_commitment for row in rows if row.arm == "active_parent_v2"}
    for index, (partition_name, scenario) in enumerate(scenarios):
        if scenario.scenario_commitment in completed:
            if progress is not None:
                progress(index + 1, len(scenarios))
            continue
        scenario_rows, replay = _run_matched_scenario(
            backend=backend,
            scenario=scenario,
            partition_name=partition_name,
            parent=parent,
            candidate=artifact,
            replay_candidate=partition_name not in replay_checks,
        )
        rows.extend(scenario_rows)
        if replay is not None:
            replay_checks[partition_name] = replay
        _write_checkpoint(
            checkpoint_path,
            identity=checkpoint_identity,
            rows=rows,
            replay_checks=replay_checks,
            complete=False,
        )
        if progress is not None:
            progress(index + 1, len(scenarios))
    aggregates = _aggregate(rows)
    paired = _paired_statistics(rows)
    gate = _gate(
        rows=rows,
        paired=paired,
        replay_checks=replay_checks,
        counts=counts,
        training_seed_count=training_seed_count,
    )
    result = CandidateMatchedEvaluation(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        candidate_policy=candidate_policy.to_dict(),
        parent_policy=parent.to_dict(),
        counts=counts.to_dict(),
        training_seed_count=training_seed_count,
        rows=tuple(rows),
        aggregates=aggregates,
        paired_statistics=paired,
        gate=gate,
        replay_checks=replay_checks,
    )
    _atomic_json(destination, result.to_dict())
    _write_checkpoint(
        checkpoint_path,
        identity=checkpoint_identity,
        rows=rows,
        replay_checks=replay_checks,
        complete=True,
    )
    return result


def _evaluation_scenarios(
    counts: CandidateEvaluationCounts,
    *,
    suite_shard: str = "",
) -> tuple[tuple[str, GoalForgeScenario], ...]:
    specifications = (
        ("recent", Partition.DEVELOPMENT, counts.recent, 9),
        ("anchor", Partition.VALIDATION, counts.anchor, 0),
        ("boundary", Partition.COUNTEREXAMPLE_REGRESSION, counts.boundary, 10),
        ("self", Partition.STRESS, counts.self_partition, 8),
    )
    result: list[tuple[str, GoalForgeScenario]] = []
    for name, partition, count, generation in specifications:
        if count == 0:
            continue
        ledger = SeedLedger(
            task_id="g1_penalty_kick",
            secret=(
                _SUITE_SECRET
                + b":"
                + name.encode("ascii")
                + (b":" + suite_shard.encode("ascii") if suite_shard else b"")
            ),
        )
        generated = generate_goalforge_scenarios(
            ledger=ledger,
            partition=partition,
            count=count,
            generation=generation,
        )
        result.extend(
            (
                name,
                replace(
                    scenario,
                    scenario_id=(
                        f"g1-candidate-eval-{name}"
                        + (f"-{suite_shard}" if suite_shard else "")
                        + f"-{index:04d}"
                    ),
                ),
            )
            for index, scenario in enumerate(generated)
        )
    return tuple(result)


def merge_g1_candidate_evaluations(
    *,
    shard_paths: Sequence[Path],
    output_path: Path,
    source_checkout: Path,
) -> CandidateMatchedEvaluation:
    """Merge resumable disjoint shards and recompute every aggregate and gate."""

    if len(shard_paths) < 2:
        raise ValueError("candidate evaluation merge requires at least two shards")
    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("candidate evaluation evidence must be outside the source checkout")
    if destination.exists():
        raise FileExistsError(f"candidate evaluation output already exists: {destination}")
    values = [
        json.loads(path.expanduser().resolve().read_text(encoding="utf-8")) for path in shard_paths
    ]
    identity_fields = (
        "body_hash",
        "kick_prior_hash",
        "backend_commit",
        "candidate_policy",
        "parent_policy",
    )
    first = values[0]
    for value in values[1:]:
        if any(value[field] != first[field] for field in identity_fields):
            raise ValueError("candidate evaluation shard identities do not match")
    rows = tuple(CandidateEvaluationRow(**row) for value in values for row in value["rows"])
    commitments = [row.scenario_commitment for row in rows if row.arm == "active_parent_v2"]
    if len(commitments) != len(set(commitments)):
        raise ValueError("candidate evaluation shards contain duplicate scenarios")
    counts = CandidateEvaluationCounts(
        recent=sum(int(value["counts"]["recent"]) for value in values),
        anchor=sum(int(value["counts"]["anchor"]) for value in values),
        boundary=sum(int(value["counts"]["boundary"]) for value in values),
        self_partition=sum(int(value["counts"]["self"]) for value in values),
    )
    replay_checks: dict[str, bool] = {}
    for value in values:
        replay_checks.update({str(key): bool(item) for key, item in value["replay_checks"].items()})
    paired = _paired_statistics(rows)
    training_seed_count = max(int(value["training_seed_count"]) for value in values)
    result = CandidateMatchedEvaluation(
        body_hash=str(first["body_hash"]),
        kick_prior_hash=str(first["kick_prior_hash"]),
        backend_commit=str(first["backend_commit"]),
        candidate_policy=dict(first["candidate_policy"]),
        parent_policy=dict(first["parent_policy"]),
        counts=counts.to_dict(),
        training_seed_count=training_seed_count,
        rows=rows,
        aggregates=_aggregate(rows),
        paired_statistics=paired,
        gate=_gate(
            rows=rows,
            paired=paired,
            replay_checks=replay_checks,
            counts=counts,
            training_seed_count=training_seed_count,
        ),
        replay_checks=replay_checks,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, result.to_dict())
    return result


def _run_matched_scenario(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    partition_name: str,
    parent: PolicyVersion,
    candidate: ResidualCandidateArtifact,
    replay_candidate: bool,
) -> tuple[tuple[CandidateEvaluationRow, ...], bool | None]:
    parameters = ShotParameters()
    prior = backend.run(scenario, parameters)
    parent_runtime = _zero_runtime(candidate)
    parent_episode = backend.run(scenario, parameters, feedback_runtime=parent_runtime)
    candidate_runtime, candidate_policy = build_g1_candidate_runtime(
        candidate,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )
    candidate_episode = backend.run(scenario, parameters, feedback_runtime=candidate_runtime)
    inference_receipt = candidate_policy.build_receipt()
    fresh_runtime = _fresh_runtime(candidate, seed=scenario.seed ^ 0xF35A71)
    fresh = backend.run(scenario, parameters, feedback_runtime=fresh_runtime)
    parent_matches_prior = bool(
        parent_episode.result.summary_dict() == prior.result.summary_dict()
        and _motion_hash(parent_episode) == _motion_hash(prior)
    )
    candidate_replay: bool | None = None
    if replay_candidate:
        replay_runtime, replay_policy = build_g1_candidate_runtime(
            candidate,
            compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
        )
        replay = backend.run(scenario, parameters, feedback_runtime=replay_runtime)
        candidate_replay = bool(
            replay.result.summary_dict() == candidate_episode.result.summary_dict()
            and _motion_hash(replay) == _motion_hash(candidate_episode)
            and replay_policy.build_receipt().trace_hash == inference_receipt.trace_hash
        )
    return (
        (
            _row(
                scenario,
                partition_name,
                "frozen_prior",
                prior,
                version=0,
                version_hash=canonical_hash(
                    {"frozen_robonaldo_prior": candidate.parent.artifact_hash}
                ),
            ),
            _row(
                scenario,
                partition_name,
                "active_parent_v2",
                parent_episode,
                version=parent.version,
                version_hash=parent.version_hash,
            ),
            _row(
                scenario,
                partition_name,
                "candidate_v3",
                candidate_episode,
                version=candidate.policy.version,
                version_hash=candidate.policy.version_hash,
                inference_receipt=inference_receipt.to_dict(),
                inference_receipt_hash=inference_receipt.receipt_hash,
            ),
            _row(
                scenario,
                partition_name,
                "fresh_network",
                fresh,
                version=None,
                version_hash=fresh_runtime.controller.controller_hash,
            ),
        ),
        candidate_replay if parent_matches_prior else False,
    )


def _row(
    scenario: GoalForgeScenario,
    partition_name: str,
    arm: str,
    episode: GoalForgeEpisode,
    *,
    version: int | None,
    version_hash: str,
    inference_receipt: Mapping[str, Any] | None = None,
    inference_receipt_hash: str | None = None,
) -> CandidateEvaluationRow:
    result = episode.result
    target_error = result.target_error_m if math.isfinite(result.target_error_m) else None
    tracking = np.asarray(episode.trajectory["policy_action"], dtype=np.float64) - np.asarray(
        episode.trajectory["joint_position"], dtype=np.float64
    )
    torque = np.asarray(episode.trajectory["joint_torque"], dtype=np.float64)
    return CandidateEvaluationRow(
        scenario_id=scenario.scenario_id,
        scenario_commitment=scenario.scenario_commitment,
        replay_partition=partition_name,
        arm=arm,
        policy_version=version,
        policy_version_hash=version_hash,
        status=result.status.value,
        success=result.success,
        contact=result.kick_foot_contacted,
        goal_crossed=result.goal_crossed,
        penalized_target_error_m=(
            target_error if result.success and target_error is not None else _FAILURE_TARGET_ERROR_M
        ),
        conditional_target_error_m=target_error,
        ball_speed_mps=result.ball_speed_mps,
        fall=result.post_kick_fall,
        joint_violation=result.joint_limit_violation,
        torque_violation=result.torque_limit_violation,
        actuator_saturation=result.actuator_saturation,
        # The backend leaves this accumulator at +inf when an episode never enters
        # a measurable single-support window.  Encode that missing safety margin
        # conservatively instead of leaking a non-JSON IEEE sentinel into evidence.
        com_margin_min_m=(
            result.com_margin_min_m if math.isfinite(result.com_margin_min_m) else -1.0
        ),
        support_slip_m=result.support_foot_slip_m,
        tracking_rms_rad=float(np.sqrt(np.mean(np.square(tracking)))),
        energy_proxy=float(np.sum(np.square(torque)) * 0.02),
        action_drift_rms=(
            float(inference_receipt.get("action_rms", 0.0)) if inference_receipt else 0.0
        ),
        trajectory_hash=_motion_hash(episode),
        inference_receipt_hash=inference_receipt_hash,
        version_switch_count=(
            int(inference_receipt.get("version_switch_count", 0)) if inference_receipt else 0
        ),
    )


def _zero_runtime(candidate: ResidualCandidateArtifact) -> FeedbackRuntime:
    controller = ZeroResidualController()
    spec = FeedbackLoopSpec(
        loop_id="g1/goalforge-parent-v2",
        body_hash=candidate.policy.body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=("torso_roll", "torso_pitch", "com_y_relative"),
        observation_signals=candidate.observation_names,
        output_limits={
            _ACTION_OUTPUTS[name]: limit
            for name, limit in zip(candidate.action_names, candidate.action_limits, strict=True)
        },
        rate_hz=100.0,
        deadline_ms=10.0,
        max_observation_age_ms=20.0,
    )
    return FeedbackRuntime(
        spec=spec,
        controller=controller,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )


class _FreshResidualController:
    def __init__(self, candidate: ResidualCandidateArtifact, *, seed: int) -> None:
        self._candidate = candidate
        rng = np.random.default_rng(seed)
        h0, h1 = candidate.hidden_dims
        self._weights = (
            rng.normal(0.0, 0.08, (h0, len(candidate.observation_names))),
            rng.normal(0.0, 0.08, (h1, h0)),
            rng.normal(0.0, 0.04, (len(candidate.action_names), h1)),
        )
        self._biases = (np.zeros(h0), np.zeros(h1), np.zeros(len(candidate.action_names)))
        self._seed = seed

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        return None

    def compute(
        self, frame: FeedbackFrame, base_action: Mapping[str, float]
    ) -> Mapping[str, float]:
        del base_action
        value = np.asarray(
            [frame.actual[name] for name in self._candidate.observation_names], dtype=np.float64
        )
        value = np.maximum(self._weights[0] @ value + self._biases[0], 0.0)
        value = np.maximum(self._weights[1] @ value + self._biases[1], 0.0)
        value = np.tanh(self._weights[2] @ value + self._biases[2]) * np.asarray(
            self._candidate.action_limits
        )
        return {
            _ACTION_OUTPUTS[name]: float(item)
            for name, item in zip(self._candidate.action_names, value, strict=True)
        }

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "fresh_untrained_residual_control",
            "seed": self._seed,
            "candidate_contract_hash": self._candidate.policy.version_hash,
        }


def _fresh_runtime(candidate: ResidualCandidateArtifact, *, seed: int) -> FeedbackRuntime:
    controller = _FreshResidualController(candidate, seed=seed)
    spec = FeedbackLoopSpec(
        loop_id="g1/goalforge-fresh-network-control",
        body_hash=candidate.policy.body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=("torso_roll", "torso_pitch", "com_y_relative"),
        observation_signals=candidate.observation_names,
        output_limits={
            _ACTION_OUTPUTS[name]: limit
            for name, limit in zip(candidate.action_names, candidate.action_limits, strict=True)
        },
        rate_hz=100.0,
        deadline_ms=10.0,
        max_observation_age_ms=20.0,
    )
    return FeedbackRuntime(
        spec=spec,
        controller=controller,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )


def _motion_hash(episode: GoalForgeEpisode) -> str:
    digest = hashlib.sha256()
    ignored = {
        "feedback_residual",
        "feedback_error_rms",
        "feedback_active",
        "feedback_phase_rate",
    }
    for name in sorted(set(episode.trajectory).difference(ignored)):
        value = np.ascontiguousarray(episode.trajectory[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _aggregate(rows: Sequence[CandidateEvaluationRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    arms = sorted({row.arm for row in rows})
    partitions = ("all", "recent", "anchor", "boundary", "self")
    for arm in arms:
        result[arm] = {}
        for partition in partitions:
            selected = [
                row
                for row in rows
                if row.arm == arm and (partition == "all" or row.replay_partition == partition)
            ]
            if not selected:
                continue
            result[arm][partition] = {
                "episodes": len(selected),
                "success_rate": _mean(row.success for row in selected),
                "contact_rate": _mean(row.contact for row in selected),
                "goal_crossing_rate": _mean(row.goal_crossed for row in selected),
                "mean_penalized_target_error_m": _mean(
                    row.penalized_target_error_m for row in selected
                ),
                "mean_ball_speed_mps": _mean(row.ball_speed_mps for row in selected),
                "fall_rate": _mean(row.fall for row in selected),
                "joint_violation_rate": _mean(row.joint_violation for row in selected),
                "torque_violation_rate": _mean(row.torque_violation for row in selected),
                "actuator_saturation_rate": _mean(row.actuator_saturation for row in selected),
                "mean_com_margin_min_m": _mean(row.com_margin_min_m for row in selected),
                "mean_support_slip_m": _mean(row.support_slip_m for row in selected),
                "mean_tracking_rms_rad": _mean(row.tracking_rms_rad for row in selected),
                "mean_energy_proxy": _mean(row.energy_proxy for row in selected),
                "anchor_action_drift_rms": _mean(row.action_drift_rms for row in selected),
            }
    return result


def _paired_statistics(rows: Sequence[CandidateEvaluationRow]) -> dict[str, Any]:
    parent = {row.scenario_commitment: row for row in rows if row.arm == "active_parent_v2"}
    candidate = {row.scenario_commitment: row for row in rows if row.arm == "candidate_v3"}
    keys = sorted(set(parent) & set(candidate))
    success_delta = np.asarray(
        [float(candidate[key].success) - float(parent[key].success) for key in keys]
    )
    error_improvement = np.asarray(
        [
            parent[key].penalized_target_error_m - candidate[key].penalized_target_error_m
            for key in keys
        ]
    )
    return {
        "paired_scenarios": len(keys),
        "success_rate_delta": float(np.mean(success_delta)),
        "success_rate_delta_ci95": list(_bootstrap_ci(success_delta, seed=71001)),
        "target_error_improvement_m": float(np.mean(error_improvement)),
        "target_error_improvement_ci95": list(_bootstrap_ci(error_improvement, seed=71002)),
    }


def _gate(
    *,
    rows: Sequence[CandidateEvaluationRow],
    paired: Mapping[str, Any],
    replay_checks: Mapping[str, bool],
    counts: CandidateEvaluationCounts,
    training_seed_count: int,
) -> dict[str, Any]:
    parent = {row.scenario_commitment: row for row in rows if row.arm == "active_parent_v2"}
    candidate = {row.scenario_commitment: row for row in rows if row.arm == "candidate_v3"}
    common = set(parent) & set(candidate)
    critical_regressions = sum(
        (candidate[key].fall or candidate[key].joint_violation or candidate[key].torque_violation)
        and not (parent[key].fall or parent[key].joint_violation or parent[key].torque_violation)
        for key in common
    )
    parent_anchor = [row.success for row in parent.values() if row.replay_partition == "anchor"]
    candidate_anchor = [
        row.success for row in candidate.values() if row.replay_partition == "anchor"
    ]
    anchor_delta = (
        _mean(candidate_anchor) - _mean(parent_anchor)
        if candidate_anchor and parent_anchor
        else None
    )
    target_ci = paired["target_error_improvement_ci95"]
    checks = {
        "minimum_250_scenarios": counts.total >= 250,
        "minimum_8_training_seeds": training_seed_count >= 8,
        "four_partition_replay": set(replay_checks) == {"recent", "anchor", "boundary", "self"}
        and all(replay_checks.values()),
        "motion_version_switches_zero": all(row.version_switch_count == 0 for row in rows),
        "critical_regression_zero": critical_regressions == 0,
        "historical_mean_degradation_lt_3pct": bool(
            anchor_delta is not None and anchor_delta > -0.03
        ),
        "critical_skill_degradation_lte_5pct": bool(
            float(paired.get("success_rate_delta", float("-inf"))) >= -0.05
        ),
        "candidate_target_improvement_ci": target_ci[0] > 0.02,
    }
    return {
        "decision": "SIM_CHAMPION" if all(checks.values()) else "REJECTED",
        "candidate_motion_effect_proven": all(checks.values()),
        "checks": checks,
        "critical_safety_regressions": critical_regressions,
        "anchor_success_delta": anchor_delta,
        "minimum_target_error_improvement_m": 0.02,
        "candidate_activated": False,
        "hardware_authorized": False,
    }


def _bootstrap_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("paired confidence interval requires values")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(5000, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _mean(values: Iterable[float]) -> float:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError("mean requires at least one value")
    return float(sum(resolved) / len(resolved))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
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


def _load_checkpoint(
    path: Path,
    *,
    expected_identity: str,
) -> tuple[list[CandidateEvaluationRow], dict[str, bool]]:
    if not path.is_file():
        return [], {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("identity") != expected_identity:
        raise ValueError("candidate evaluation checkpoint identity does not match request")
    rows = []
    for serialized in value.get("rows", []):
        row = dict(serialized)
        # Checkpoints written before the finite-evidence invariant could contain
        # the backend's +inf sentinel for an unobserved single-support margin.
        # Migrate only that known field; CandidateEvaluationRow still rejects any
        # other non-finite value before the checkpoint can be rewritten.
        if not math.isfinite(float(row["com_margin_min_m"])):
            row["com_margin_min_m"] = -1.0
        rows.append(CandidateEvaluationRow(**row))
    replay = {str(key): bool(item) for key, item in value.get("replay_checks", {}).items()}
    return rows, replay


def _write_checkpoint(
    path: Path,
    *,
    identity: str,
    rows: Sequence[CandidateEvaluationRow],
    replay_checks: Mapping[str, bool],
    complete: bool,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "rosclaw.continual.g1_candidate_evaluation_checkpoint.v1",
            "identity": identity,
            "complete": complete,
            "completed_scenarios": sum(row.arm == "active_parent_v2" for row in rows),
            "rows": [asdict(row) for row in rows],
            "replay_checks": dict(replay_checks),
        },
    )


__all__ = [
    "CandidateEvaluationCounts",
    "CandidateEvaluationRow",
    "CandidateMatchedEvaluation",
    "merge_g1_candidate_evaluations",
    "run_g1_candidate_matched_evaluation",
]
