"""Rollout-driven training and matched qualification for G1 muscle memory."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.feedback.runtime import FeedbackRuntime
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
)
from rosclaw.simforge.g1_recovery_quality import (
    G1RecoveryQuality,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_TRAINING_SECRET = b"rosclaw-g1-post-kick-muscle-memory-v1"
_GENOME_SIZE = 33
_RECORDED_LATENCY_NS = 100_000
_CLOCK_CAPACITY = 5000
_RECOVERY_EQUIVALENCE_MARGIN = 0.03
# Biases of the seven posture/capture synergies.  The trainer probes both
# directions with real rollouts before sampling any multivariate candidate.
_PROBE_DIMENSIONS = (7, 12, 16, 20, 23, 26)
_DURATION_CANDIDATES_SEC = (0.7, 0.8, 0.9)
_SAGITTAL_BIAS_CANDIDATES = (0.03, 0.05, 0.25)
_IMPULSE_THRESHOLD_CANDIDATES_NS = (0.0, 1.75, 2.25)


@dataclass(frozen=True)
class G1MuscleMemoryTrainingConfig:
    population_size: int = 16
    generations: int = 5
    elite_fraction: float = 0.25
    initial_std: float = 0.12
    minimum_std: float = 0.015
    coordinate_probe: float = 0.15
    seed: int = 20260730

    def __post_init__(self) -> None:
        if not 6 <= self.population_size <= 128:
            raise ValueError("muscle-memory population must be in [6, 128]")
        if not 1 <= self.generations <= 40:
            raise ValueError("muscle-memory generations must be in [1, 40]")
        if not 0.10 <= self.elite_fraction <= 0.50:
            raise ValueError("muscle-memory elite fraction must be in [0.10, 0.50]")
        if not 0.05 <= self.initial_std <= 2.0:
            raise ValueError("muscle-memory initial std must be in [0.05, 2]")
        if not 0.01 <= self.minimum_std <= self.initial_std:
            raise ValueError("muscle-memory minimum std is invalid")
        if not 0.02 <= self.coordinate_probe <= 0.50:
            raise ValueError("muscle-memory coordinate probe must be in [0.02, 0.50]")
        if self.seed < 0:
            raise ValueError("muscle-memory seed must be non-negative")


@dataclass(frozen=True)
class G1MuscleMemoryCase:
    name: str
    scenario: GoalForgeScenario
    parameters: ShotParameters
    feedback_enabled: bool = False


@dataclass(frozen=True)
class G1MuscleMemoryCaseResult:
    name: str
    parent_result: dict[str, Any]
    candidate_result: dict[str, Any]
    parent_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    score: float
    safe: bool
    goal_preserved: bool
    naturalness_preserved: bool
    strict_replay: bool


@dataclass(frozen=True)
class G1MuscleMemoryHoldoutSummary:
    suite_hash: str
    case_count: int
    passed_count: int
    strict_replay_count: int
    mean_score: float
    minimum_score: float
    qualified: bool
    recovery_equivalence_margin: float = _RECOVERY_EQUIVALENCE_MARGIN
    case_rows_disclosed: bool = False
    schema_version: str = "rosclaw.g1_goalforge.muscle_memory_holdout_summary.v1"


@dataclass(frozen=True)
class G1MuscleMemoryTrainingReport:
    artifact: G1MuscleMemoryArtifact
    cases: tuple[G1MuscleMemoryCaseResult, ...]
    generation_best_scores: tuple[float, ...]
    baseline_score: float
    candidate_score: float
    candidate_selection_score: float
    candidate_worst_case_score: float
    training_rollout_count: int
    holdout_rollout_count: int
    holdout: G1MuscleMemoryHoldoutSummary
    qualified: bool
    rejection_reasons: tuple[str, ...]
    recovery_equivalence_margin: float = _RECOVERY_EQUIVALENCE_MARGIN
    evidence_domain: str = "SIM"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.muscle_memory_training.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": {
                **self.artifact.to_dict(),
                "artifact_hash": self.artifact.artifact_hash,
            },
            "cases": [asdict(item) for item in self.cases],
            "generation_best_scores": list(self.generation_best_scores),
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "candidate_selection_score": self.candidate_selection_score,
            "candidate_worst_case_score": self.candidate_worst_case_score,
            "training_rollout_count": self.training_rollout_count,
            "holdout_rollout_count": self.holdout_rollout_count,
            "holdout": asdict(self.holdout),
            "qualified": self.qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "recovery_equivalence_margin": self.recovery_equivalence_margin,
            "evidence_domain": self.evidence_domain,
            "hardware_command_sent": self.hardware_command_sent,
        }


def g1_muscle_memory_parent_config() -> G1CerebellarRecoveryConfig:
    """Stable two-stage parent shared across training and matched evaluation."""

    return G1CerebellarRecoveryConfig(
        start_policy_frame=300,
        blend_frames=100,
        standing_pose_blend=0.30,
        roll_posture_bias_rad=-0.05,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.45,
        settling_roll_posture_bias_rad=-0.02,
        settling_waist_pitch_bias_rad=0.09,
        target_smoothing_alpha=0.60,
        target_smoothing_start_policy_frame=300,
        target_smoothing_joint_group="upper_body",
    )


def build_g1_muscle_memory_cases() -> tuple[G1MuscleMemoryCase, ...]:
    base = generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_TRAINING_SECRET),
        partition=Partition.DEVELOPMENT,
        count=1,
        generation=7,
    )[0]
    base = replace(
        base,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_mass_kg=0.41,
        ball_ground_friction=0.05,
        support_ground_friction=1.0,
        restitution=0.55,
        disturbance_n=0.0,
        control_latency_ms=0.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        target_y_m=0.0,
        target_z_m=0.20,
        reachable=True,
    )
    static = replace(
        base,
        scenario_id="muscle-memory-static-high",
        ball_y_m=0.10,
        target_y_m=0.0,
        target_z_m=0.55,
    )
    moving = replace(
        base,
        scenario_id="muscle-memory-moving-ball",
        ball_x_m=1.12,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.08,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=4.0,
        ball_ground_friction=0.03,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    moving_plan = MovingBallInterceptAdapter().plan(moving)
    if not moving_plan.eligible:
        raise RuntimeError("muscle-memory moving-ball training case is ineligible")
    disturbed = replace(
        base,
        scenario_id="muscle-memory-disturbed-80n",
        disturbance_n=80.0,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    return (
        G1MuscleMemoryCase(
            name="static_high",
            scenario=static,
            parameters=ShotParameters(
                stance_offset_y=0.12,
                pelvis_yaw_offset=-0.20,
                foot_yaw_offset=-0.06,
                swing_amplitude=0.90,
                contact_phase_offset=0.08,
                policy_type="parameter",
            ),
        ),
        G1MuscleMemoryCase(
            name="moving_ball",
            scenario=moving,
            parameters=moving_plan.parameters,
        ),
        G1MuscleMemoryCase(
            name="disturbed_80n",
            scenario=disturbed,
            parameters=ShotParameters(
                stance_offset_y=-0.08,
                swing_amplitude=1.125,
                policy_type="parameter",
            ),
            feedback_enabled=True,
        ),
    )


def _build_g1_muscle_memory_holdout_cases() -> tuple[G1MuscleMemoryCase, ...]:
    base = generate_goalforge_scenarios(
        ledger=SeedLedger(
            task_id="g1_penalty_kick",
            secret=b"rosclaw-g1-post-kick-muscle-memory-private-holdout-v2",
        ),
        partition=Partition.HOLDOUT,
        count=1,
        generation=10,
    )[0]
    base = replace(
        base,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_mass_kg=0.41,
        ball_ground_friction=0.05,
        support_ground_friction=0.98,
        restitution=0.55,
        disturbance_n=0.0,
        control_latency_ms=2.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        target_y_m=0.0,
        target_z_m=0.20,
        reachable=True,
    )
    static = replace(
        base,
        scenario_id="private-holdout-v2-static-high-offset-ball",
        ball_y_m=-0.05,
        target_y_m=0.0,
        target_z_m=0.55,
    )
    moving = replace(
        base,
        scenario_id="private-holdout-v2-moving-faster",
        ball_x_m=1.12,
        ball_velocity_x_mps=-0.12,
        ball_launch_delay_sec=4.0,
        ball_ground_friction=0.03,
    )
    moving_plan = MovingBallInterceptAdapter().plan(moving)
    if not moving_plan.eligible:
        raise RuntimeError("private moving-ball holdout is ineligible")
    disturbed = replace(
        base,
        scenario_id="private-holdout-v2-disturbed-70n",
        disturbance_n=70.0,
    )
    return (
        G1MuscleMemoryCase(
            name="private_v2_static_high_offset_ball",
            scenario=static,
            parameters=ShotParameters(
                stance_offset_y=0.08,
                pelvis_yaw_offset=-0.12,
                foot_yaw_offset=-0.04,
                swing_amplitude=0.90,
                contact_phase_offset=0.06,
                policy_type="parameter",
            ),
        ),
        G1MuscleMemoryCase(
            name="private_v2_moving_faster",
            scenario=moving,
            parameters=moving_plan.parameters,
        ),
        G1MuscleMemoryCase(
            name="private_v2_disturbed_70n",
            scenario=disturbed,
            parameters=ShotParameters(
                stance_offset_y=0.08,
                swing_amplitude=1.08,
                policy_type="parameter",
            ),
            feedback_enabled=True,
        ),
    )


class G1MuscleMemoryTrainer:
    """Cross-entropy policy search over a sparse proprioceptive actor."""

    def __init__(
        self,
        *,
        asset_root: Path,
        cases: tuple[G1MuscleMemoryCase, ...] | None = None,
        recovery_config: G1CerebellarRecoveryConfig | None = None,
        config: G1MuscleMemoryTrainingConfig | None = None,
    ) -> None:
        self.backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
        self.cases = cases or build_g1_muscle_memory_cases()
        if not self.cases:
            raise ValueError("muscle-memory training requires at least one case")
        self.recovery_config = recovery_config or g1_muscle_memory_parent_config()
        self.config = config or G1MuscleMemoryTrainingConfig()

    def train(self) -> G1MuscleMemoryTrainingReport:
        parents = tuple(self._run_parent(case) for case in self.cases)
        parent_metrics = tuple(measure_g1_recovery_quality(item.trajectory) for item in parents)
        mean, scale = _normalization(parents)
        dataset_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.muscle_memory_dataset.v1",
                "cases": [
                    {
                        "name": case.name,
                        "scenario_commitment": case.scenario.scenario_commitment,
                        "policy_hash": case.parameters.policy_hash,
                        "trajectory_hash": trajectory_digest(parent.trajectory),
                    }
                    for case, parent in zip(self.cases, parents, strict=True)
                ],
                "observation_mean": list(mean),
                "observation_scale": list(scale),
            }
        )
        parent_controller = self.backend.build_cerebellar_recovery_controller(
            self.cases[0].scenario,
            self.recovery_config,
        )
        rng = np.random.default_rng(self.config.seed)
        distribution_mean = np.zeros(_GENOME_SIZE, dtype=np.float64)
        distribution_std = np.full(_GENOME_SIZE, self.config.initial_std, dtype=np.float64)
        zero = distribution_mean.copy()
        best_genome = zero.copy()
        best_score = self._evaluate_genome(
            zero,
            parents=parents,
            parent_metrics=parent_metrics,
            observation_mean=mean,
            observation_scale=scale,
            dataset_hash=dataset_hash,
            parent_config_hash=parent_controller.config_hash,
        )[0]
        baseline_score = best_score
        rollout_count = len(self.cases)
        for duration_sec in _DURATION_CANDIDATES_SEC:
            for sagittal_bias in _SAGITTAL_BIAS_CANDIDATES:
                for impulse_threshold_ns in _IMPULSE_THRESHOLD_CANDIDATES_NS:
                    probe = zero.copy()
                    probe[3] = sagittal_bias
                    probe[31] = (duration_sec - 1.6) / 3.0
                    probe[32] = (impulse_threshold_ns - 1.75) / 2.0
                    score, _ = self._evaluate_genome(
                        probe,
                        parents=parents,
                        parent_metrics=parent_metrics,
                        observation_mean=mean,
                        observation_scale=scale,
                        dataset_hash=dataset_hash,
                        parent_config_hash=parent_controller.config_hash,
                    )
                    rollout_count += len(self.cases)
                    if score > best_score:
                        best_score = score
                        best_genome = probe
        for index in _PROBE_DIMENSIONS:
            for direction in (-1.0, 1.0):
                probe = zero.copy()
                probe[index] = direction * self.config.coordinate_probe
                score, _ = self._evaluate_genome(
                    probe,
                    parents=parents,
                    parent_metrics=parent_metrics,
                    observation_mean=mean,
                    observation_scale=scale,
                    dataset_hash=dataset_hash,
                    parent_config_hash=parent_controller.config_hash,
                )
                rollout_count += len(self.cases)
                if score > best_score:
                    best_score = score
                    best_genome = probe
        distribution_mean = best_genome.copy()
        generation_best = []
        elite_count = max(
            2, int(math.ceil(self.config.population_size * self.config.elite_fraction))
        )
        for _generation in range(self.config.generations):
            population = rng.normal(
                distribution_mean,
                distribution_std,
                size=(self.config.population_size, _GENOME_SIZE),
            )
            population[0] = distribution_mean
            population[1] = zero
            scored: list[tuple[float, np.ndarray]] = []
            for genome in population:
                score, _ = self._evaluate_genome(
                    genome,
                    parents=parents,
                    parent_metrics=parent_metrics,
                    observation_mean=mean,
                    observation_scale=scale,
                    dataset_hash=dataset_hash,
                    parent_config_hash=parent_controller.config_hash,
                )
                scored.append((score, genome.copy()))
            rollout_count += len(population) * len(self.cases)
            scored.sort(key=lambda item: item[0], reverse=True)
            elites = np.stack([item[1] for item in scored[:elite_count]])
            distribution_mean = 0.25 * distribution_mean + 0.75 * np.mean(elites, axis=0)
            elite_std = np.std(elites, axis=0)
            distribution_std = np.maximum(
                self.config.minimum_std,
                0.35 * distribution_std + 0.65 * elite_std,
            )
            generation_best.append(scored[0][0])
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_genome = scored[0][1]
        artifact = _artifact_from_genome(
            best_genome,
            body_hash=self.backend.qualification.body_hash,
            motion_hash=self.backend.qualification.motion_hash,
            parent_config_hash=parent_controller.config_hash,
            dataset_hash=dataset_hash,
            observation_mean=mean,
            observation_scale=scale,
            training_episode_count=rollout_count,
            training_seed=self.config.seed,
        )
        candidate_selection_score, candidate_episodes = self._evaluate_artifact(
            artifact,
            parents=parents,
            parent_metrics=parent_metrics,
        )
        case_results = []
        reasons = []
        for case, parent, parent_quality, candidate in zip(
            self.cases,
            parents,
            parent_metrics,
            candidate_episodes,
            strict=True,
        ):
            replay = self._run_candidate(case, artifact)
            strict = bool(
                candidate.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(candidate.trajectory) == trajectory_digest(replay.trajectory)
            )
            candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=candidate_quality,
            )
            if not safe:
                reasons.append(case.name + ":safety_regressed")
            if not goal:
                reasons.append(case.name + ":goal_regressed")
            if not natural:
                reasons.append(case.name + ":naturalness_regressed")
            if not strict:
                reasons.append(case.name + ":strict_replay_failed")
            if score < -_RECOVERY_EQUIVALENCE_MARGIN:
                reasons.append(case.name + ":recovery_quality_regressed")
            case_results.append(
                G1MuscleMemoryCaseResult(
                    name=case.name,
                    parent_result=parent.result.summary_dict(),
                    candidate_result=candidate.result.summary_dict(),
                    parent_metrics=parent_quality.to_dict(),
                    candidate_metrics=candidate_quality.to_dict(),
                    score=score,
                    safe=safe,
                    goal_preserved=goal,
                    naturalness_preserved=natural,
                    strict_replay=strict,
                )
            )
        candidate_score = float(np.mean([item.score for item in case_results]))
        candidate_worst_case_score = float(min(item.score for item in case_results))
        if candidate_score <= baseline_score + 0.05:
            reasons.append("aggregate_improvement_below_gate")
        holdout, holdout_rollouts = self._evaluate_private_holdout(artifact)
        if not holdout.qualified:
            reasons.append("private_holdout_failed")
        return G1MuscleMemoryTrainingReport(
            artifact=artifact,
            cases=tuple(case_results),
            generation_best_scores=tuple(generation_best),
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            candidate_selection_score=candidate_selection_score,
            candidate_worst_case_score=candidate_worst_case_score,
            training_rollout_count=rollout_count + 2 * len(self.cases),
            holdout_rollout_count=holdout_rollouts,
            holdout=holdout,
            qualified=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def _evaluate_private_holdout(
        self,
        artifact: G1MuscleMemoryArtifact,
    ) -> tuple[G1MuscleMemoryHoldoutSummary, int]:
        cases = _build_g1_muscle_memory_holdout_cases()
        scores = []
        passed = 0
        strict_count = 0
        for case in cases:
            parent = self._run_parent(case)
            parent_quality = measure_g1_recovery_quality(parent.trajectory)
            candidate = self._run_candidate(case, artifact)
            candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            replay = self._run_candidate(case, artifact)
            strict = bool(
                candidate.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(candidate.trajectory) == trajectory_digest(replay.trajectory)
            )
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=candidate_quality,
            )
            strict_count += int(strict)
            passed += int(
                safe and goal and natural and strict and score >= -_RECOVERY_EQUIVALENCE_MARGIN
            )
            scores.append(score)
        suite_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.private_holdout_suite.v2",
                "case_commitments": [case.scenario.scenario_commitment for case in cases],
            }
        )
        return (
            G1MuscleMemoryHoldoutSummary(
                suite_hash=suite_hash,
                case_count=len(cases),
                passed_count=passed,
                strict_replay_count=strict_count,
                mean_score=float(np.mean(scores)),
                minimum_score=float(np.min(scores)),
                qualified=passed == len(cases),
            ),
            3 * len(cases),
        )

    def _run_parent(self, case: G1MuscleMemoryCase) -> GoalForgeEpisode:
        controller = self.backend.build_cerebellar_recovery_controller(
            case.scenario,
            self.recovery_config,
        )
        return self.backend.run(
            case.scenario,
            case.parameters,
            feedback_runtime=self._feedback_runtime(case),
            recovery_controller=controller,
        )

    def _run_candidate(
        self,
        case: G1MuscleMemoryCase,
        artifact: G1MuscleMemoryArtifact,
    ) -> GoalForgeEpisode:
        controller = self.backend.build_cerebellar_recovery_controller(
            case.scenario,
            self.recovery_config,
            artifact,
        )
        return self.backend.run(
            case.scenario,
            case.parameters,
            feedback_runtime=self._feedback_runtime(case),
            recovery_controller=controller,
        )

    def _feedback_runtime(self, case: G1MuscleMemoryCase) -> FeedbackRuntime | None:
        if not case.feedback_enabled:
            return None
        return build_g1_balance_runtime(
            body_hash=self.backend.qualification.body_hash,
            compute_clock_ns=RecordedLatencyClock((_RECORDED_LATENCY_NS,) * _CLOCK_CAPACITY),
        )

    def _evaluate_genome(
        self,
        genome: np.ndarray,
        *,
        parents: tuple[GoalForgeEpisode, ...],
        parent_metrics: tuple[G1RecoveryQuality, ...],
        observation_mean: tuple[float, ...],
        observation_scale: tuple[float, ...],
        dataset_hash: str,
        parent_config_hash: str,
    ) -> tuple[float, tuple[GoalForgeEpisode, ...]]:
        artifact = _artifact_from_genome(
            genome,
            body_hash=self.backend.qualification.body_hash,
            motion_hash=self.backend.qualification.motion_hash,
            parent_config_hash=parent_config_hash,
            dataset_hash=dataset_hash,
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            training_episode_count=1,
            training_seed=self.config.seed,
        )
        return self._evaluate_artifact(
            artifact,
            parents=parents,
            parent_metrics=parent_metrics,
        )

    def _evaluate_artifact(
        self,
        artifact: G1MuscleMemoryArtifact,
        *,
        parents: tuple[GoalForgeEpisode, ...],
        parent_metrics: tuple[G1RecoveryQuality, ...],
    ) -> tuple[float, tuple[GoalForgeEpisode, ...]]:
        candidates = tuple(self._run_candidate(case, artifact) for case in self.cases)
        scores = []
        valid = True
        for parent, parent_quality, candidate in zip(
            parents, parent_metrics, candidates, strict=True
        ):
            try:
                candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            except ValueError:
                return -1_000_000.0, candidates
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=candidate_quality,
            )
            valid = valid and safe and goal and natural
            scores.append(score)
        if not valid:
            return -1_000_000.0, candidates
        regressions = sum(max(0.0, -_RECOVERY_EQUIVALENCE_MARGIN - score) for score in scores)
        selection_score = float(np.mean(scores) - 25.0 * regressions)
        return selection_score, candidates


def write_g1_muscle_memory_report(
    report: G1MuscleMemoryTrainingReport,
    *,
    output_dir: Path,
    source_checkout: Path,
) -> tuple[Path, Path]:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("muscle-memory evidence must remain outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    artifact_path = root / "g1-muscle-memory.json"
    report_path = root / "g1-muscle-memory-training.json"
    _atomic_json(artifact_path, report.artifact.to_dict())
    _atomic_json(report_path, report.to_dict())
    return artifact_path, report_path


def _artifact_from_genome(
    genome: np.ndarray,
    *,
    body_hash: str,
    motion_hash: str,
    parent_config_hash: str,
    dataset_hash: str,
    observation_mean: tuple[float, ...],
    observation_scale: tuple[float, ...],
    training_episode_count: int,
    training_seed: int,
) -> G1MuscleMemoryArtifact:
    vector = np.asarray(genome, dtype=np.float64)
    if vector.shape != (_GENOME_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"muscle-memory genome must contain {_GENOME_SIZE} finite values")
    weights = np.zeros(
        (len(G1_MUSCLE_MEMORY_ACTIONS), len(G1_MUSCLE_MEMORY_OBSERVATIONS)),
        dtype=np.float64,
    )
    bias = np.zeros(len(G1_MUSCLE_MEMORY_ACTIONS), dtype=np.float64)
    obs = {name: index for index, name in enumerate(G1_MUSCLE_MEMORY_OBSERVATIONS)}
    cursor = 0

    def assign(action: str, features: tuple[str, ...], *, with_bias: bool = True) -> None:
        nonlocal cursor
        action_index = G1_MUSCLE_MEMORY_ACTIONS.index(action)
        for feature in features:
            weights[action_index, obs[feature]] = vector[cursor]
            cursor += 1
        if with_bias:
            bias[action_index] = vector[cursor]
            cursor += 1

    assign(
        "sagittal_common",
        (
            "pelvis_velocity_x_m_s",
            "torso_pitch_rad",
            "torso_angular_velocity_y_rad_s",
        ),
    )
    assign(
        "sagittal_split",
        ("pelvis_velocity_x_m_s", "left_support", "right_support"),
    )
    assign(
        "lateral_common",
        (
            "pelvis_velocity_y_m_s",
            "torso_roll_rad",
            "torso_angular_velocity_x_rad_s",
            "com_y_relative_m",
        ),
    )
    assign(
        "lateral_split",
        ("pelvis_velocity_y_m_s", "left_support", "right_support"),
    )
    assign(
        "leg_absorption",
        (
            "pelvis_velocity_z_m_s",
            "contact_impulse_ns",
            "post_contact_time_sec",
        ),
    )
    assign("waist_pitch", ("torso_pitch_rad", "torso_angular_velocity_y_rad_s"))
    assign("waist_roll", ("torso_roll_rad", "torso_angular_velocity_x_rad_s"))
    assign(
        "arm_pitch_counter",
        ("torso_angular_velocity_y_rad_s", "pelvis_velocity_x_m_s"),
        with_bias=False,
    )
    assign(
        "arm_roll_counter",
        ("torso_angular_velocity_x_rad_s", "pelvis_velocity_y_m_s"),
        with_bias=False,
    )
    if cursor != _GENOME_SIZE - 2:
        raise RuntimeError(f"muscle-memory genome mapping consumed {cursor} values")
    activation_duration_sec = float(np.clip(1.6 + 3.0 * vector[31], 0.4, 4.0))
    minimum_impulse_ns = float(np.clip(1.75 + 2.0 * vector[32], 0.0, 4.0))
    return G1MuscleMemoryArtifact(
        body_hash=body_hash,
        motion_hash=motion_hash,
        parent_recovery_config_hash=parent_config_hash,
        training_dataset_hash=dataset_hash,
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        weights=tuple(tuple(map(float, row)) for row in weights),
        bias=tuple(map(float, bias)),
        action_limits_rad=(0.055, 0.045, 0.050, 0.045, 0.050, 0.040, 0.040, 0.045, 0.045),
        training_episode_count=training_episode_count,
        training_seed=training_seed,
        activation_duration_sec=activation_duration_sec,
        fade_out_sec=min(0.4, activation_duration_sec / 2.0),
        sagittal_minimum_impulse_ns=minimum_impulse_ns,
    )


def _normalization(
    episodes: tuple[GoalForgeEpisode, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows = np.concatenate([_observation_rows(item) for item in episodes], axis=0)
    mean = np.mean(rows, axis=0)
    scale = np.std(rows, axis=0)
    scale = np.maximum(
        scale,
        np.asarray(
            (
                0.50,
                0.05,
                0.10,
                0.10,
                0.10,
                0.05,
                0.05,
                0.10,
                0.10,
                0.10,
                0.03,
                0.25,
                0.25,
                0.10,
                0.10,
                0.25,
            ),
            dtype=np.float64,
        ),
    )
    return tuple(map(float, mean)), tuple(map(float, scale))


def _observation_rows(episode: GoalForgeEpisode) -> np.ndarray:
    trace = episode.trajectory
    recorded = trace.get("recovery_proprioception")
    if recorded is not None:
        rows = np.asarray(recorded, dtype=np.float64)
        expected = (len(np.asarray(trace["time"])), len(G1_MUSCLE_MEMORY_OBSERVATIONS))
        if rows.shape != expected or not np.all(np.isfinite(rows)):
            raise ValueError("recorded recovery proprioception has an invalid shape")
        return rows
    time = np.asarray(trace["time"], dtype=np.float64)
    pelvis = np.asarray(trace["pelvis_pose"], dtype=np.float64)
    torso = np.asarray(trace["torso_quaternion"], dtype=np.float64)
    phase = np.asarray(trace["policy_phase"], dtype=np.float64)
    com_y = np.asarray(trace["com_y_relative"], dtype=np.float64)
    left = np.asarray(trace["left_foot_contact"], dtype=np.float64)
    right = np.asarray(trace["right_foot_contact"], dtype=np.float64)
    force = np.asarray(trace["ground_reaction_force"], dtype=np.float64) / 500.0
    impulse = np.asarray(trace["contact_impulse"], dtype=np.float64)
    roll, pitch = _roll_pitch(torso)
    pelvis_velocity = np.gradient(pelvis[:, :3], time, axis=0)
    roll_rate = np.gradient(np.unwrap(roll), time)
    pitch_rate = np.gradient(np.unwrap(pitch), time)
    yaw_rate = np.zeros_like(roll_rate)
    contacts = np.flatnonzero(np.diff(impulse, prepend=0.0) > 1e-9)
    contact_time = float(time[int(contacts[0])]) if len(contacts) else float(time[-1])
    post_contact = np.maximum(0.0, time - contact_time)
    return np.column_stack(
        (
            post_contact,
            phase,
            pelvis_velocity,
            roll,
            pitch,
            roll_rate,
            pitch_rate,
            yaw_rate,
            com_y,
            left,
            right,
            force,
            impulse,
        )
    )


def _case_score(
    *,
    parent: GoalForgeEpisode,
    parent_quality: G1RecoveryQuality,
    candidate: GoalForgeEpisode,
    candidate_quality: G1RecoveryQuality,
) -> tuple[float, bool, bool, bool]:
    result = candidate.result
    parent_result = parent.result
    safe = bool(
        not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.support_foot_slip_m <= 0.04
        and candidate_quality.terminal_bilateral_support
    )
    goal = bool(
        result.success
        and result.goal_crossed
        and result.target_zone_hit
        and result.target_error_m <= min(0.48, parent_result.target_error_m + 0.03)
        and result.ball_speed_mps >= 0.95 * parent_result.ball_speed_mps
    )
    natural = _naturalness_preserved(parent_quality, candidate_quality)
    reductions = (
        2.5
        * _reduction(
            parent_quality.post_contact_backward_reversal_m,
            candidate_quality.post_contact_backward_reversal_m,
            scale_floor=0.05,
        )
        + 2.0
        * _reduction(
            parent_quality.post_contact_lateral_peak_return_m,
            candidate_quality.post_contact_lateral_peak_return_m,
            scale_floor=0.05,
        )
        + 1.5
        * _reduction(
            parent_quality.post_contact_pelvis_path_length_m,
            candidate_quality.post_contact_pelvis_path_length_m,
            scale_floor=0.20,
        )
        + 1.0
        * _reduction(
            parent_quality.tail_wobble_index,
            candidate_quality.tail_wobble_index,
            scale_floor=0.05,
        )
        + 0.75
        * _reduction(
            parent_quality.post_contact_leg_joint_jerk_rms_rad_s3,
            candidate_quality.post_contact_leg_joint_jerk_rms_rad_s3,
            scale_floor=100.0,
        )
        + 0.75
        * _reduction(
            float(parent_quality.post_contact_support_transition_count),
            float(candidate_quality.post_contact_support_transition_count),
            scale_floor=2.0,
        )
        + 0.50
        * _reduction(
            parent_quality.settling_time_sec or 20.0,
            candidate_quality.settling_time_sec or 20.0,
            scale_floor=0.50,
        )
    )
    precision_penalty = max(
        0.0,
        result.target_error_m - parent_result.target_error_m,
    )
    return reductions - 2.0 * precision_penalty, safe, goal, natural


def _naturalness_preserved(
    parent: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
) -> bool:
    settling_preserved = bool(
        parent.settling_time_sec is None
        or (
            candidate.settling_time_sec is not None
            and candidate.settling_time_sec <= parent.settling_time_sec + 0.25
        )
    )
    return bool(
        candidate.post_contact_lateral_peak_return_m
        <= parent.post_contact_lateral_peak_return_m
        + max(0.02, 0.05 * parent.post_contact_lateral_peak_return_m)
        and candidate.post_contact_pelvis_path_length_m
        <= parent.post_contact_pelvis_path_length_m
        + max(0.05, 0.05 * parent.post_contact_pelvis_path_length_m)
        and candidate.tail_wobble_index
        <= parent.tail_wobble_index + max(0.01, 0.05 * parent.tail_wobble_index)
        and candidate.post_contact_leg_joint_jerk_rms_rad_s3
        <= 1.05 * parent.post_contact_leg_joint_jerk_rms_rad_s3 + 10.0
        and candidate.post_contact_support_transition_count
        <= parent.post_contact_support_transition_count + 2
        and settling_preserved
    )


def _reduction(parent: float, candidate: float, *, scale_floor: float) -> float:
    return (parent - candidate) / max(abs(parent), scale_floor)


def _roll_pitch(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = quaternion.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = [
    "G1MuscleMemoryCase",
    "G1MuscleMemoryCaseResult",
    "G1MuscleMemoryTrainer",
    "G1MuscleMemoryTrainingConfig",
    "G1MuscleMemoryTrainingReport",
    "build_g1_muscle_memory_cases",
    "g1_muscle_memory_parent_config",
    "write_g1_muscle_memory_report",
]
