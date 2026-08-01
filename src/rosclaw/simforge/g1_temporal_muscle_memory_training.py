"""Rollout training for the temporal G1 post-kick muscle-memory policy.

MuJoCo remains the source of physical truth.  The optimizer evolves a small
recurrent radial-basis policy from matched SIM rollouts, while optional CUDA
parity checks only verify that the exported NumPy actor agrees on every GPU.
No CUDA result is accepted as physics evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import trajectory_digest
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
)
from rosclaw.simforge.g1_muscle_memory_training import (
    G1MuscleMemoryCase,
    G1MuscleMemoryCaseResult,
    G1MuscleMemoryHoldoutSummary,
    G1MuscleMemoryTrainer,
    G1MuscleMemoryTrainingConfig,
    _case_score,
    _normalization,
    build_g1_muscle_memory_cases,
)
from rosclaw.simforge.g1_recovery_quality import (
    G1RecoveryQuality,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_TEMPORAL_GENOME_SIZE = 37
_BASIS_CENTERS_SEC = (0.0, 0.30, 0.60, 0.90)
_TEMPORAL_REDUCTION_FLOOR = 0.02
_MOVING_BACKWARD_REDUCTION_GATE = 0.10
_MOVING_WOBBLE_REDUCTION_GATE = 0.02
_MOVING_LEG_JERK_REDUCTION_GATE = 0.02
_RESIDUAL_INCREMENTAL_REGRESSION_LIMIT = 1e-6
# The v12 gate was 2e-4 on one moving rollout.  v2 reports the mean over three
# independently perturbed moving regimes, so it requires a still-positive
# 1e-4 aggregate effect plus a separate sealed generalization effect below.
_RESIDUAL_INCREMENTAL_EFFECT_GATE = 1e-4
_EXPERT_GENERALIZATION_REGRESSION_LIMIT = 0.03
_EXPERT_GENERALIZATION_EFFECT_GATE = 0.02
_EXPERT_GENERALIZATION_CAUSAL_EFFECT_GATE = 1e-6


@dataclass(frozen=True)
class G1TemporalMuscleMemoryTrainingConfig:
    population_size: int = 10
    generations: int = 3
    elite_fraction: float = 0.30
    initial_std: float = 0.08
    minimum_std: float = 0.015
    seed: int = 20260731
    cuda_devices: tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if not 6 <= self.population_size <= 64:
            raise ValueError("temporal muscle-memory population must be in [6, 64]")
        if not 1 <= self.generations <= 20:
            raise ValueError("temporal muscle-memory generations must be in [1, 20]")
        if not 0.20 <= self.elite_fraction <= 0.50:
            raise ValueError("temporal muscle-memory elite fraction must be in [0.20, 0.50]")
        if not 0.01 <= self.minimum_std <= self.initial_std <= 0.50:
            raise ValueError("temporal muscle-memory search standard deviation is invalid")
        if self.seed < 0:
            raise ValueError("temporal muscle-memory seed must be non-negative")
        if len(set(self.cuda_devices)) != len(self.cuda_devices) or any(
            device < 0 for device in self.cuda_devices
        ):
            raise ValueError("temporal muscle-memory CUDA devices must be unique and non-negative")


@dataclass(frozen=True)
class G1TemporalGpuParity:
    requested_devices: tuple[int, ...]
    validated_devices: tuple[int, ...]
    device_names: tuple[str, ...]
    maximum_absolute_error: float
    output_hashes: tuple[str, ...]
    passed: bool
    physics_authority: bool = False
    schema_version: str = "rosclaw.g1_goalforge.temporal_gpu_parity.v1"


@dataclass(frozen=True)
class G1TemporalRecoveryConfigTrial:
    config_hash: str
    settling_standing_pose_blend: float
    settling_waist_pitch_bias_rad: float
    target_smoothing_alpha: float
    score: float
    moving_backward_reduction: float
    moving_tail_wobble_reduction: float
    moving_leg_jerk_reduction: float
    safe: bool
    goal_preserved: bool
    naturalness_preserved: bool
    selected: bool = False
    schema_version: str = "rosclaw.g1_goalforge.temporal_recovery_config_trial.v1"


@dataclass(frozen=True)
class G1TemporalExpertGeneralizationSummary:
    """Aggregate-only evidence from moving cases excluded from policy search."""

    suite_hash: str
    case_count: int
    parent_valid_count: int
    expert_route_count: int
    strict_replay_count: int
    passed_count: int
    mean_score: float
    minimum_score: float
    moving_backward_reduction: float
    moving_tail_wobble_reduction: float
    moving_leg_jerk_reduction: float
    residual_backward_reduction: float
    residual_tail_wobble_reduction: float
    residual_leg_jerk_reduction: float
    qualified: bool
    development_search_excluded: bool = True
    case_rows_disclosed: bool = False
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw.g1_goalforge.temporal_expert_generalization.v1"


@dataclass(frozen=True)
class G1TemporalMuscleMemoryTrainingReport:
    artifact: G1MuscleMemoryArtifact
    cases: tuple[G1MuscleMemoryCaseResult, ...]
    generation_best_scores: tuple[float, ...]
    baseline_score: float
    candidate_score: float
    candidate_worst_case_score: float
    training_rollout_count: int
    holdout_rollout_count: int
    holdout: G1MuscleMemoryHoldoutSummary
    expert_generalization_rollout_count: int
    expert_generalization: G1TemporalExpertGeneralizationSummary
    gpu_parity: G1TemporalGpuParity
    structured_recovery_trials: tuple[G1TemporalRecoveryConfigTrial, ...]
    retained_recovery_config_hash: str
    temporal_recovery_config_hash: str
    temporal_recovery_config: dict[str, Any]
    moving_backward_reduction: float
    moving_tail_wobble_reduction: float
    moving_leg_jerk_reduction: float
    moving_structured_backward_reduction: float
    moving_structured_tail_wobble_reduction: float
    moving_structured_leg_jerk_reduction: float
    moving_residual_backward_reduction: float
    moving_residual_tail_wobble_reduction: float
    moving_residual_leg_jerk_reduction: float
    residual_causal_gate_passed: bool
    development_expert_route_count: int
    development_fallback_route_count: int
    holdout_expert_route_count: int
    holdout_fallback_route_count: int
    holdout_parent_valid_count: int
    qualified: bool
    rejection_reasons: tuple[str, ...]
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.temporal_muscle_memory_training.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": {**self.artifact.to_dict(), "artifact_hash": self.artifact.artifact_hash},
            "cases": [asdict(item) for item in self.cases],
            "generation_best_scores": list(self.generation_best_scores),
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "candidate_worst_case_score": self.candidate_worst_case_score,
            "training_rollout_count": self.training_rollout_count,
            "holdout_rollout_count": self.holdout_rollout_count,
            "holdout": asdict(self.holdout),
            "expert_generalization_rollout_count": self.expert_generalization_rollout_count,
            "expert_generalization": asdict(self.expert_generalization),
            "gpu_parity": asdict(self.gpu_parity),
            "structured_recovery_trials": [
                asdict(item) for item in self.structured_recovery_trials
            ],
            "retained_recovery_config_hash": self.retained_recovery_config_hash,
            "temporal_recovery_config_hash": self.temporal_recovery_config_hash,
            "temporal_recovery_config": self.temporal_recovery_config,
            "moving_backward_reduction": self.moving_backward_reduction,
            "moving_tail_wobble_reduction": self.moving_tail_wobble_reduction,
            "moving_leg_jerk_reduction": self.moving_leg_jerk_reduction,
            "moving_structured_backward_reduction": (self.moving_structured_backward_reduction),
            "moving_structured_tail_wobble_reduction": (
                self.moving_structured_tail_wobble_reduction
            ),
            "moving_structured_leg_jerk_reduction": self.moving_structured_leg_jerk_reduction,
            "moving_residual_backward_reduction": self.moving_residual_backward_reduction,
            "moving_residual_tail_wobble_reduction": self.moving_residual_tail_wobble_reduction,
            "moving_residual_leg_jerk_reduction": self.moving_residual_leg_jerk_reduction,
            "residual_causal_gate_passed": self.residual_causal_gate_passed,
            "development_expert_route_count": self.development_expert_route_count,
            "development_fallback_route_count": self.development_fallback_route_count,
            "holdout_expert_route_count": self.holdout_expert_route_count,
            "holdout_fallback_route_count": self.holdout_fallback_route_count,
            "holdout_parent_valid_count": self.holdout_parent_valid_count,
            "qualified": self.qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_domain": self.evidence_domain,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
            "qualification_thresholds": {
                "moving_backward_reduction": _MOVING_BACKWARD_REDUCTION_GATE,
                "moving_tail_wobble_reduction": _MOVING_WOBBLE_REDUCTION_GATE,
                "moving_leg_jerk_reduction": _MOVING_LEG_JERK_REDUCTION_GATE,
                "development_residual_effect": _RESIDUAL_INCREMENTAL_EFFECT_GATE,
                "development_residual_regression": (_RESIDUAL_INCREMENTAL_REGRESSION_LIMIT),
                "expert_generalization_effect": _EXPERT_GENERALIZATION_EFFECT_GATE,
                "expert_generalization_regression": (_EXPERT_GENERALIZATION_REGRESSION_LIMIT),
                "expert_generalization_causal_effect": (_EXPERT_GENERALIZATION_CAUSAL_EFFECT_GATE),
            },
        }


def build_g1_temporal_muscle_memory_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Build replay-consolidated DEVELOPMENT cases for the temporal expert.

    The static and disturbed cases retain the old controller.  Three moving
    cases vary actual ball/contact dynamics and are jointly consumed by the
    optimizer, so an impact prototype cannot be qualified from one replay.
    """

    static, moving, disturbed = build_g1_muscle_memory_cases()
    lateral_scenario = replace(
        moving.scenario,
        scenario_id="temporal-development-moving-lateral-005",
        ball_y_m=0.005,
    )
    light_ball_scenario = replace(
        moving.scenario,
        scenario_id="temporal-development-moving-light-400g",
        ball_mass_kg=0.40,
    )
    adapter = MovingBallInterceptAdapter()
    lateral_plan = adapter.plan(lateral_scenario)
    light_ball_plan = adapter.plan(light_ball_scenario)
    if not lateral_plan.eligible or not light_ball_plan.eligible:
        raise RuntimeError("temporal moving-ball development curriculum is ineligible")
    return (
        static,
        replace(moving, name="moving_ball_nominal"),
        G1MuscleMemoryCase(
            name="moving_ball_lateral_005",
            scenario=lateral_scenario,
            parameters=lateral_plan.parameters,
        ),
        G1MuscleMemoryCase(
            name="moving_ball_light_400g",
            scenario=light_ball_scenario,
            parameters=light_ball_plan.parameters,
        ),
        disturbed,
    )


def _build_temporal_expert_validation_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Predeclare moving VALIDATION cases never consumed by candidate search."""

    base = generate_goalforge_scenarios(
        ledger=SeedLedger(
            task_id="g1_penalty_kick",
            secret=b"rosclaw-g1-temporal-expert-generalization-v1",
        ),
        partition=Partition.VALIDATION,
        count=1,
        generation=10,
    )[0]
    scenario = replace(
        base,
        scenario_id="temporal-validation-moving-restitution-535",
        ball_x_m=1.12,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.08,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=4.0,
        ball_mass_kg=0.41,
        ball_ground_friction=0.03,
        support_ground_friction=1.0,
        restitution=0.535,
        disturbance_n=0.0,
        control_latency_ms=0.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        target_y_m=0.0,
        target_z_m=0.20,
        reachable=True,
    )
    plan = MovingBallInterceptAdapter().plan(scenario)
    if not plan.eligible:
        raise RuntimeError("temporal moving-ball expert validation case is ineligible")
    return (
        G1MuscleMemoryCase(
            name="temporal_validation_moving_restitution_535",
            scenario=scenario,
            parameters=plan.parameters,
        ),
    )


class G1TemporalMuscleMemoryTrainer:
    """CEM over a recurrent RBF actor with fail-closed matched evaluation."""

    def __init__(
        self,
        *,
        asset_root: Path,
        config: G1TemporalMuscleMemoryTrainingConfig | None = None,
    ) -> None:
        self.config = config or G1TemporalMuscleMemoryTrainingConfig()
        self.base = G1MuscleMemoryTrainer(
            asset_root=asset_root,
            cases=build_g1_temporal_muscle_memory_cases(),
            config=G1MuscleMemoryTrainingConfig(population_size=6, generations=1),
        )

    def train(self) -> G1TemporalMuscleMemoryTrainingReport:
        retained_recovery_config = self.base.recovery_config
        parents = tuple(self.base._run_parent(case) for case in self.base.cases)
        parent_metrics = tuple(measure_g1_recovery_quality(row.trajectory) for row in parents)
        moving_indices = tuple(
            index for index, case in enumerate(self.base.cases) if _is_moving_case(case)
        )
        if len(moving_indices) < 3:
            raise RuntimeError("temporal training requires at least three moving-ball regimes")
        observation_mean, observation_scale = _normalization(parents)
        retained_controller = self.base.backend.build_cerebellar_recovery_controller(
            self.base.cases[0].scenario,
            retained_recovery_config,
        )
        # Search the structured motor primitive before the neural residual.
        # The private holdout remains sealed until both policy heads are frozen.
        (
            temporal_recovery_config,
            structured_trials,
            structured_rollout_count,
        ) = self._select_structured_recovery(
            retained_recovery_config=retained_recovery_config,
            parents=parents,
            parent_metrics=parent_metrics,
            moving_indices=moving_indices,
        )
        self.base.recovery_config = temporal_recovery_config
        self.base.recovery_fallback_config = retained_recovery_config
        parent_controller = self.base.backend.build_cerebellar_recovery_controller(
            self.base.cases[0].scenario,
            temporal_recovery_config,
        )
        expert_validation_cases = _build_temporal_expert_validation_cases()
        expert_validation_suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.temporal_expert_generalization.v1",
            expert_validation_cases,
        )
        dataset_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.temporal_muscle_memory_dataset.v2",
                "cases": [
                    {
                        "name": case.name,
                        "scenario_commitment": case.scenario.scenario_commitment,
                        "policy_hash": case.parameters.policy_hash,
                        "trajectory_hash": trajectory_digest(parent.trajectory),
                    }
                    for case, parent in zip(self.base.cases, parents, strict=True)
                ],
                "observation_mean": list(observation_mean),
                "observation_scale": list(observation_scale),
                "private_holdout_excluded": True,
                "expert_validation_excluded": True,
                "expert_validation_suite_hash": expert_validation_suite_hash,
                "structured_recovery_trials": [asdict(item) for item in structured_trials],
            }
        )
        impact_prototypes = tuple(
            float(parents[index].result.contact_impulse_ns) for index in moving_indices
        )
        expert_regime_prototypes = tuple(
            _expert_route_prototype(
                parents[index],
                observation_mean=observation_mean,
                observation_scale=observation_scale,
            )
            for index in moving_indices
        )
        context: dict[str, Any] = {
            "body_hash": self.base.backend.qualification.body_hash,
            "motion_hash": self.base.backend.qualification.motion_hash,
            "parent_config_hash": parent_controller.config_hash,
            "fallback_config_hash": retained_controller.config_hash,
            "dataset_hash": dataset_hash,
            "observation_mean": observation_mean,
            "observation_scale": observation_scale,
            "expert_impact_prototypes_ns": impact_prototypes,
            "expert_impact_max_distance_ns": _calibrate_impact_radius(impact_prototypes),
            "expert_regime_prototypes": expert_regime_prototypes,
            "expert_regime_max_distance": 0.25,
            "structured_recovery_parameters": (
                temporal_recovery_config.settling_standing_pose_blend,
                temporal_recovery_config.settling_waist_pitch_bias_rad,
                temporal_recovery_config.target_smoothing_alpha,
            ),
        }
        ablation_artifact = _artifact_from_temporal_genome(
            np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64),
            training_episode_count=1,
            training_seed=self.config.seed,
            **context,
        )
        _, ablation_episodes = self._evaluate_artifact(
            ablation_artifact,
            parents=parents,
            parent_metrics=parent_metrics,
            require_temporal_capacity=False,
        )
        ablation_metrics = tuple(
            measure_g1_recovery_quality(item.trajectory) for item in ablation_episodes
        )
        baseline_score = float(
            np.mean(
                [
                    _case_score(
                        parent=parent,
                        parent_quality=parent_quality,
                        candidate=ablation,
                        candidate_quality=ablation_quality,
                    )[0]
                    for parent, parent_quality, ablation, ablation_quality in zip(
                        parents,
                        parent_metrics,
                        ablation_episodes,
                        ablation_metrics,
                        strict=True,
                    )
                ]
            )
        )
        rng = np.random.default_rng(self.config.seed)
        zero: np.ndarray = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
        seeds = _seed_genomes()
        best_genome = zero.copy()
        best_score = self._evaluate_genome(
            best_genome,
            parents=parents,
            parent_metrics=parent_metrics,
            ablation_metrics=ablation_metrics,
            context=context,
        )[0]
        rollout_count = len(self.base.cases)
        for genome in seeds:
            score, _ = self._evaluate_genome(
                genome,
                parents=parents,
                parent_metrics=parent_metrics,
                ablation_metrics=ablation_metrics,
                context=context,
            )
            rollout_count += len(self.base.cases)
            if score > best_score:
                best_score = score
                best_genome = genome.copy()
        distribution_mean = best_genome.copy()
        distribution_std: np.ndarray = np.full(
            _TEMPORAL_GENOME_SIZE, self.config.initial_std, dtype=np.float64
        )
        elite_count = max(
            2, int(math.ceil(self.config.population_size * self.config.elite_fraction))
        )
        generation_best: list[float] = []
        for _generation in range(self.config.generations):
            population = rng.normal(
                distribution_mean,
                distribution_std,
                size=(self.config.population_size, _TEMPORAL_GENOME_SIZE),
            )
            population[0] = distribution_mean
            population[1] = zero
            scored = []
            for genome in population:
                score, _ = self._evaluate_genome(
                    genome,
                    parents=parents,
                    parent_metrics=parent_metrics,
                    ablation_metrics=ablation_metrics,
                    context=context,
                )
                scored.append((score, genome.copy()))
            rollout_count += len(population) * len(self.base.cases)
            scored.sort(key=lambda item: item[0], reverse=True)
            elites = np.stack([item[1] for item in scored[:elite_count]])
            distribution_mean = 0.20 * distribution_mean + 0.80 * np.mean(elites, axis=0)
            distribution_std = np.maximum(
                self.config.minimum_std,
                0.35 * distribution_std + 0.65 * np.std(elites, axis=0),
            )
            generation_best.append(scored[0][0])
            if scored[0][0] > best_score:
                best_score, best_genome = scored[0]
        artifact = _artifact_from_temporal_genome(
            best_genome,
            training_episode_count=rollout_count,
            training_seed=self.config.seed,
            **context,
        )
        _, candidate_episodes = self._evaluate_artifact(
            artifact,
            parents=parents,
            parent_metrics=parent_metrics,
            ablation_metrics=ablation_metrics,
        )
        reasons: list[str] = []
        case_rows: list[G1MuscleMemoryCaseResult] = []
        candidate_metrics: list[G1RecoveryQuality] = []
        development_expert_routes = 0
        development_fallback_routes = 0
        for case, parent, parent_quality, candidate in zip(
            self.base.cases, parents, parent_metrics, candidate_episodes, strict=True
        ):
            replay = self.base._run_candidate(case, artifact)
            strict = bool(
                candidate.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(candidate.trajectory) == trajectory_digest(replay.trajectory)
            )
            candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            route = (
                candidate.recovery_receipt.expert_route_latched
                if candidate.recovery_receipt is not None
                else None
            )
            candidate_metrics.append(candidate_quality)
            development_expert_routes += int(route is True)
            development_fallback_routes += int(route is False)
            if route is None:
                reasons.append(case.name + ":expert_route_missing")
            elif _is_moving_case(case) and route is not True:
                reasons.append(case.name + ":moving_expert_route_not_selected")
            elif not _is_moving_case(case) and route is not False:
                reasons.append(case.name + ":retained_replay_route_not_selected")
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
            case_rows.append(
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
        candidate_metrics_tuple = tuple(candidate_metrics)
        backward, wobble, jerk = _moving_suite_reductions(
            moving_indices,
            parent_metrics,
            candidate_metrics_tuple,
        )
        structured_reductions = _moving_suite_reductions(
            moving_indices,
            parent_metrics,
            ablation_metrics,
        )
        residual_reductions = _moving_suite_reductions(
            moving_indices,
            ablation_metrics,
            candidate_metrics_tuple,
        )
        residual_causal_gate = bool(
            min(residual_reductions) >= -_RESIDUAL_INCREMENTAL_REGRESSION_LIMIT
            and max(residual_reductions) >= _RESIDUAL_INCREMENTAL_EFFECT_GATE
        )
        if not residual_causal_gate:
            reasons.append("learned_residual_causal_ablation_failed")
        for actual, threshold, reason in (
            (backward, _MOVING_BACKWARD_REDUCTION_GATE, "moving_backward_reduction_below_gate"),
            (wobble, _MOVING_WOBBLE_REDUCTION_GATE, "moving_tail_wobble_reduction_below_gate"),
            (jerk, _MOVING_LEG_JERK_REDUCTION_GATE, "moving_leg_jerk_reduction_below_gate"),
        ):
            if actual < threshold:
                reasons.append(reason)
        learned_temporal: float = float(
            max(
                np.max(np.abs(np.asarray(artifact.weights))),
                np.max(np.abs(np.asarray(artifact.temporal_basis_weights))),
                np.max(np.abs(np.asarray(artifact.proprioceptive_trend_weights))),
            )
        )
        if learned_temporal <= 1e-6:
            reasons.append("temporal_policy_capacity_unused")
        expert_generalization, expert_generalization_rollouts = (
            self._evaluate_expert_generalization(
                artifact,
                ablation_artifact=ablation_artifact,
                retained_recovery_config=retained_recovery_config,
                cases=expert_validation_cases,
                expected_suite_hash=expert_validation_suite_hash,
            )
        )
        if not expert_generalization.qualified:
            reasons.append("moving_expert_generalization_failed")
        (
            holdout,
            holdout_rollouts,
            holdout_expert_routes,
            holdout_fallback_routes,
            holdout_parent_valid,
        ) = self._evaluate_private_holdout(
            artifact,
            retained_recovery_config=retained_recovery_config,
        )
        if not holdout.qualified:
            reasons.append("private_holdout_failed")
        parity = _validate_cuda_parity(
            artifact,
            observations=np.asarray(
                parents[moving_indices[0]].trajectory["recovery_proprioception"],
                dtype=np.float64,
            ),
            devices=self.config.cuda_devices,
        )
        if not parity.passed:
            reasons.append("cuda_numpy_inference_parity_failed")
        candidate_score = float(np.mean([row.score for row in case_rows]))
        return G1TemporalMuscleMemoryTrainingReport(
            artifact=artifact,
            cases=tuple(case_rows),
            generation_best_scores=tuple(generation_best),
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            candidate_worst_case_score=float(min(row.score for row in case_rows)),
            training_rollout_count=(
                rollout_count
                + structured_rollout_count
                + len(ablation_episodes)
                + 2 * len(self.base.cases)
            ),
            holdout_rollout_count=holdout_rollouts,
            holdout=holdout,
            expert_generalization_rollout_count=expert_generalization_rollouts,
            expert_generalization=expert_generalization,
            gpu_parity=parity,
            structured_recovery_trials=structured_trials,
            retained_recovery_config_hash=retained_controller.config_hash,
            temporal_recovery_config_hash=parent_controller.config_hash,
            temporal_recovery_config=asdict(temporal_recovery_config),
            moving_backward_reduction=backward,
            moving_tail_wobble_reduction=wobble,
            moving_leg_jerk_reduction=jerk,
            moving_structured_backward_reduction=structured_reductions[0],
            moving_structured_tail_wobble_reduction=structured_reductions[1],
            moving_structured_leg_jerk_reduction=structured_reductions[2],
            moving_residual_backward_reduction=residual_reductions[0],
            moving_residual_tail_wobble_reduction=residual_reductions[1],
            moving_residual_leg_jerk_reduction=residual_reductions[2],
            residual_causal_gate_passed=residual_causal_gate,
            development_expert_route_count=development_expert_routes,
            development_fallback_route_count=development_fallback_routes,
            holdout_expert_route_count=holdout_expert_routes,
            holdout_fallback_route_count=holdout_fallback_routes,
            holdout_parent_valid_count=holdout_parent_valid,
            qualified=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def _select_structured_recovery(
        self,
        *,
        retained_recovery_config: G1CerebellarRecoveryConfig,
        parents: tuple[Any, ...],
        parent_metrics: tuple[G1RecoveryQuality, ...],
        moving_indices: tuple[int, ...],
    ) -> tuple[
        G1CerebellarRecoveryConfig,
        tuple[G1TemporalRecoveryConfigTrial, ...],
        int,
    ]:
        """Select a bounded motor primitive from development rollouts only."""

        candidates = (
            retained_recovery_config,
            replace(
                retained_recovery_config,
                settling_standing_pose_blend=0.40,
                settling_waist_pitch_bias_rad=0.10,
                target_smoothing_alpha=0.52,
            ),
            replace(
                retained_recovery_config,
                settling_standing_pose_blend=0.42,
                settling_waist_pitch_bias_rad=0.11,
                target_smoothing_alpha=0.54,
            ),
            replace(
                retained_recovery_config,
                settling_standing_pose_blend=0.42,
                settling_waist_pitch_bias_rad=0.11,
                target_smoothing_alpha=0.545,
            ),
            replace(
                retained_recovery_config,
                settling_standing_pose_blend=0.44,
                settling_waist_pitch_bias_rad=0.12,
                target_smoothing_alpha=0.56,
            ),
        )
        if len(moving_indices) < 2:
            raise RuntimeError("structured recovery search requires multiple moving-ball cases")
        scored: list[tuple[float, G1CerebellarRecoveryConfig, G1TemporalRecoveryConfigTrial]] = []
        for config in candidates:
            case_scores: list[float] = []
            case_reductions: list[tuple[float, float, float]] = []
            safe_rows: list[bool] = []
            goal_rows: list[bool] = []
            natural_rows: list[bool] = []
            controller_hash = ""
            for index in moving_indices:
                case = self.base.cases[index]
                controller = self.base.backend.build_cerebellar_recovery_controller(
                    case.scenario,
                    config,
                )
                controller_hash = controller.config_hash
                episode = self.base.backend.run(
                    case.scenario,
                    case.parameters,
                    feedback_runtime=self.base._feedback_runtime(case),
                    recovery_controller=controller,
                )
                try:
                    quality = measure_g1_recovery_quality(episode.trajectory)
                    score, safe, goal, natural = _case_score(
                        parent=parents[index],
                        parent_quality=parent_metrics[index],
                        candidate=episode,
                        candidate_quality=quality,
                    )
                    reductions = _moving_reductions(parent_metrics[index], quality)
                except ValueError:
                    score, safe, goal, natural = -1_000_000.0, False, False, False
                    reductions = (-1_000_000.0, -1_000_000.0, -1_000_000.0)
                case_scores.append(score)
                case_reductions.append(reductions)
                safe_rows.append(safe)
                goal_rows.append(goal)
                natural_rows.append(natural)
            reduction_array = np.asarray(case_reductions, dtype=np.float64)
            mean_reductions = np.mean(reduction_array, axis=0)
            reductions = (
                float(mean_reductions[0]),
                float(mean_reductions[1]),
                float(mean_reductions[2]),
            )
            safe = all(safe_rows)
            goal = all(goal_rows)
            natural = all(natural_rows)
            eligible = bool(
                safe
                and goal
                and natural
                and np.all(
                    reduction_array
                    >= np.asarray(
                        (
                            _MOVING_BACKWARD_REDUCTION_GATE,
                            _MOVING_WOBBLE_REDUCTION_GATE,
                            _MOVING_LEG_JERK_REDUCTION_GATE,
                        )
                    )
                )
            )
            selection_score = (
                float(np.mean(case_scores))
                + 4.0 * min(0.30, reductions[0])
                + 5.0 * min(0.30, reductions[1])
                + 2.0 * min(0.30, reductions[2])
                if eligible
                else -1_000_000.0
            )
            scored.append(
                (
                    selection_score,
                    config,
                    G1TemporalRecoveryConfigTrial(
                        config_hash=controller_hash,
                        settling_standing_pose_blend=float(
                            config.settling_standing_pose_blend or 0.0
                        ),
                        settling_waist_pitch_bias_rad=config.settling_waist_pitch_bias_rad,
                        target_smoothing_alpha=config.target_smoothing_alpha,
                        score=selection_score,
                        moving_backward_reduction=reductions[0],
                        moving_tail_wobble_reduction=reductions[1],
                        moving_leg_jerk_reduction=reductions[2],
                        safe=safe,
                        goal_preserved=goal,
                        naturalness_preserved=natural,
                    ),
                )
            )
        scored.sort(key=lambda item: (item[0], item[2].config_hash), reverse=True)
        if scored[0][0] <= -1_000_000.0:
            raise RuntimeError("no structured recovery candidate passed development gates")
        selected_config = scored[0][1]
        selected_hash = scored[0][2].config_hash
        trials = tuple(
            replace(trial, selected=trial.config_hash == selected_hash)
            for _score, _config, trial in scored
        )
        return selected_config, trials, len(candidates) * len(moving_indices)

    def _evaluate_expert_generalization(
        self,
        artifact: G1MuscleMemoryArtifact,
        *,
        ablation_artifact: G1MuscleMemoryArtifact,
        retained_recovery_config: G1CerebellarRecoveryConfig,
        cases: tuple[G1MuscleMemoryCase, ...],
        expected_suite_hash: str,
    ) -> tuple[G1TemporalExpertGeneralizationSummary, int]:
        """Open a precommitted VALIDATION suite only after the actor is frozen."""

        scores: list[float] = []
        total_reductions: list[tuple[float, float, float]] = []
        residual_reductions: list[tuple[float, float, float]] = []
        parent_valid_count = 0
        expert_route_count = 0
        strict_replay_count = 0
        passed_count = 0
        for case in cases:
            parent_controller = self.base.backend.build_cerebellar_recovery_controller(
                case.scenario,
                retained_recovery_config,
            )
            parent = self.base.backend.run(
                case.scenario,
                case.parameters,
                feedback_runtime=self.base._feedback_runtime(case),
                recovery_controller=parent_controller,
            )
            structured = self.base._run_candidate(case, ablation_artifact)
            candidate = self.base._run_candidate(case, artifact)
            replay = self.base._run_candidate(case, artifact)
            route = (
                candidate.recovery_receipt.expert_route_latched
                if candidate.recovery_receipt is not None
                else None
            )
            structured_route = (
                structured.recovery_receipt.expert_route_latched
                if structured.recovery_receipt is not None
                else None
            )
            expert_route_count += int(route is True)
            strict = bool(
                candidate.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(candidate.trajectory) == trajectory_digest(replay.trajectory)
            )
            strict_replay_count += int(strict)
            try:
                parent_quality = measure_g1_recovery_quality(parent.trajectory)
                structured_quality = measure_g1_recovery_quality(structured.trajectory)
                candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            except ValueError:
                scores.append(-1_000_000.0)
                total_reductions.append((-1_000_000.0,) * 3)
                residual_reductions.append((-1_000_000.0,) * 3)
                continue
            parent_valid = _parent_valid(parent, parent_quality)
            parent_valid_count += int(parent_valid)
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=candidate_quality,
            )
            scores.append(score)
            total_reductions.append(_moving_reductions(parent_quality, candidate_quality))
            residual_reductions.append(_moving_reductions(structured_quality, candidate_quality))
            passed_count += int(
                parent_valid
                and safe
                and goal
                and natural
                and strict
                and route is True
                and structured_route is True
                and score >= -_EXPERT_GENERALIZATION_REGRESSION_LIMIT
            )
        total = tuple(map(float, np.mean(np.asarray(total_reductions), axis=0)))
        causal = tuple(map(float, np.mean(np.asarray(residual_reductions), axis=0)))
        suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.temporal_expert_generalization.v1",
            cases,
        )
        qualified = bool(
            suite_hash == expected_suite_hash
            and passed_count == len(cases)
            and parent_valid_count == len(cases)
            and expert_route_count == len(cases)
            and strict_replay_count == len(cases)
            and min(total) >= -_EXPERT_GENERALIZATION_REGRESSION_LIMIT
            and max(total) >= _EXPERT_GENERALIZATION_EFFECT_GATE
            and min(causal) >= -_RESIDUAL_INCREMENTAL_REGRESSION_LIMIT
            and max(causal) >= _EXPERT_GENERALIZATION_CAUSAL_EFFECT_GATE
        )
        return (
            G1TemporalExpertGeneralizationSummary(
                suite_hash=suite_hash,
                case_count=len(cases),
                parent_valid_count=parent_valid_count,
                expert_route_count=expert_route_count,
                strict_replay_count=strict_replay_count,
                passed_count=passed_count,
                mean_score=float(np.mean(scores)),
                minimum_score=float(np.min(scores)),
                moving_backward_reduction=total[0],
                moving_tail_wobble_reduction=total[1],
                moving_leg_jerk_reduction=total[2],
                residual_backward_reduction=causal[0],
                residual_tail_wobble_reduction=causal[1],
                residual_leg_jerk_reduction=causal[2],
                qualified=qualified,
            ),
            4 * len(cases),
        )

    def _evaluate_private_holdout(
        self,
        artifact: G1MuscleMemoryArtifact,
        *,
        retained_recovery_config: G1CerebellarRecoveryConfig,
    ) -> tuple[G1MuscleMemoryHoldoutSummary, int, int, int, int]:
        cases = _build_temporal_holdout_cases()
        scores: list[float] = []
        passed = 0
        strict_count = 0
        expert_routes = 0
        fallback_routes = 0
        parent_valid_count = 0
        for case in cases:
            parent_controller = self.base.backend.build_cerebellar_recovery_controller(
                case.scenario,
                retained_recovery_config,
            )
            parent = self.base.backend.run(
                case.scenario,
                case.parameters,
                feedback_runtime=self.base._feedback_runtime(case),
                recovery_controller=parent_controller,
            )
            candidate = self.base._run_candidate(case, artifact)
            route = (
                candidate.recovery_receipt.expert_route_latched
                if candidate.recovery_receipt is not None
                else None
            )
            expert_routes += int(route is True)
            fallback_routes += int(route is False)
            replay = self.base._run_candidate(case, artifact)
            strict = bool(
                candidate.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(candidate.trajectory) == trajectory_digest(replay.trajectory)
            )
            strict_count += int(strict)
            try:
                parent_quality = measure_g1_recovery_quality(parent.trajectory)
                candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
            except ValueError:
                scores.append(-1_000_000.0)
                continue
            parent_valid = _parent_valid(parent, parent_quality)
            parent_valid_count += int(parent_valid)
            if not parent_valid:
                scores.append(-1_000_000.0)
                continue
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=candidate_quality,
            )
            passed += int(
                safe and goal and natural and strict and route is False and score >= -0.03
            )
            scores.append(score)
        suite_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.temporal_private_holdout.v10",
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
                qualified=bool(
                    passed == len(cases)
                    and parent_valid_count == len(cases)
                    and expert_routes == 0
                    and fallback_routes == len(cases)
                ),
            ),
            3 * len(cases),
            expert_routes,
            fallback_routes,
            parent_valid_count,
        )

    def _evaluate_genome(
        self,
        genome: np.ndarray,
        *,
        parents: tuple[Any, ...],
        parent_metrics: tuple[G1RecoveryQuality, ...],
        ablation_metrics: tuple[G1RecoveryQuality, ...],
        context: dict[str, Any],
    ) -> tuple[float, tuple[Any, ...]]:
        artifact = _artifact_from_temporal_genome(
            genome,
            training_episode_count=1,
            training_seed=self.config.seed,
            **context,
        )
        return self._evaluate_artifact(
            artifact,
            parents=parents,
            parent_metrics=parent_metrics,
            ablation_metrics=ablation_metrics,
            require_temporal_capacity=False,
        )

    def _evaluate_artifact(
        self,
        artifact: G1MuscleMemoryArtifact,
        *,
        parents: tuple[Any, ...],
        parent_metrics: tuple[G1RecoveryQuality, ...],
        ablation_metrics: tuple[G1RecoveryQuality, ...] | None = None,
        require_temporal_capacity: bool = True,
    ) -> tuple[float, tuple[Any, ...]]:
        candidates = tuple(self.base._run_candidate(case, artifact) for case in self.base.cases)
        scores: list[float] = []
        residual_rows: list[tuple[float, float, float]] = []
        temporal_capacity: float = float(
            max(
                np.max(np.abs(np.asarray(artifact.weights))),
                np.max(np.abs(np.asarray(artifact.temporal_basis_weights))),
                np.max(np.abs(np.asarray(artifact.proprioceptive_trend_weights))),
            )
        )
        valid = bool(not require_temporal_capacity or temporal_capacity > 1e-6)
        for index, (case, parent, parent_quality, candidate) in enumerate(
            zip(self.base.cases, parents, parent_metrics, candidates, strict=True)
        ):
            try:
                quality = measure_g1_recovery_quality(candidate.trajectory)
            except ValueError:
                return -1_000_000.0, candidates
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_quality,
                candidate=candidate,
                candidate_quality=quality,
            )
            valid = valid and safe and goal and natural
            if _is_moving_case(case):
                backward, wobble, jerk = _moving_reductions(parent_quality, quality)
                valid = valid and bool(
                    backward >= _MOVING_BACKWARD_REDUCTION_GATE
                    and wobble >= _MOVING_WOBBLE_REDUCTION_GATE
                    and jerk >= _MOVING_LEG_JERK_REDUCTION_GATE
                )
                if ablation_metrics is not None:
                    residual_reductions = _moving_reductions(ablation_metrics[index], quality)
                    residual_rows.append(residual_reductions)
                    score += 4.0 * residual_reductions[0]
                    score += 5.0 * residual_reductions[1]
                    score += 2.0 * residual_reductions[2]
                    score -= 40.0 * sum(
                        max(0.0, -_RESIDUAL_INCREMENTAL_REGRESSION_LIMIT - value)
                        for value in residual_reductions
                    )
                score += 4.0 * min(0.30, backward)
                score += 5.0 * min(0.30, wobble)
                score += 2.0 * min(0.30, jerk)
                score -= 12.0 * max(0.0, -wobble)
                score -= 8.0 * max(0.0, -backward)
            scores.append(score)
        if not valid:
            return -1_000_000.0, candidates
        causal_penalty = 0.0
        if residual_rows:
            aggregate_residual = np.mean(np.asarray(residual_rows), axis=0)
            causal_penalty = 50.0 * max(
                0.0,
                _RESIDUAL_INCREMENTAL_EFFECT_GATE - float(np.max(aggregate_residual)),
            )
        worst = min(scores)
        return float(np.mean(scores) + 0.25 * worst - causal_penalty), candidates


def _is_moving_case(case: G1MuscleMemoryCase) -> bool:
    scenario = case.scenario
    return bool(
        scenario.ball_launch_delay_sec > 0.0
        and math.hypot(scenario.ball_velocity_x_mps, scenario.ball_velocity_y_mps) > 0.0
    )


def _case_commitment_hash(
    schema_version: str,
    cases: tuple[G1MuscleMemoryCase, ...],
) -> str:
    return hash_json(
        {
            "schema_version": schema_version,
            "cases": [
                {
                    "scenario_commitment": case.scenario.scenario_commitment,
                    "policy_hash": case.parameters.policy_hash,
                }
                for case in cases
            ],
        }
    )


def _calibrate_impact_radius(prototypes: tuple[float, ...]) -> float:
    """Derive an expert radius from DEVELOPMENT spacing, never validation rows."""

    values = np.asarray(prototypes, dtype=np.float64)
    if not 1 <= len(values) <= 8 or not np.all(np.isfinite(values)):
        raise ValueError("impact calibration requires 1 to 8 finite prototypes")
    unique = np.unique(values)
    if len(unique) == 1:
        return 1e-6
    largest_development_gap = float(np.max(np.diff(np.sort(unique))))
    return float(min(0.20, max(0.01, largest_development_gap + 0.005)))


def _expert_route_prototype(
    episode: Any,
    *,
    observation_mean: tuple[float, ...],
    observation_scale: tuple[float, ...],
) -> tuple[float, ...]:
    """Capture the normalized proprioceptive state used by the causal router."""

    rows = np.asarray(episode.trajectory["recovery_proprioception"], dtype=np.float64)
    impulse_index = G1_MUSCLE_MEMORY_OBSERVATIONS.index("contact_impulse_ns")
    right_support_index = G1_MUSCLE_MEMORY_OBSERVATIONS.index("right_support")
    route_rows = np.flatnonzero(
        (rows[:, impulse_index] > 0.0) & (rows[:, right_support_index] > 0.5)
    )
    if not len(route_rows):
        raise ValueError("development episode has no contact-and-landing route state")
    normalized = (
        rows[int(route_rows[0])] - np.asarray(observation_mean, dtype=np.float64)
    ) / np.asarray(observation_scale, dtype=np.float64)
    indices = [
        G1_MUSCLE_MEMORY_OBSERVATIONS.index(name)
        for name in G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES
    ]
    return tuple(map(float, normalized[indices]))


def _moving_suite_reductions(
    moving_indices: tuple[int, ...],
    parent_metrics: tuple[G1RecoveryQuality, ...],
    candidate_metrics: tuple[G1RecoveryQuality, ...],
) -> tuple[float, float, float]:
    if not moving_indices:
        raise ValueError("moving reduction suite cannot be empty")
    reductions = np.asarray(
        [
            _moving_reductions(parent_metrics[index], candidate_metrics[index])
            for index in moving_indices
        ],
        dtype=np.float64,
    )
    mean_reductions = np.mean(reductions, axis=0)
    return (
        float(mean_reductions[0]),
        float(mean_reductions[1]),
        float(mean_reductions[2]),
    )


def _parent_valid(parent: Any, quality: G1RecoveryQuality) -> bool:
    result = parent.result
    return bool(
        result.success
        and result.contact_observed
        and result.goal_crossed
        and result.target_zone_hit
        and not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.support_foot_slip_m <= 0.04
        and quality.terminal_bilateral_support
    )


def _build_temporal_holdout_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Create a sealed suite that is never consumed by candidate search."""

    base = generate_goalforge_scenarios(
        ledger=SeedLedger(
            task_id="g1_penalty_kick",
            secret=b"rosclaw-g1-temporal-muscle-memory-private-holdout-v10",
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
        ball_mass_kg=0.415,
        ball_ground_friction=0.045,
        support_ground_friction=0.965,
        restitution=0.53,
        disturbance_n=0.0,
        control_latency_ms=4.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        target_y_m=0.0,
        target_z_m=0.20,
        reachable=True,
    )
    static = replace(
        base,
        scenario_id="temporal-private-v10-static-offset",
        ball_y_m=0.10,
        target_z_m=0.55,
    )
    moving = replace(
        base,
        scenario_id="temporal-private-v10-moving-variable",
        ball_x_m=1.12,
        ball_velocity_x_mps=-0.08,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=4.0,
        ball_ground_friction=0.03,
    )
    moving_plan = MovingBallInterceptAdapter().plan(moving)
    if not moving_plan.eligible:
        raise RuntimeError("temporal private moving-ball holdout is ineligible")
    disturbed = replace(
        base,
        scenario_id="temporal-private-v10-disturbed-70n",
        disturbance_n=70.0,
    )
    return (
        G1MuscleMemoryCase(
            name="temporal_private_v10_static",
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
            name="temporal_private_v10_moving",
            scenario=moving,
            parameters=moving_plan.parameters,
        ),
        G1MuscleMemoryCase(
            name="temporal_private_v10_disturbed",
            scenario=disturbed,
            parameters=ShotParameters(
                stance_offset_y=-0.08,
                swing_amplitude=1.125,
                policy_type="parameter",
            ),
            feedback_enabled=True,
        ),
    )


def _seed_genomes() -> tuple[np.ndarray, ...]:
    seeds: list[np.ndarray] = []
    value: np.ndarray
    for sagittal, absorption in ((0.03, 0.0), (-0.03, 0.0), (0.0, 0.01), (0.03, 0.01)):
        value = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
        value[22] = sagittal
        value[23] = absorption
        seeds.append(value)
    for profile in (
        (0.05, 0.03, -0.01, -0.03),
        (0.03, 0.02, 0.00, -0.02),
        (-0.02, 0.00, 0.02, 0.03),
    ):
        value = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
        value[:4] = profile
        value[4:8] = (0.0, 0.01, 0.02, 0.01)
        seeds.append(value)
    sagittal_release: np.ndarray = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
    sagittal_release[0] = 0.01
    sagittal_release[22] = -0.03
    seeds.append(sagittal_release)
    causal_probe: np.ndarray = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
    causal_probe[0] = 0.01
    causal_probe[22] = 0.001
    seeds.append(causal_probe)
    for index, gain in ((29, 0.01), (32, 0.01), (33, 0.01), (35, 0.01)):
        state_feedback: np.ndarray = np.zeros(_TEMPORAL_GENOME_SIZE, dtype=np.float64)
        state_feedback[index] = gain
        seeds.append(state_feedback)
    return tuple(seeds)


def _artifact_from_temporal_genome(
    genome: np.ndarray,
    *,
    body_hash: str,
    motion_hash: str,
    parent_config_hash: str,
    fallback_config_hash: str,
    dataset_hash: str,
    observation_mean: tuple[float, ...],
    observation_scale: tuple[float, ...],
    expert_impact_prototypes_ns: tuple[float, ...],
    structured_recovery_parameters: tuple[float, ...],
    training_episode_count: int,
    training_seed: int,
    expert_impact_max_distance_ns: float = 1e-6,
    expert_regime_prototypes: tuple[tuple[float, ...], ...] = (),
    expert_regime_max_distance: float = 0.25,
) -> G1MuscleMemoryArtifact:
    value = np.asarray(genome, dtype=np.float64)
    if value.shape != (_TEMPORAL_GENOME_SIZE,) or not np.all(np.isfinite(value)):
        raise ValueError(
            f"temporal muscle-memory genome must contain {_TEMPORAL_GENOME_SIZE} values"
        )
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    weights: np.ndarray = np.zeros((actions, observations), dtype=np.float64)
    basis: np.ndarray = np.zeros((actions, len(_BASIS_CENTERS_SEC)), dtype=np.float64)
    trend: np.ndarray = np.zeros((actions, observations), dtype=np.float64)
    bias: np.ndarray = np.zeros(actions, dtype=np.float64)
    action = {name: G1_MUSCLE_MEMORY_ACTIONS.index(name) for name in G1_MUSCLE_MEMORY_ACTIONS}
    obs = {
        name: G1_MUSCLE_MEMORY_OBSERVATIONS.index(name) for name in G1_MUSCLE_MEMORY_OBSERVATIONS
    }
    basis[action["sagittal_common"]] = np.clip(value[0:4], -0.35, 0.35)
    basis[action["leg_absorption"]] = np.clip(value[4:8], -0.35, 0.35)
    basis[action["waist_pitch"]] = np.clip(value[8:12], -0.25, 0.25)
    trend[action["sagittal_common"], obs["pelvis_velocity_x_m_s"]] = value[12]
    trend[action["sagittal_common"], obs["torso_pitch_rad"]] = value[13]
    trend[action["sagittal_common"], obs["torso_angular_velocity_y_rad_s"]] = value[14]
    trend[action["leg_absorption"], obs["pelvis_velocity_z_m_s"]] = value[15]
    trend[action["leg_absorption"], obs["left_ground_force_scale"]] = value[16]
    trend[action["leg_absorption"], obs["right_ground_force_scale"]] = value[17]
    trend[action["leg_absorption"], obs["contact_impulse_ns"]] = value[18]
    trend[action["waist_pitch"], obs["torso_pitch_rad"]] = value[19]
    trend[action["waist_pitch"], obs["torso_angular_velocity_y_rad_s"]] = value[20]
    trend[action["arm_pitch_counter"], obs["torso_angular_velocity_y_rad_s"]] = value[21]
    bias[action["sagittal_common"]] = value[22]
    bias[action["leg_absorption"]] = value[23]
    bias[action["waist_pitch"]] = value[24]
    bias[action["arm_pitch_counter"]] = value[25]
    weights[action["sagittal_common"], obs["pelvis_velocity_x_m_s"]] = value[29]
    weights[action["sagittal_common"], obs["torso_pitch_rad"]] = value[30]
    weights[action["sagittal_common"], obs["torso_angular_velocity_y_rad_s"]] = value[31]
    weights[action["leg_absorption"], obs["pelvis_velocity_z_m_s"]] = value[32]
    weights[action["leg_absorption"], obs["left_ground_force_scale"]] = value[33]
    weights[action["leg_absorption"], obs["right_ground_force_scale"]] = -value[33]
    weights[action["leg_absorption"], obs["contact_impulse_ns"]] = value[34]
    weights[action["waist_pitch"], obs["torso_pitch_rad"]] = value[35]
    weights[action["waist_pitch"], obs["torso_angular_velocity_y_rad_s"]] = value[36]
    duration = float(0.9 + 0.50 * np.tanh(value[26]))
    width = float(0.22 + 0.15 / (1.0 + np.exp(-value[27])))
    memory_alpha = float(0.12 + 0.38 / (1.0 + np.exp(-value[28])))
    return G1MuscleMemoryArtifact(
        body_hash=body_hash,
        motion_hash=motion_hash,
        parent_recovery_config_hash=parent_config_hash,
        training_dataset_hash=dataset_hash,
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        weights=tuple(tuple(map(float, row)) for row in np.clip(weights, -0.25, 0.25)),
        bias=tuple(map(float, np.clip(bias, -0.25, 0.25))),
        action_limits_rad=(0.055, 0.045, 0.050, 0.045, 0.050, 0.040, 0.040, 0.045, 0.045),
        training_episode_count=training_episode_count,
        training_seed=training_seed,
        temporal_basis_centers_sec=_BASIS_CENTERS_SEC,
        temporal_basis_width_sec=width,
        temporal_basis_weights=tuple(tuple(map(float, row)) for row in basis),
        proprioceptive_trend_weights=tuple(tuple(map(float, row)) for row in trend),
        proprioceptive_memory_alpha=memory_alpha,
        policy_architecture="leaky_rbf_recurrent_v1",
        fallback_recovery_config_hash=fallback_config_hash,
        expert_impact_prototypes_ns=expert_impact_prototypes_ns,
        expert_impact_max_distance_ns=expert_impact_max_distance_ns,
        expert_regime_feature_names=(
            G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES if expert_regime_prototypes else ()
        ),
        expert_regime_prototypes=expert_regime_prototypes,
        expert_regime_max_distance=expert_regime_max_distance,
        structured_recovery_parameters=structured_recovery_parameters,
        maximum_feature_z=8.0,
        activation_duration_sec=duration,
        fade_out_sec=min(0.35, duration / 2.0),
        sagittal_minimum_impulse_ns=1.75,
        schema_version=(
            "rosclaw.g1_goalforge.muscle_memory_artifact.v3"
            if expert_regime_prototypes
            else "rosclaw.g1_goalforge.muscle_memory_artifact.v2"
        ),
    )


def _moving_reductions(
    parent: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
) -> tuple[float, float, float]:
    return (
        _physical_reduction(
            parent.post_contact_backward_reversal_m,
            candidate.post_contact_backward_reversal_m,
            0.05,
        ),
        _physical_reduction(parent.tail_wobble_index, candidate.tail_wobble_index, 0.05),
        _physical_reduction(
            parent.post_contact_leg_joint_jerk_rms_rad_s3,
            candidate.post_contact_leg_joint_jerk_rms_rad_s3,
            100.0,
        ),
    )


def _physical_reduction(parent: float, candidate: float, floor: float) -> float:
    return (parent - candidate) / max(abs(parent), floor)


def _validate_cuda_parity(
    artifact: G1MuscleMemoryArtifact,
    *,
    observations: np.ndarray,
    devices: tuple[int, ...],
) -> G1TemporalGpuParity:
    if not devices:
        return G1TemporalGpuParity((), (), (), math.inf, (), False)
    try:
        import torch
    except ImportError:
        return G1TemporalGpuParity(devices, (), (), math.inf, (), False)
    if not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count():
        return G1TemporalGpuParity(devices, (), (), math.inf, (), False)
    rows = np.asarray(observations, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(G1_MUSCLE_MEMORY_OBSERVATIONS):
        raise ValueError("CUDA parity observations do not match the policy contract")
    # This check covers the learned logits only.  MuJoCo rollout and safety
    # projection remain CPU-authoritative and are separately strict-replayed.
    selected = rows[-min(256, len(rows)) :]
    normalized = (selected - np.asarray(artifact.observation_mean)) / np.asarray(
        artifact.observation_scale
    )
    elapsed = np.maximum(0.0, selected[:, 0] - selected[0, 0])
    expected = _numpy_temporal_logits(artifact, normalized, elapsed)
    hashes: list[str] = []
    names: list[str] = []
    maximum_error = 0.0
    for device_index in devices:
        device = torch.device(f"cuda:{device_index}")
        current = torch.as_tensor(normalized, dtype=torch.float64, device=device)
        time = torch.as_tensor(elapsed, dtype=torch.float64, device=device)
        weights = torch.as_tensor(artifact.weights, dtype=torch.float64, device=device)
        bias = torch.as_tensor(artifact.bias, dtype=torch.float64, device=device)
        basis_weights = torch.as_tensor(
            artifact.temporal_basis_weights, dtype=torch.float64, device=device
        )
        trend_weights = torch.as_tensor(
            artifact.proprioceptive_trend_weights, dtype=torch.float64, device=device
        )
        centers = torch.as_tensor(
            artifact.temporal_basis_centers_sec, dtype=torch.float64, device=device
        )
        memory = current[0].clone()
        output = []
        for row, timestamp in zip(current, time, strict=True):
            delta = row - memory
            radial = torch.exp(
                -0.5 * torch.square((timestamp - centers) / artifact.temporal_basis_width_sec)
            )
            output.append(weights @ row + bias + basis_weights @ radial + trend_weights @ delta)
            memory = memory + artifact.proprioceptive_memory_alpha * delta
        actual = torch.stack(output).cpu().numpy()
        maximum_error = max(maximum_error, float(np.max(np.abs(expected - actual))))
        hashes.append("sha256:" + hashlib.sha256(actual.tobytes()).hexdigest())
        names.append(torch.cuda.get_device_name(device_index))
    return G1TemporalGpuParity(
        requested_devices=devices,
        validated_devices=devices,
        device_names=tuple(names),
        maximum_absolute_error=maximum_error,
        output_hashes=tuple(hashes),
        passed=maximum_error <= 1e-10 and len(set(hashes)) == 1,
    )


def _numpy_temporal_logits(
    artifact: G1MuscleMemoryArtifact,
    normalized: np.ndarray,
    elapsed: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(artifact.weights)
    bias = np.asarray(artifact.bias)
    basis_weights = np.asarray(artifact.temporal_basis_weights)
    trend_weights = np.asarray(artifact.proprioceptive_trend_weights)
    centers = np.asarray(artifact.temporal_basis_centers_sec)
    memory = normalized[0].copy()
    output = []
    for row, timestamp in zip(normalized, elapsed, strict=True):
        delta = row - memory
        radial = np.exp(-0.5 * np.square((timestamp - centers) / artifact.temporal_basis_width_sec))
        output.append(weights @ row + bias + basis_weights @ radial + trend_weights @ delta)
        memory += artifact.proprioceptive_memory_alpha * delta
    return np.asarray(output)


def write_g1_temporal_muscle_memory_report(
    report: G1TemporalMuscleMemoryTrainingReport,
    *,
    output_dir: Path,
    source_checkout: Path,
) -> tuple[Path, Path]:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("temporal muscle-memory evidence must remain outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    artifact_path = root / "g1-temporal-muscle-memory.json"
    report_path = root / "g1-temporal-muscle-memory-training.json"
    _atomic_json(artifact_path, report.artifact.to_dict())
    _atomic_json(report_path, report.to_dict())
    return artifact_path, report_path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "G1TemporalExpertGeneralizationSummary",
    "G1TemporalGpuParity",
    "G1TemporalMuscleMemoryTrainer",
    "G1TemporalMuscleMemoryTrainingConfig",
    "G1TemporalMuscleMemoryTrainingReport",
    "build_g1_temporal_muscle_memory_cases",
    "write_g1_temporal_muscle_memory_report",
]
