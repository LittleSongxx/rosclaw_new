"""Real-MuJoCo closed-loop validation for recoverable continual services."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.continual.boundary_feedback import extract_boundary_replay_requests
from rosclaw.continual.contracts import ExperiencePartition, PolicyVersion, SkillPhase
from rosclaw.continual.g1_goalforge import (
    G1_CONTINUAL_ACTION_LIMITS,
    adapt_goalforge_episode,
    build_g1_policy_lineage,
)
from rosclaw.continual.learner import ConstrainedResidualSAC, ResidualSACConfig
from rosclaw.continual.serde import policy_version_from_dict
from rosclaw.continual.services.experience import ExperienceService
from rosclaw.continual.services.inference import InferenceService
from rosclaw.continual.services.learner import (
    LearnerService,
    ResidualSACServiceExecutor,
)
from rosclaw.continual.services.persistence import (
    atomic_write_json,
    require_external_service_root,
)
from rosclaw.continual.services.rollout import RolloutService
from rosclaw.continual.services.weight_update import WeightUpdateService
from rosclaw.continual.stability import (
    ContinualCandidateEvidence,
    StabilityPlasticityGate,
    TaskRetention,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.self_model.agency import AgencyEstimator, AgencyEvidence
from rosclaw.self_model.forward_model import (
    ForwardAction,
    ForwardModelInput,
    ForwardPrediction,
    ForwardState,
    HybridForwardSelfModel,
)
from rosclaw.self_model.prediction_monitor import (
    AdaptationState,
    AdaptationTrigger,
    PredictionResiduals,
)
from rosclaw.self_model.regime import RegimeEncoder, RegimeMemory, RegimeObservation
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_continual_physical_validation import (
    build_foundation_scenarios,
    record_goalforge_experience,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES, ShotParameters


def run_g1_service_validation(
    *,
    asset_root: Path,
    candidate_artifact_path: Path,
    matched_report_path: Path,
    state_root: Path,
    output_path: Path,
    source_checkout: Path,
    learner_device: str = "cpu",
    learner_updates: int = 1,
) -> dict[str, Any]:
    """Exercise all five services against physics and fail-closed candidate evidence."""

    checkout = source_checkout.expanduser().resolve()
    root = require_external_service_root(state_root, checkout)
    output = output_path.expanduser().resolve()
    require_external_service_root(output.parent, checkout)
    if root.exists():
        raise FileExistsError(f"service validation state root already exists: {root}")
    if output.exists():
        raise FileExistsError(f"service validation output already exists: {output}")
    if not 1 <= learner_updates <= 20:
        raise ValueError("service validation learner updates must be in [1, 20]")
    report_bytes = matched_report_path.expanduser().resolve().read_bytes()
    source_evidence_hash = _digest(report_bytes)
    report = json.loads(report_bytes)
    if not isinstance(report, dict):
        raise ValueError("matched candidate report must be a JSON object")
    boundary_requests = extract_boundary_replay_requests(
        report,
        source_evidence_hash=source_evidence_hash,
    )
    parent_from_report = policy_version_from_dict(_mapping(report, "parent_policy"))
    candidate = policy_version_from_dict(_mapping(report, "candidate_policy"))
    candidate_artifact = candidate_artifact_path.expanduser().resolve().read_bytes()
    if _digest(candidate_artifact) != candidate.artifact_hash:
        raise ValueError("candidate artifact does not match the matched report")

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
    if parent.version_hash != parent_from_report.version_hash:
        raise ValueError("matched report parent does not match qualified local G1 assets")
    if candidate.parent_version_hash != parent.version_hash:
        raise ValueError("matched candidate is not the direct successor of the local parent")

    root.mkdir(parents=True)
    trajectory_root = root / "trajectories"
    trajectory_root.mkdir()

    trigger = _open_shadow_learning(report)
    rollout = RolloutService(root, source_checkout=checkout)
    inference = InferenceService(
        root,
        source_checkout=checkout,
        active=parent,
        active_artifact=lineage.artifact(2),
    )
    experience = ExperienceService(root, source_checkout=checkout)
    regime_encoder = RegimeEncoder(tuple(G1_DDS_JOINT_NAMES))
    scenarios = build_foundation_scenarios()
    parameters = ShotParameters()
    rollout_evidence: list[dict[str, Any]] = []
    first_trace: dict[str, np.ndarray] | None = None
    first_episode_hash: str | None = None
    for partition in ExperiencePartition:
        scenario = scenarios[partition]
        assignment = rollout.assign(
            episode_id=scenario.scenario_id,
            scenario_commitment=scenario.scenario_commitment,
            policy=parent,
        )
        rollout.start(assignment.assignment_id, worker_id="local-mujoco-worker")
        lease = inference.begin_motion(
            episode_id=scenario.scenario_id,
            phase=SkillPhase.PREPARE,
        )
        episode = backend.run(scenario, parameters)
        replay = backend.run(scenario, parameters)
        strict_replay = bool(
            episode.result.summary_dict() == replay.result.summary_dict()
            and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
        )
        adaptation = adapt_goalforge_episode(
            episode,
            policy=parent,
            strict_replay=strict_replay,
        )
        completed = rollout.complete(
            assignment.assignment_id,
            trajectory=adaptation.trajectory,
        )
        inference.end_motion(lease.lease_id)
        record = record_goalforge_experience(
            partition,
            adaptation.trajectory,
            scenario.scenario_commitment,
        )
        experience.append(record)
        path = trajectory_root / f"{partition.value}.npz"
        np.savez_compressed(path, **episode.trajectory)
        rollout_evidence.append(
            {
                "partition": partition.value,
                "scenario_id": scenario.scenario_id,
                "scenario_commitment": scenario.scenario_commitment,
                "assignment_id": completed.assignment_id,
                "policy_version_hash": completed.policy.version_hash,
                "version_switch_count": completed.version_switch_count,
                "strict_replay": completed.strict_replay,
                "physics_steps": episode.result.physics_steps,
                "status": episode.result.status.value,
                "trajectory_hash": adaptation.trajectory.trajectory_hash,
                "trajectory_path": str(path),
                "critical_cost": adaptation.trajectory.has_critical_cost,
            }
        )
        _observe_regime(regime_encoder, scenario)
        if first_trace is None:
            first_trace = episode.trajectory
            first_episode_hash = adaptation.trajectory.trajectory_hash

    for request in boundary_requests:
        experience.enqueue_boundary(request)
    experience_before_recovery = experience.audit_receipt()
    experience.close()
    with ExperienceService(root, source_checkout=checkout) as recovered_experience:
        experience_after_recovery = recovered_experience.audit_receipt()
        batch = recovered_experience.sample(
            batch_size=64,
            learner_version=parent.version,
            seed=7107,
        )

    if first_trace is None or first_episode_hash is None:
        raise RuntimeError("service validation did not produce a physical trajectory")
    forward = _calibrate_forward_model(
        first_trace, shadow_learning=trigger.state is AdaptationState.SHADOW_LEARNING
    )
    regime_estimate = regime_encoder.estimate()
    regime_assignment = RegimeMemory().assign(regime_estimate.persistent)
    boundary_scenario = scenarios[ExperiencePartition.BOUNDARY]
    agency = AgencyEstimator().estimate(
        AgencyEvidence(
            action_magnitude=0.7,
            prediction_error=min(1.0, forward["error_after"] * 100.0),
            external_force_evidence=min(1.0, boundary_scenario.disturbance_n / 80.0),
            sensor_inconsistency=min(1.0, boundary_scenario.observation_noise_m / 0.08),
            action_hash=canonical_hash(asdict(parameters)),
            predicted_outcome_hash=canonical_hash(
                {"forward_model_hash": forward["model_hash"], "episode": first_episode_hash}
            ),
            observed_outcome_hash=first_episode_hash,
            timestamp_ns=0,
        )
    )

    learner = ConstrainedResidualSAC(
        ResidualSACConfig(
            observation_names=parent.observation_names,
            action_names=parent.residual_action_names,
            action_limits=G1_CONTINUAL_ACTION_LIMITS,
            hidden_dims=(64, 64),
            batch_size=64,
            seed=7107,
            device=learner_device,
        )
    )
    learner_service = LearnerService(root, source_checkout=checkout, parent=parent)
    learned = learner_service.execute(
        batch,
        executor=ResidualSACServiceExecutor(
            learner,
            parent=parent,
            updates_per_job=learner_updates,
        ),
    )
    recovered_learner = LearnerService(root, source_checkout=checkout, parent=parent)

    crash_assignment = rollout.assign(
        episode_id="crash-injection-rollout",
        scenario_commitment=canonical_hash({"scenario": "SIM crash recovery injection"}),
        policy=parent,
    )
    rollout.start(crash_assignment.assignment_id, worker_id="crash-injection-worker")
    inference.begin_motion(
        episode_id="crash-injection-inference",
        phase=SkillPhase.PREPARE,
    )
    recovered_rollout = RolloutService(root, source_checkout=checkout)
    recovered_inference = InferenceService(root, source_checkout=checkout)

    updater = WeightUpdateService(
        root,
        source_checkout=checkout,
        inference=recovered_inference,
    )
    published = updater.publish(candidate, artifact=candidate_artifact)
    verified = updater.verify()
    staged = updater.stage()
    gate_report = StabilityPlasticityGate().evaluate(
        _matched_gate_evidence(report, parent=parent, candidate=candidate)
    )
    activation = updater.activate(phase=SkillPhase.COMPLETE, gate_report=gate_report)
    active_parent_unchanged = recovered_inference.active.version_hash == parent.version_hash

    candidate_update = trigger.candidate_update(
        sample_count=int(_mapping(report, "counts")["total"]),
        target_improvement=float(
            _mapping(report, "paired_statistics")["target_error_improvement_m"]
        ),
        anchor_degradation=max(
            0.0,
            _aggregate(report, "active_parent_v2", "anchor", "success_rate")
            - _aggregate(report, "candidate_v3", "anchor", "success_rate"),
        ),
        critical_safety_regressions=int(_mapping(report, "gate")["critical_safety_regressions"]),
        converged=True,
    )

    checks = {
        "four_physics_partitions": len(rollout_evidence) == 4
        and all(item["physics_steps"] > 0 for item in rollout_evidence),
        "strict_replay": all(item["strict_replay"] for item in rollout_evidence),
        "motion_version_switches_zero": all(
            item["version_switch_count"] == 0 for item in rollout_evidence
        ),
        "experience_catalog_recovered": (
            experience_before_recovery["catalog_counts"]
            == experience_after_recovery["catalog_counts"]
        ),
        "boundary_regressions_enqueued": len(boundary_requests)
        == int(_mapping(report, "gate")["critical_safety_regressions"]),
        "learner_checkpoint_recovered": learned.batch_hash
        in recovered_learner.completed_batch_hashes,
        "forward_calibration_error_decreased": forward["error_after"] < forward["error_before"],
        "rollout_crash_aborted_without_replay": recovered_rollout.recovered_abort_count == 1,
        "inference_crash_aborted_without_replay": recovered_inference.recovered_abort_count == 1,
        "unsafe_candidate_not_activated": not gate_report.activation_allowed
        and active_parent_unchanged,
        "adaptation_rolled_back": candidate_update.state is AdaptationState.ROLLBACK,
    }
    result: dict[str, Any] = {
        "schema_version": "rosclaw.continual.g1_service_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "source": {
            "matched_report": str(matched_report_path.expanduser().resolve()),
            "matched_report_hash": source_evidence_hash,
            "candidate_artifact": str(candidate_artifact_path.expanduser().resolve()),
            "candidate_artifact_hash": candidate.artifact_hash,
            "qualified_body_hash": qualification.body_hash,
            "backend_commit": qualification.backend_commit,
        },
        "rollout_service": {
            "rollouts": rollout_evidence,
            "event_count": len(recovered_rollout.log.events),
            "recovered_abort_count": recovered_rollout.recovered_abort_count,
        },
        "experience_service": {
            "before_recovery": experience_before_recovery,
            "after_recovery": experience_after_recovery,
            "boundary_requests": [request.to_dict() for request in boundary_requests],
            "batch_hash": batch.batch_hash,
        },
        "learner_service": {
            **asdict(learned),
            "checkpoint_recovered": learned.batch_hash in recovered_learner.completed_batch_hashes,
            "device": learner_device,
        },
        "inference_service": {
            "active_policy_hash": recovered_inference.active.version_hash,
            "candidate_policy_hash": (
                recovered_inference.candidate.version_hash
                if recovered_inference.candidate is not None
                else None
            ),
            "active_parent_unchanged": active_parent_unchanged,
            "recovered_abort_count": recovered_inference.recovered_abort_count,
        },
        "weight_update_service": {
            "publish": published.to_dict(),
            "verify": verified.to_dict(),
            "stage": staged.to_dict(),
            "activation": activation.to_dict(),
            "gate": _gate_to_dict(gate_report),
        },
        "adaptation": asdict(candidate_update),
        "forward_self_model": forward,
        "regime": {
            "fast": regime_estimate.fast.to_dict(),
            "episode": regime_estimate.episode.to_dict(),
            "persistent": regime_estimate.persistent.to_dict(),
            "expert_assignment": asdict(regime_assignment),
        },
        "agency": agency.to_dict(),
        "claims": {
            "evidence_domain": "SIM",
            "real_mujoco_physics": True,
            "service_state_external_to_checkout": True,
            "candidate_activated": False,
            "registry_mutated": False,
            "dds_opened": False,
            "hardware_authorized": False,
            "consciousness_claimed": False,
            "operational_embodied_self_model": True,
        },
    }
    result["report_hash"] = canonical_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    return result


def _open_shadow_learning(report: dict[str, Any]) -> AdaptationTrigger:
    critical = int(_mapping(report, "gate")["critical_safety_regressions"])
    if critical <= 0:
        raise ValueError("this validation requires rejected safety-boundary evidence")
    trigger = AdaptationTrigger()
    for index in range(5):
        trigger.observe(
            PredictionResiduals(
                body_state=1.0,
                contact_outcome=1.0,
                contact_mode=1.0,
                control_latency=0.0,
                energy=0.0,
                task_performance=min(
                    1.0,
                    _aggregate(report, "candidate_v3", "all", "mean_penalized_target_error_m")
                    / 2.0,
                ),
                timestamp_ns=index,
                episode_id="matched-candidate-safety-shift",
            )
        )
    if trigger.state is not AdaptationState.CONFIRMED_SHIFT:
        raise RuntimeError("persistent matched safety residual did not confirm adaptation shift")
    trigger.begin_shadow_learning()
    return trigger


def _observe_regime(regime: RegimeEncoder, scenario: Any) -> None:
    for index in range(4):
        regime.observe(
            RegimeObservation(
                episode_id=scenario.scenario_id,
                timestamp_ns=index,
                support_friction=scenario.support_ground_friction,
                ball_friction=scenario.ball_ground_friction,
                control_latency_ms=scenario.control_latency_ms,
                joint_zero_bias=dict.fromkeys(G1_DDS_JOINT_NAMES, scenario.joint_zero_bias_rad),
                motor_gain=dict.fromkeys(G1_DDS_JOINT_NAMES, 1.0),
                payload_kg=max(0.0, scenario.ball_mass_kg - 0.41),
                disturbance_magnitude=scenario.disturbance_n / 80.0,
                sensor_confidence=max(0.0, 1.0 - scenario.observation_noise_m / 0.08),
            )
        )
    regime.end_episode(scenario.scenario_id)


def _calibrate_forward_model(
    trace: dict[str, np.ndarray], *, shadow_learning: bool
) -> dict[str, Any]:
    sample_count = len(np.asarray(trace["time"]))
    stride = max(1, (sample_count - 3) // 96)
    indices = tuple(range(0, sample_count - 3, stride))[:96]
    if len(indices) < 16:
        raise ValueError("physical trace is too short for forward-model calibration")
    pairs = [(_forward_input(trace, index), _forward_state(trace, index + 1)) for index in indices]
    model = HybridForwardSelfModel(
        tuple(G1_DDS_JOINT_NAMES),
        hidden_size=64,
        residual_limit=0.05,
        learning_rate=0.2,
        seed=7107,
    )
    before = _mean_prediction_error(model, pairs)
    learning_receipts = []
    for _ in range(8):
        learning_receipts.extend(
            model.learn_transition(model_input, actual, shadow_learning=shadow_learning)
            for model_input, actual in pairs
        )
    after = _mean_prediction_error(model, pairs)
    return {
        "schema_version": "rosclaw.self.forward_model_validation.v1",
        "trace_samples": sample_count,
        "calibration_transitions": len(pairs),
        "shadow_learning": shadow_learning,
        "trained_updates": sum(receipt.trained for receipt in learning_receipts),
        "error_before": before,
        "error_after": after,
        "error_reduction": before - after,
        "model_hash": model.model_hash,
        "bounded_residual_limit": model.residual_limit,
        "evaluation_semantics": "within-trace shadow calibration; not an unseen-regime benchmark",
    }


def _forward_input(trace: dict[str, np.ndarray], index: int) -> ForwardModelInput:
    state = _forward_state(trace, index)
    dt = _dt(trace, index)
    velocity = np.asarray(trace["joint_velocity"], dtype=float)
    acceleration = (velocity[index + 1] - velocity[index]) / dt
    pelvis_velocity = _pelvis_velocity(trace, index)
    next_pelvis_velocity = _pelvis_velocity(trace, index + 1)
    pelvis_acceleration = (next_pelvis_velocity - pelvis_velocity) / dt
    ball_velocity = np.asarray(trace["ball_velocity"], dtype=float)
    ball_impulse = ball_velocity[index + 1, :3] - ball_velocity[index, :3]
    contact = (
        _scalar(trace["left_foot_contact"], index),
        _scalar(trace["right_foot_contact"], index),
    )
    return ForwardModelInput(
        state=state,
        action=ForwardAction(
            joint_acceleration=dict(zip(G1_DDS_JOINT_NAMES, acceleration.tolist(), strict=True)),
            pelvis_acceleration=tuple(pelvis_acceleration.tolist()),
            ball_impulse=tuple(ball_impulse.tolist()),
        ),
        dt_seconds=dt,
        phase_progress=float(np.clip(_scalar(trace["policy_phase"], index), 0.0, 1.0)),
        contact_mode=tuple(float(np.clip(value, 0.0, 1.0)) for value in contact),
    )


def _forward_state(trace: dict[str, np.ndarray], index: int) -> ForwardState:
    position = np.asarray(trace["joint_position"], dtype=float)[index]
    velocity = np.asarray(trace["joint_velocity"], dtype=float)[index]
    pelvis = np.asarray(trace["pelvis_pose"], dtype=float)[index, :3]
    com = np.asarray(trace["com"], dtype=float)[index, :3]
    ball = np.asarray(trace["ball_pose"], dtype=float)[index, :3]
    ball_velocity = np.asarray(trace["ball_velocity"], dtype=float)[index, :3]
    sample_count = len(np.asarray(trace["time"]))
    return ForwardState(
        joint_position=dict(zip(G1_DDS_JOINT_NAMES, position.tolist(), strict=True)),
        joint_velocity=dict(zip(G1_DDS_JOINT_NAMES, velocity.tolist(), strict=True)),
        pelvis_position=tuple(pelvis.tolist()),
        pelvis_velocity=tuple(_pelvis_velocity(trace, index).tolist()),
        com_position=tuple(com.tolist()),
        foot_contact=(
            float(np.clip(_scalar(trace["left_foot_contact"], index), 0.0, 1.0)),
            float(np.clip(_scalar(trace["right_foot_contact"], index), 0.0, 1.0)),
        ),
        ball_position=tuple(ball.tolist()),
        ball_velocity=tuple(ball_velocity.tolist()),
        energy_state=max(0.0, 1.0 - index / (2.0 * sample_count)),
        balance_margin=0.08 - abs(_scalar(trace["com_y_relative"], index)),
    )


def _mean_prediction_error(
    model: HybridForwardSelfModel,
    pairs: list[tuple[ForwardModelInput, ForwardState]],
) -> float:
    return sum(
        _prediction_error(model.predict(model_input), actual) for model_input, actual in pairs
    ) / len(pairs)


def _prediction_error(prediction: ForwardPrediction, actual: ForwardState) -> float:
    predicted = prediction.next_state
    values = [
        *(
            predicted.joint_position[name] - actual.joint_position[name]
            for name in G1_DDS_JOINT_NAMES
        ),
        *(
            predicted.joint_velocity[name] - actual.joint_velocity[name]
            for name in G1_DDS_JOINT_NAMES
        ),
        *(
            left - right
            for left, right in zip(predicted.pelvis_position, actual.pelvis_position, strict=True)
        ),
        *(
            left - right
            for left, right in zip(predicted.com_position, actual.com_position, strict=True)
        ),
        *(
            left - right
            for left, right in zip(predicted.ball_position, actual.ball_position, strict=True)
        ),
        predicted.energy_state - actual.energy_state,
        predicted.balance_margin - actual.balance_margin,
    ]
    return sum(value * value for value in values) / len(values)


def _pelvis_velocity(trace: dict[str, np.ndarray], index: int) -> np.ndarray:
    pose = np.asarray(trace["pelvis_pose"], dtype=float)
    return (pose[index + 1, :3] - pose[index, :3]) / _dt(trace, index)


def _dt(trace: dict[str, np.ndarray], index: int) -> float:
    time = np.asarray(trace["time"], dtype=float)
    return float(np.clip(time[index + 1] - time[index], 1e-6, 0.1))


def _scalar(value: np.ndarray, index: int) -> float:
    return float(np.asarray(value, dtype=float)[index].reshape(-1)[0])


def _matched_gate_evidence(
    report: dict[str, Any], *, parent: PolicyVersion, candidate: PolicyVersion
) -> ContinualCandidateEvidence:
    counts = _mapping(report, "counts")
    return ContinualCandidateEvidence(
        parent_policy_hash=parent.artifact_hash,
        candidate_policy_hash=candidate.artifact_hash,
        body_hash=candidate.body_hash,
        parent_body_hash=parent.body_hash,
        safety_kernel_hash=candidate.safety_kernel_hash,
        parent_safety_kernel_hash=parent.safety_kernel_hash,
        task_retention=(
            TaskRetention(
                "anchor_kick",
                _aggregate(report, "active_parent_v2", "anchor", "success_rate"),
                _aggregate(report, "candidate_v3", "anchor", "success_rate"),
                critical=True,
            ),
            TaskRetention(
                "all_matched_scenarios",
                _aggregate(report, "active_parent_v2", "all", "success_rate"),
                _aggregate(report, "candidate_v3", "all", "success_rate"),
            ),
        ),
        plasticity=None,
        self_core=None,
        replay_recent_count=int(counts["recent"]),
        replay_anchor_count=int(counts["anchor"]),
        replay_boundary_count=int(counts["boundary"]),
        replay_self_count=int(counts["self"]),
        anchor_action_drift_rms=_aggregate(
            report, "candidate_v3", "anchor", "anchor_action_drift_rms"
        ),
        critical_safety_regressions=int(_mapping(report, "gate")["critical_safety_regressions"]),
        stale_action_executions=0,
        old_version_replays=0,
        candidate_evaluation_complete=True,
    )


def _gate_to_dict(report: Any) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "decision": report.decision.value,
        "checks": [
            {"name": item.name, "status": item.status.value, "detail": item.detail}
            for item in report.checks
        ],
        "parent_policy_hash": report.parent_policy_hash,
        "candidate_policy_hash": report.candidate_policy_hash,
        "rollback_target_hash": report.rollback_target_hash,
        "activation_allowed": report.activation_allowed,
        "evidence_domain": report.evidence_domain,
        "report_hash": report.report_hash,
    }


def _aggregate(report: dict[str, Any], arm: str, partition: str, metric: str) -> float:
    aggregates = _mapping(report, "aggregates")
    arm_value = _mapping(aggregates, arm)
    partition_value = _mapping(arm_value, partition)
    value = float(partition_value[metric])
    if not math.isfinite(value):
        raise ValueError(f"matched report aggregate is not finite: {arm}/{partition}/{metric}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be a JSON object")
    return result


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = ["run_g1_service_validation"]
