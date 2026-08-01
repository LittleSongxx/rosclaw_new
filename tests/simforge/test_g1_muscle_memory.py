from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw.simforge.g1_cerebellar_recovery import (
    G1CerebellarRecoveryConfig,
    G1CerebellarRecoveryController,
)
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
    G1MuscleMemoryPolicy,
    load_g1_muscle_memory_artifact,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _artifact(*, parent_config_hash: str, active: bool = True) -> G1MuscleMemoryArtifact:
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    return G1MuscleMemoryArtifact(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        parent_recovery_config_hash=parent_config_hash,
        training_dataset_hash=_digest("4"),
        observation_mean=(0.0,) * observations,
        observation_scale=(1.0,) * observations,
        weights=tuple((0.0,) * observations for _ in range(actions)),
        bias=(1.0 if active else 0.0,) + (0.0,) * (actions - 1),
        action_limits_rad=(0.05,) * actions,
        training_episode_count=48,
        training_seed=20260730,
    )


def _observation(**changes: float) -> dict[str, float]:
    value = dict.fromkeys(G1_MUSCLE_MEMORY_OBSERVATIONS, 0.0)
    value.update(changes)
    return value


def _controller(
    artifact: G1MuscleMemoryArtifact | None = None,
) -> G1CerebellarRecoveryController:
    return G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        muscle_memory_artifact=artifact,
    )


def test_artifact_roundtrips_as_safe_json(tmp_path) -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)
    path = tmp_path / "muscle-memory.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    loaded = load_g1_muscle_memory_artifact(path)

    assert loaded == artifact
    assert loaded.artifact_hash == artifact.artifact_hash
    assert loaded.activation_ceiling == "SIM_ONLY"


def test_v1_artifact_serialization_omits_temporal_fields() -> None:
    artifact = _artifact(parent_config_hash=_controller().config_hash)
    payload = artifact.to_dict()

    assert payload["schema_version"].endswith(".v1")
    assert "policy_architecture" not in payload
    assert "temporal_basis_weights" not in payload


def test_policy_is_rate_limited_bounded_and_ood_fail_closed() -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)
    policy = G1MuscleMemoryPolicy(artifact)

    first = policy.infer(_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0))
    second = policy.infer(_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0))
    policy.reset()
    ood = policy.infer(_observation(pelvis_velocity_x_m_s=100.0))

    assert first.active
    assert np.max(np.abs(first.residual)) <= artifact.output_rate_limit_rad + 1e-12
    assert np.max(np.abs(second.residual - first.residual)) <= (
        artifact.output_rate_limit_rad + 1e-12
    )
    assert ood.out_of_distribution
    assert not ood.active
    assert np.count_nonzero(ood.residual) == 0
    receipt = policy.build_receipt()
    assert receipt.inference_count == 1
    assert receipt.out_of_distribution_count == 1
    assert not receipt.hardware_command_sent


def test_temporal_policy_uses_history_and_reset_is_deterministic() -> None:
    base = _artifact(parent_config_hash=_controller().config_hash)
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    basis = np.zeros((actions, 3), dtype=np.float64)
    trend = np.zeros((actions, observations), dtype=np.float64)
    velocity_index = G1_MUSCLE_MEMORY_OBSERVATIONS.index("pelvis_velocity_x_m_s")
    trend[0, velocity_index] = 0.5
    artifact = replace(
        base,
        schema_version="rosclaw.g1_goalforge.muscle_memory_artifact.v2",
        policy_architecture="leaky_rbf_recurrent_v1",
        temporal_basis_centers_sec=(0.0, 0.3, 0.6),
        temporal_basis_weights=tuple(tuple(map(float, row)) for row in basis),
        proprioceptive_trend_weights=tuple(tuple(map(float, row)) for row in trend),
        fallback_recovery_config_hash=_digest("5"),
        expert_impact_prototypes_ns=(3.0,),
        structured_recovery_parameters=(0.42, 0.11, 0.54),
    )
    payload = artifact.to_dict()
    policy = G1MuscleMemoryPolicy(artifact)
    first = policy.infer(_observation(pelvis_velocity_x_m_s=-0.05, contact_impulse_ns=3.0))
    second = policy.infer(_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0))

    assert not np.array_equal(first.synergy_actions, second.synergy_actions)
    assert "expert_regime_feature_names" not in payload
    assert "expert_regime_prototypes" not in payload
    policy.reset()
    replay = policy.infer(_observation(pelvis_velocity_x_m_s=-0.05, contact_impulse_ns=3.0))
    assert np.array_equal(first.residual, replay.residual)


def test_temporal_policy_falls_back_whole_body_on_low_confidence_impact() -> None:
    base = _artifact(parent_config_hash=_controller().config_hash)
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    basis = np.ones((actions, 3), dtype=np.float64)
    trend = np.zeros((actions, observations), dtype=np.float64)
    artifact = replace(
        base,
        schema_version="rosclaw.g1_goalforge.muscle_memory_artifact.v2",
        policy_architecture="leaky_rbf_recurrent_v1",
        temporal_basis_centers_sec=(0.0, 0.3, 0.6),
        temporal_basis_weights=tuple(tuple(map(float, row)) for row in basis),
        proprioceptive_trend_weights=tuple(tuple(map(float, row)) for row in trend),
        fallback_recovery_config_hash=_digest("5"),
        expert_impact_prototypes_ns=(3.0,),
        structured_recovery_parameters=(0.42, 0.11, 0.54),
    )

    effect = G1MuscleMemoryPolicy(artifact).infer(
        _observation(
            pelvis_velocity_x_m_s=-0.20,
            contact_impulse_ns=artifact.sagittal_minimum_impulse_ns - 0.1,
        )
    )

    assert not effect.active
    assert np.count_nonzero(effect.synergy_actions) == 0


def test_v3_router_requires_matching_impact_and_proprioceptive_regime() -> None:
    base = _artifact(parent_config_hash=_controller().config_hash)
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    prototype_observation = _observation(contact_impulse_ns=3.0)
    prototype = tuple(
        prototype_observation[name] for name in G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES
    )
    artifact = replace(
        base,
        schema_version="rosclaw.g1_goalforge.muscle_memory_artifact.v3",
        policy_architecture="leaky_rbf_recurrent_v1",
        temporal_basis_centers_sec=(0.0, 0.3, 0.6),
        temporal_basis_weights=tuple((0.0,) * 3 for _ in range(actions)),
        proprioceptive_trend_weights=tuple((0.0,) * observations for _ in range(actions)),
        fallback_recovery_config_hash=_digest("5"),
        expert_impact_prototypes_ns=(3.0,),
        expert_regime_feature_names=G1_MUSCLE_MEMORY_EXPERT_REGIME_FEATURES,
        expert_regime_prototypes=(prototype,),
        expert_regime_max_distance=0.25,
        structured_recovery_parameters=(0.42, 0.11, 0.54),
    )
    policy = G1MuscleMemoryPolicy(artifact)

    assert G1MuscleMemoryArtifact.from_dict(artifact.to_dict()) == artifact
    assert policy.expert_regime_confident(prototype_observation)
    assert not policy.expert_regime_confident(
        _observation(contact_impulse_ns=3.0, torso_pitch_rad=0.30)
    )
    assert not policy.expert_regime_confident(_observation(contact_impulse_ns=2.7))
    first = policy.infer(prototype_observation)
    after_route = policy.infer(_observation(contact_impulse_ns=3.0, torso_pitch_rad=0.30))
    assert not first.out_of_distribution
    assert not after_route.out_of_distribution


def test_temporal_controller_routes_low_impulse_to_bound_fallback() -> None:
    primary = G1CerebellarRecoveryConfig(target_smoothing_alpha=0.54)
    fallback = G1CerebellarRecoveryConfig(target_smoothing_alpha=0.60)
    primary_controller = G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=primary,
    )
    fallback_controller = G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=fallback,
    )
    base = _artifact(parent_config_hash=primary_controller.config_hash)
    actions = len(G1_MUSCLE_MEMORY_ACTIONS)
    observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    artifact = replace(
        base,
        schema_version="rosclaw.g1_goalforge.muscle_memory_artifact.v2",
        policy_architecture="leaky_rbf_recurrent_v1",
        temporal_basis_centers_sec=(0.0, 0.3, 0.6),
        temporal_basis_weights=tuple((0.0,) * 3 for _ in range(actions)),
        proprioceptive_trend_weights=tuple((0.0,) * observations for _ in range(actions)),
        fallback_recovery_config_hash=fallback_controller.config_hash,
        expert_impact_prototypes_ns=(3.0,),
        structured_recovery_parameters=(0.0, 0.0, 0.54),
    )
    controller = G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=primary,
        muscle_memory_artifact=artifact,
        fallback_config=fallback,
    )

    controller.adapt_target(
        target=np.zeros(29),
        policy_frame=500,
        timestamp_sec=5.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(contact_impulse_ns=1.0),
    )
    receipt = controller.build_receipt(strict_replay=True)

    assert receipt.fallback_config_hash == fallback_controller.config_hash
    assert receipt.fallback_routed_count == 1

    controller.reset()
    controller.adapt_target(
        target=np.zeros(29),
        policy_frame=500,
        timestamp_sec=5.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(contact_impulse_ns=3.0),
    )
    expert_receipt = controller.build_receipt(strict_replay=True)
    assert expert_receipt.expert_route_latched is True
    assert expert_receipt.fallback_routed_count == 0

    with pytest.raises(ValueError, match="structured recovery parameters mismatch"):
        G1CerebellarRecoveryController(
            body_hash=_digest("1"),
            motion_hash=_digest("2"),
            regime_commitment=_digest("3"),
            regime_eligible=True,
            regime_reasons=(),
            standing_pose=np.zeros(29),
            config=primary,
            muscle_memory_artifact=replace(
                artifact,
                structured_recovery_parameters=(0.42, 0.11, 0.54),
            ),
            fallback_config=fallback,
        )


def test_policy_fades_out_instead_of_shifting_the_terminal_standing_pose() -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)
    policy = G1MuscleMemoryPolicy(artifact)

    policy.infer(_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0))
    for _ in range(100):
        expired = policy.infer(
            _observation(
                post_contact_time_sec=artifact.activation_duration_sec + 0.1,
                pelvis_velocity_x_m_s=-0.20,
                contact_impulse_ns=3.0,
            )
        )

    assert np.max(np.abs(expired.residual)) < 1e-12
    assert np.count_nonzero(expired.synergy_actions) == 0


def test_sagittal_capture_reflex_requires_confident_contact_impulse() -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)
    policy = G1MuscleMemoryPolicy(artifact)

    effect = policy.infer(
        _observation(
            pelvis_velocity_x_m_s=-0.30,
            contact_impulse_ns=artifact.sagittal_minimum_impulse_ns - 0.1,
        )
    )

    assert not effect.active
    assert np.count_nonzero(effect.synergy_actions) == 0


def test_cerebellar_controller_keeps_learned_policy_behind_contact_and_landing() -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)
    controller = _controller(artifact)
    target = np.zeros(29)

    before_contact = controller.adapt_target(
        target=target,
        policy_frame=500,
        timestamp_sec=5.0,
        ball_contact_detected=False,
        left_support=True,
        right_support=False,
        muscle_memory_observation=_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0),
    )
    before_landing = controller.adapt_target(
        target=target,
        policy_frame=500,
        timestamp_sec=5.1,
        ball_contact_detected=True,
        left_support=True,
        right_support=False,
        muscle_memory_observation=_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0),
    )
    active = controller.adapt_target(
        target=target,
        policy_frame=500,
        timestamp_sec=5.2,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(pelvis_velocity_x_m_s=-0.20, contact_impulse_ns=3.0),
    )

    assert not before_contact.muscle_memory_active
    assert not before_landing.muscle_memory_active
    assert np.array_equal(before_contact.target, before_landing.target)
    assert active.muscle_memory_active
    assert not np.array_equal(active.target, before_landing.target)
    receipt = controller.build_receipt(strict_replay=True)
    assert receipt.muscle_memory_receipt is not None
    assert receipt.muscle_memory_receipt["artifact_hash"] == artifact.artifact_hash
    assert receipt.muscle_memory_receipt["inference_count"] == 1


def test_cerebellar_smoothing_does_not_integrate_learned_residual() -> None:
    config = G1CerebellarRecoveryConfig(
        start_policy_frame=0,
        blend_frames=1,
        standing_pose_blend=0.0,
        roll_posture_bias_rad=0.0,
        target_smoothing_alpha=0.50,
        target_smoothing_start_policy_frame=0,
    )
    parent = G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=config,
    )
    artifact = _artifact(parent_config_hash=parent.config_hash)
    controller = G1CerebellarRecoveryController(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        regime_commitment=_digest("3"),
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=config,
        muscle_memory_artifact=artifact,
    )
    reference_policy = G1MuscleMemoryPolicy(artifact)
    observation = _observation(
        pelvis_velocity_x_m_s=-0.20,
        contact_impulse_ns=3.0,
    )

    first = controller.adapt_target(
        target=np.zeros(29),
        policy_frame=1,
        timestamp_sec=0.02,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=observation,
    )
    expected_first = reference_policy.infer(observation)
    second = controller.adapt_target(
        target=np.zeros(29),
        policy_frame=2,
        timestamp_sec=0.04,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=observation,
    )
    expected_second = reference_policy.infer(observation)

    np.testing.assert_allclose(first.target, expected_first.residual)
    np.testing.assert_allclose(second.target, expected_second.residual)


def test_artifact_rejects_identity_and_shape_tampering() -> None:
    parent = _controller()
    artifact = _artifact(parent_config_hash=parent.config_hash)

    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(artifact, activation_ceiling="REAL")
    with pytest.raises(ValueError, match="weight matrix"):
        replace(artifact, weights=artifact.weights[:-1])
    with pytest.raises(ValueError, match="Body hash mismatch"):
        G1MuscleMemoryPolicy(artifact).require_compatible(
            body_hash=_digest("9"),
            motion_hash=artifact.motion_hash,
            parent_recovery_config_hash=artifact.parent_recovery_config_hash,
        )
