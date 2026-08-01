from __future__ import annotations

import json

from rosclaw.simforge.g1_contextual_recovery import (
    G1_CONTEXTUAL_RECOVERY_FEATURES,
    G1ContextualRecoveryArtifact,
    G1ContextualRecoveryPrimitive,
)
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
)
from rosclaw.simforge.g1_muscle_memory_cli import dispatch_muscle_memory_argv
from rosclaw.simforge.g1_recovery_state_memory import (
    G1_RECOVERY_STATE_FEATURES,
    G1_RECOVERY_STATE_OBSERVATIONS,
    G1RecoveryStateArtifact,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_dispatch_ignores_unrelated_commands() -> None:
    assert dispatch_muscle_memory_argv(["goalforge", "hat-trick", "run"]) is None


def test_inspect_validates_and_summarizes_safe_json(tmp_path, capsys) -> None:
    artifact = G1MuscleMemoryArtifact(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        parent_recovery_config_hash=_digest("3"),
        training_dataset_hash=_digest("4"),
        observation_mean=(0.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        weights=tuple(
            (0.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS) for _ in G1_MUSCLE_MEMORY_ACTIONS
        ),
        bias=(0.0,) * len(G1_MUSCLE_MEMORY_ACTIONS),
        action_limits_rad=(0.05,) * len(G1_MUSCLE_MEMORY_ACTIONS),
        training_episode_count=12,
        training_seed=7,
    )
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    result = dispatch_muscle_memory_argv(["goalforge", "muscle-memory", "inspect", str(path)])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["artifact_hash"] == artifact.artifact_hash
    assert payload["activation_ceiling"] == "SIM_ONLY"
    assert payload["expert_regime_feature_names"] == []
    assert payload["expert_regime_prototype_count"] == 0


def test_contextual_inspect_summarizes_bounded_primitive_commitments(tmp_path, capsys) -> None:
    artifact = G1ContextualRecoveryArtifact(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        baseline_recovery_config_hash=_digest("3"),
        fallback_recovery_config_hash=_digest("4"),
        training_dataset_hash=_digest("5"),
        observation_mean=(0.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        regime_feature_names=G1_CONTEXTUAL_RECOVERY_FEATURES,
        regime_prototypes=((0.0,) * len(G1_CONTEXTUAL_RECOVERY_FEATURES),),
        primitives=(
            G1ContextualRecoveryPrimitive(
                start_policy_frame=300,
                blend_frames=100,
                settling_start_policy_frame=400,
                settling_blend_frames=100,
                settling_standing_pose_blend=0.42,
                settling_waist_pitch_bias_rad=0.11,
                target_smoothing_alpha=0.54,
            ),
        ),
        maximum_regime_distance=0.25,
        maximum_feature_z=8.0,
        training_episode_count=20,
        training_seed=7,
    )
    path = tmp_path / "contextual.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    result = dispatch_muscle_memory_argv(
        ["goalforge", "muscle-memory", "contextual-inspect", str(path)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["artifact_hash"] == artifact.artifact_hash
    assert payload["prototype_count"] == 1
    assert payload["primitive_hashes"] == [artifact.primitives[0].primitive_hash]
    assert payload["activation_ceiling"] == "SIM_ONLY"


def test_state_inspect_summarizes_temporal_evidence_commitments(tmp_path, capsys) -> None:
    primitive = G1ContextualRecoveryPrimitive(
        start_policy_frame=300,
        blend_frames=100,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.42,
        settling_waist_pitch_bias_rad=0.11,
        target_smoothing_alpha=0.54,
    )
    width = 2 * len(G1_RECOVERY_STATE_FEATURES)
    artifact = G1RecoveryStateArtifact(
        body_hash=_digest("1"),
        motion_hash=_digest("2"),
        baseline_recovery_config_hash=_digest("3"),
        fallback_recovery_config_hash=_digest("4"),
        training_dataset_hash=_digest("5"),
        observation_mean=(0.0,) * len(G1_RECOVERY_STATE_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_RECOVERY_STATE_OBSERVATIONS),
        descriptor_feature_names=G1_RECOVERY_STATE_FEATURES,
        descriptor_prototypes=((0.0,) * width, (0.1,) * width, (0.2,) * width),
        prototype_primitive_indices=(0, 0, -1),
        prototype_composite_advantages=(0.10, 0.08, -0.05),
        prototype_component_minimums=(0.01, 0.02, -0.10),
        primitives=(primitive,),
        selection_window_frames=5,
        neighbor_count=3,
        maximum_neighbor_distance=0.25,
        minimum_primitive_consensus=2.0 / 3.0,
        minimum_advantage_lower_bound=0.02,
        minimum_component_lower_bound=-0.02,
        maximum_feature_z=8.0,
        training_episode_count=24,
        training_seed=17,
    )
    path = tmp_path / "state.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    result = dispatch_muscle_memory_argv(["goalforge", "muscle-memory", "state-inspect", str(path)])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["artifact_hash"] == artifact.artifact_hash
    assert payload["selection_window_frames"] == 5
    assert payload["positive_route_count"] == 2
    assert payload["abstention_prototype_count"] == 1
    assert payload["activation_ceiling"] == "SIM_ONLY"
