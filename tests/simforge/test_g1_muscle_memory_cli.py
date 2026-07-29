from __future__ import annotations

import json

from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
    G1MuscleMemoryArtifact,
)
from rosclaw.simforge.g1_muscle_memory_cli import dispatch_muscle_memory_argv


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
