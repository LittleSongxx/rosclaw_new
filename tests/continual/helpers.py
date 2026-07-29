from __future__ import annotations

import hashlib

from rosclaw.continual.contracts import (
    ControlSegment,
    CostVector,
    PolicyVersion,
    RewardVector,
    SkillPhase,
    VersionedTrajectory,
)


def digest(value: bytes | str) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def policy(version: int, *, parent: PolicyVersion | None = None) -> tuple[PolicyVersion, bytes]:
    artifact = f"policy-{version}".encode()
    value = PolicyVersion(
        version=version,
        artifact_hash=digest(artifact),
        parent_version_hash=parent.version_hash if parent else None,
        controller_snapshot_hash=digest("controller"),
        body_hash=digest("body"),
        safety_kernel_hash=digest("safety"),
        observation_names=("roll", "pitch"),
        residual_action_names=("pelvis_roll",),
    )
    return value, artifact


def trajectory(
    value: PolicyVersion,
    *,
    episode: str = "episode-1",
    critical: bool = False,
) -> VersionedTrajectory:
    first = segment(value, episode=episode, start=0, end=1, phase=SkillPhase.PREPARE)
    final = segment(
        value,
        episode=episode,
        start=1,
        end=2,
        phase=SkillPhase.RECOVERY,
        terminal=True,
        critical=critical,
    )
    return VersionedTrajectory((first, final), strict_replay=True)


def segment(
    value: PolicyVersion,
    *,
    episode: str,
    start: int,
    end: int,
    phase: SkillPhase,
    terminal: bool = False,
    critical: bool = False,
) -> ControlSegment:
    return ControlSegment(
        segment_id=f"{episode}:{start}",
        episode_id=episode,
        task_id="g1_target_kick",
        phase=phase,
        start_step=start,
        end_step=end,
        policy=value,
        controller_snapshot_hash=value.controller_snapshot_hash,
        body_hash=value.body_hash,
        regime_hash=digest("regime"),
        self_state_hash=digest("self-state"),
        observation={"roll": 0.1, "pitch": -0.1},
        residual_action={"pelvis_roll": -0.01},
        next_observation={"roll": 0.08, "pitch": -0.09},
        behavior_logprob=-0.5,
        reward=RewardVector(task=1.0, balance=0.2),
        cost=CostVector(fall=1.0 if critical else 0.0, energy=0.1),
        terminal=terminal,
    )
