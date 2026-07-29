"""Immutable contracts for continual residual control.

The contracts deliberately keep learning outside the synchronous Feedback
Plane.  A rollout may be consumed asynchronously, but one high-dynamic G1
episode is bound to exactly one policy version and one body/controller
snapshot.  This is the robot-control analogue of AReaL's per-sample version
tracking with a stricter no-mid-motion-switch rule.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _finite_mapping(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    normalized = {str(key): float(item) for key, item in value.items()}
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if any(not math.isfinite(item) for item in normalized.values()):
        raise ValueError(f"{label} must contain only finite values")
    return MappingProxyType(normalized)


class SkillPhase(StrEnum):
    """Motion boundaries used to prohibit unsafe weight changes."""

    STAND = "stand"
    PREPARE = "prepare"
    WEIGHT_TRANSFER = "weight_transfer"
    SWING = "swing"
    CONTACT = "contact"
    RECOVERY = "recovery"
    COMPLETE = "complete"


class ExperiencePartition(StrEnum):
    RECENT = "recent"
    ANCHOR = "anchor"
    BOUNDARY = "boundary"
    SELF = "self"


class ExperienceUse(StrEnum):
    """Permitted use of a trajectory relative to the learner version."""

    ACTOR_CRITIC_SELF = "actor_critic_self"
    CRITIC_SELF_ONLY = "critic_self_only"
    REJECT = "reject"


@dataclass(frozen=True)
class RewardVector:
    """Task and motion quality rewards; never hides safety costs."""

    task: float = 0.0
    tracking: float = 0.0
    balance: float = 0.0
    contact: float = 0.0
    learning: float = 0.0
    style: float = 0.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in self.to_dict().values()):
            raise ValueError("reward vector must contain only finite values")

    def to_dict(self) -> dict[str, float]:
        return {
            "task": self.task,
            "tracking": self.tracking,
            "balance": self.balance,
            "contact": self.contact,
            "learning": self.learning,
            "style": self.style,
        }


@dataclass(frozen=True)
class CostVector:
    """Independent non-negative safety and resource costs."""

    fall: float = 0.0
    joint_limit: float = 0.0
    torque: float = 0.0
    slip: float = 0.0
    energy: float = 0.0
    stale: float = 0.0
    collision: float = 0.0
    feedback_saturation: float = 0.0

    def __post_init__(self) -> None:
        values = self.to_dict().values()
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("cost vector must contain finite non-negative values")

    @property
    def critical(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.fall,
                self.joint_limit,
                self.torque,
                self.stale,
                self.collision,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "fall": self.fall,
            "joint_limit": self.joint_limit,
            "torque": self.torque,
            "slip": self.slip,
            "energy": self.energy,
            "stale": self.stale,
            "collision": self.collision,
            "feedback_saturation": self.feedback_saturation,
        }


@dataclass(frozen=True)
class PolicyVersion:
    """Content-addressed residual policy identity and rollback lineage."""

    version: int
    artifact_hash: str
    controller_snapshot_hash: str
    body_hash: str
    safety_kernel_hash: str
    observation_names: tuple[str, ...]
    residual_action_names: tuple[str, ...]
    parent_version_hash: str | None = None
    schema_version: str = "rosclaw.continual.policy_version.v1"

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("policy version must be non-negative")
        for label, value in (
            ("artifact_hash", self.artifact_hash),
            ("controller_snapshot_hash", self.controller_snapshot_hash),
            ("body_hash", self.body_hash),
            ("safety_kernel_hash", self.safety_kernel_hash),
        ):
            _require_hash(label, value)
        if self.version == 0 and self.parent_version_hash is not None:
            raise ValueError("policy version zero cannot have a parent")
        if self.version > 0:
            if self.parent_version_hash is None:
                raise ValueError("non-zero policy versions require a parent")
            _require_hash("parent_version_hash", self.parent_version_hash)
        for label, names in (
            ("observation_names", self.observation_names),
            ("residual_action_names", self.residual_action_names),
        ):
            if (
                not names
                or len(names) != len(set(names))
                or any(not name.strip() for name in names)
            ):
                raise ValueError(f"{label} must be non-empty, unique names")
        if len(self.residual_action_names) > 32:
            raise ValueError("residual policy action dimension must not exceed 32")

    @property
    def version_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "artifact_hash": self.artifact_hash,
            "parent_version_hash": self.parent_version_hash,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "body_hash": self.body_hash,
            "safety_kernel_hash": self.safety_kernel_hash,
            "observation_names": list(self.observation_names),
            "residual_action_names": list(self.residual_action_names),
        }


@dataclass(frozen=True)
class ControlSegment:
    """One version-pinned transition segment emitted by a rollout worker."""

    segment_id: str
    episode_id: str
    task_id: str
    phase: SkillPhase
    start_step: int
    end_step: int
    policy: PolicyVersion
    controller_snapshot_hash: str
    body_hash: str
    regime_hash: str
    self_state_hash: str
    observation: Mapping[str, float]
    residual_action: Mapping[str, float]
    next_observation: Mapping[str, float]
    behavior_logprob: float
    reward: RewardVector
    cost: CostVector
    terminal: bool = False
    schema_version: str = "rosclaw.continual.control_segment.v1"

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or not self.episode_id.strip() or not self.task_id.strip():
            raise ValueError("segment, episode, and task identifiers must not be empty")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("control segment steps must define a positive interval")
        for label, value in (
            ("controller_snapshot_hash", self.controller_snapshot_hash),
            ("body_hash", self.body_hash),
            ("regime_hash", self.regime_hash),
            ("self_state_hash", self.self_state_hash),
        ):
            _require_hash(label, value)
        if self.controller_snapshot_hash != self.policy.controller_snapshot_hash:
            raise ValueError("segment controller snapshot does not match its policy")
        if self.body_hash != self.policy.body_hash:
            raise ValueError("segment body does not match its policy")
        if not math.isfinite(self.behavior_logprob):
            raise ValueError("behavior_logprob must be finite")
        observation = _finite_mapping(self.observation, label="observation")
        next_observation = _finite_mapping(self.next_observation, label="next_observation")
        action = _finite_mapping(self.residual_action, label="residual_action")
        if tuple(observation) != self.policy.observation_names:
            raise ValueError("observation order must match the policy contract")
        if tuple(next_observation) != self.policy.observation_names:
            raise ValueError("next observation order must match the policy contract")
        if tuple(action) != self.policy.residual_action_names:
            raise ValueError("residual action order must match the policy contract")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "residual_action", action)

    @property
    def segment_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "segment_id": self.segment_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "phase": self.phase.value,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "policy": self.policy.to_dict(),
            "policy_version_hash": self.policy.version_hash,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "body_hash": self.body_hash,
            "regime_hash": self.regime_hash,
            "self_state_hash": self.self_state_hash,
            "observation": dict(self.observation),
            "residual_action": dict(self.residual_action),
            "next_observation": dict(self.next_observation),
            "behavior_logprob": self.behavior_logprob,
            "reward": self.reward.to_dict(),
            "cost": self.cost.to_dict(),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class VersionedTrajectory:
    """A complete episode that cannot straddle residual policy versions."""

    segments: tuple[ControlSegment, ...]
    strict_replay: bool
    schema_version: str = "rosclaw.continual.versioned_trajectory.v1"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("versioned trajectory must contain at least one segment")
        first = self.segments[0]
        if self.segments[-1].terminal is not True:
            raise ValueError("versioned trajectory must end in a terminal segment")
        expected_start = first.start_step
        phase_order = {phase: index for index, phase in enumerate(SkillPhase)}
        previous_phase = -1
        for segment in self.segments:
            if segment.start_step != expected_start:
                raise ValueError("trajectory segments must be contiguous")
            expected_start = segment.end_step
            if (
                segment.episode_id != first.episode_id
                or segment.task_id != first.task_id
                or segment.body_hash != first.body_hash
                or segment.regime_hash != first.regime_hash
                or segment.controller_snapshot_hash != first.controller_snapshot_hash
            ):
                raise ValueError("trajectory identity cannot change within an episode")
            if segment.policy.version_hash != first.policy.version_hash:
                raise ValueError("a high-dynamic episode cannot cross policy versions")
            current_phase = phase_order[segment.phase]
            if current_phase < previous_phase:
                raise ValueError("skill phases must not move backwards")
            previous_phase = current_phase
        if any(segment.terminal for segment in self.segments[:-1]):
            raise ValueError("only the final trajectory segment may be terminal")

    @property
    def policy(self) -> PolicyVersion:
        return self.segments[0].policy

    @property
    def trajectory_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def has_critical_cost(self) -> bool:
        return any(segment.cost.critical for segment in self.segments)

    def permitted_use(self, *, learner_version: int, max_policy_lag: int = 1) -> ExperienceUse:
        if max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        lag = learner_version - self.policy.version
        if lag < 0:
            return ExperienceUse.REJECT
        if lag <= max_policy_lag:
            return ExperienceUse.ACTOR_CRITIC_SELF
        return ExperienceUse.CRITIC_SELF_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "segments": [segment.to_dict() for segment in self.segments],
            "strict_replay": self.strict_replay,
        }


__all__ = [
    "ControlSegment",
    "CostVector",
    "ExperiencePartition",
    "ExperienceUse",
    "PolicyVersion",
    "RewardVector",
    "SkillPhase",
    "VersionedTrajectory",
]
