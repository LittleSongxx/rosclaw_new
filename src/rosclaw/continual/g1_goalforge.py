"""Adapters from real GoalForge MuJoCo rollouts to continual-RL contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from rosclaw.continual.contracts import (
    ControlSegment,
    CostVector,
    PolicyVersion,
    RewardVector,
    SkillPhase,
    VersionedTrajectory,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.self_model.contracts import (
    CapabilityBelief,
    ScalarBelief,
    SelfIdentity,
    SelfStateSnapshot,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    GOALFORGE_TASK_ID,
    GoalForgeStatus,
)

G1_CONTINUAL_OBSERVATIONS = (
    "torso_roll",
    "torso_pitch",
    "com_y_relative",
    "support_slip_m",
    "ball_lateral_error_m",
    "contact_phase",
    "energy_margin",
    "sensor_quality",
)
G1_CONTINUAL_ACTIONS = (
    "waist_roll_residual",
    "right_hip_roll_residual",
    "right_hip_yaw_residual",
    "kick_phase_rate",
)
G1_CONTINUAL_ACTION_LIMITS = (0.04, 0.08, 0.035, 0.08)

_ACTION_JOINT_INDICES = (13, 7, 8)


@dataclass(frozen=True)
class G1PolicyLineage:
    policies: tuple[PolicyVersion, ...]
    artifacts: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if len(self.policies) != len(self.artifacts) or not self.policies:
            raise ValueError("policy lineage policies and artifacts must align")
        for policy, artifact in zip(self.policies, self.artifacts, strict=True):
            if _digest(artifact) != policy.artifact_hash:
                raise ValueError("policy lineage artifact hash mismatch")

    def policy(self, version: int) -> PolicyVersion:
        try:
            return next(item for item in self.policies if item.version == version)
        except StopIteration as exc:
            raise KeyError(f"policy version {version} is absent") from exc

    def artifact(self, version: int) -> bytes:
        for policy, artifact in zip(self.policies, self.artifacts, strict=True):
            if policy.version == version:
                return artifact
        raise KeyError(f"policy artifact version {version} is absent")


@dataclass(frozen=True)
class G1TrajectoryAdaptation:
    trajectory: VersionedTrajectory
    self_identity_hash: str
    self_state_hashes: tuple[str, ...]
    source_trajectory_hash: str
    schema_version: str = "rosclaw.continual.g1_goalforge_adaptation.v1"


def build_g1_policy_lineage(
    *,
    body_hash: str,
    kick_prior_hash: str,
    motion_hash: str,
    backend_commit: str,
    torque_guard_scale: float,
    through_version: int = 2,
) -> G1PolicyLineage:
    """Build deterministic zero-residual parent identities for a qualified body."""

    if through_version < 0:
        raise ValueError("through_version must be non-negative")
    controller_hash = canonical_hash(
        {
            "kick_prior_hash": kick_prior_hash,
            "motion_hash": motion_hash,
            "backend_commit": backend_commit,
            "adapter": "rosclaw.g1_goalforge.fixed_prior.v1",
        }
    )
    safety_hash = canonical_hash(
        {
            "hard_torque_limits": list(G1_HARD_TORQUE_LIMITS),
            "torque_guard_scale": torque_guard_scale,
            "immutable": [
                "body_identity",
                "hard_torque_limits",
                "joint_limits",
                "permit",
                "lease",
                "evidence_semantics",
            ],
        }
    )
    policies: list[PolicyVersion] = []
    artifacts: list[bytes] = []
    parent: PolicyVersion | None = None
    for version in range(through_version + 1):
        artifact = json.dumps(
            {
                "schema_version": "rosclaw.continual.zero_residual_artifact.v1",
                "version": version,
                "parent_version_hash": parent.version_hash if parent else None,
                "observation_names": list(G1_CONTINUAL_OBSERVATIONS),
                "action_names": list(G1_CONTINUAL_ACTIONS),
                "action_limits": list(G1_CONTINUAL_ACTION_LIMITS),
                "semantics": "deterministic zero residual over frozen RoboNaldo prior",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        policy = PolicyVersion(
            version=version,
            artifact_hash=_digest(artifact),
            parent_version_hash=parent.version_hash if parent else None,
            controller_snapshot_hash=controller_hash,
            body_hash=body_hash,
            safety_kernel_hash=safety_hash,
            observation_names=G1_CONTINUAL_OBSERVATIONS,
            residual_action_names=G1_CONTINUAL_ACTIONS,
        )
        policies.append(policy)
        artifacts.append(artifact)
        parent = policy
    return G1PolicyLineage(tuple(policies), tuple(artifacts))


def adapt_goalforge_episode(
    episode: GoalForgeEpisode,
    *,
    policy: PolicyVersion,
    strict_replay: bool,
) -> G1TrajectoryAdaptation:
    """Convert a stepped MuJoCo trajectory into immutable control segments."""

    if not episode.result.physics_executed:
        raise ValueError("continual adaptation requires an executed physics episode")
    trace = episode.trajectory
    required = {
        "time",
        "joint_torque",
        "torso_quaternion",
        "com_y_relative",
        "support_foot_slip",
        "ball_lateral_error_m",
        "policy_phase",
        "contact_impulse",
    }
    missing = sorted(required - set(trace))
    if missing:
        raise ValueError("GoalForge trajectory lacks continual signals: " + ", ".join(missing))
    lengths = {len(np.asarray(trace[name])) for name in required}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("GoalForge continual signals must be aligned with at least two samples")
    sample_count = next(iter(lengths))
    source_hash = _trajectory_hash(trace)
    identity = _self_identity(policy)
    phases = _monotonic_phases(trace, sample_count)
    segments: list[ControlSegment] = []
    snapshots: list[str] = []
    for index in range(sample_count - 1):
        observation = _observation(trace, index)
        next_observation = _observation(trace, index + 1)
        action = _action(trace, index)
        snapshot = _self_snapshot(
            episode=episode,
            identity=identity,
            policy=policy,
            sequence=index,
            observation=observation,
            timestamp_sec=float(np.asarray(trace["time"])[index]),
        )
        snapshot_hash = snapshot.snapshot_hash
        snapshots.append(snapshot_hash)
        terminal = index == sample_count - 2
        segments.append(
            ControlSegment(
                segment_id=f"{episode.scenario.scenario_id}:{index:05d}",
                episode_id=episode.scenario.scenario_id,
                task_id=GOALFORGE_TASK_ID,
                phase=SkillPhase.COMPLETE if terminal else phases[index],
                start_step=index,
                end_step=index + 1,
                policy=policy,
                controller_snapshot_hash=policy.controller_snapshot_hash,
                body_hash=policy.body_hash,
                regime_hash=episode.scenario.scenario_commitment,
                self_state_hash=snapshot_hash,
                observation=observation,
                residual_action=action,
                next_observation=next_observation,
                behavior_logprob=0.0,
                reward=_reward(episode, trace, index, terminal, observation, action),
                cost=_cost(episode, trace, index, terminal, observation),
                terminal=terminal,
            )
        )
    trajectory = VersionedTrajectory(segments=tuple(segments), strict_replay=strict_replay)
    return G1TrajectoryAdaptation(
        trajectory=trajectory,
        self_identity_hash=identity.identity_hash,
        self_state_hashes=tuple(snapshots),
        source_trajectory_hash=source_hash,
    )


def _self_identity(policy: PolicyVersion) -> SelfIdentity:
    return SelfIdentity(
        body_hash=policy.body_hash,
        sensor_layout_hash=canonical_hash({"signals": list(G1_CONTINUAL_OBSERVATIONS)}),
        actuator_layout_hash=canonical_hash({"residuals": list(G1_CONTINUAL_ACTIONS)}),
        safety_kernel_hash=policy.safety_kernel_hash,
        controller_lineage=(policy.controller_snapshot_hash, policy.version_hash),
        current_policy_versions={GOALFORGE_TASK_ID: policy.version},
    )


def _self_snapshot(
    *,
    episode: GoalForgeEpisode,
    identity: SelfIdentity,
    policy: PolicyVersion,
    sequence: int,
    observation: dict[str, float],
    timestamp_sec: float,
) -> SelfStateSnapshot:
    joint_health = dict.fromkeys(G1_DDS_JOINT_NAMES, 1.0)
    motor = {name: ScalarBelief(1.0, 0.05, 0.80, "scale") for name in G1_DDS_JOINT_NAMES}
    zero = {
        name: ScalarBelief(
            episode.scenario.joint_zero_bias_rad,
            0.005,
            0.95,
            "rad",
        )
        for name in G1_DDS_JOINT_NAMES
    }
    return SelfStateSnapshot(
        identity_hash=identity.identity_hash,
        body_hash=policy.body_hash,
        sequence=sequence,
        timestamp_ns=max(0, int(round(timestamp_sec * 1_000_000_000.0))),
        joint_health=joint_health,
        motor_gain_beliefs=motor,
        joint_zero_bias_beliefs=zero,
        latency_belief=ScalarBelief(
            episode.scenario.control_latency_ms,
            1.0,
            0.95,
            "ms",
        ),
        friction_belief=ScalarBelief(
            episode.scenario.support_ground_friction,
            0.05,
            0.90,
            "coefficient",
        ),
        payload_belief=ScalarBelief(0.0, 1.0, 0.0, "kg"),
        balance_margin=0.11 - abs(observation["com_y_relative"]),
        energy_state=observation["energy_margin"],
        sensor_quality={"mujoco_state": observation["sensor_quality"]},
        capabilities={
            GOALFORGE_TASK_ID: CapabilityBelief(
                success_probability=0.5,
                uncertainty=1.0,
                evidence_count=0,
                policy_version=policy.version,
            )
        },
    )


def _observation(trace: dict[str, np.ndarray], index: int) -> dict[str, float]:
    quaternion = np.asarray(trace["torso_quaternion"][index], dtype=np.float64)
    roll, pitch = _roll_pitch(quaternion)
    torque = np.asarray(trace["joint_torque"][index], dtype=np.float64)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    ratio = float(np.max(np.abs(torque) / limits))
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            quaternion,
            torque,
            trace["com_y_relative"][index],
            trace["support_foot_slip"][index],
            trace["ball_lateral_error_m"][index],
            trace["policy_phase"][index],
        )
    )
    return {
        "torso_roll": roll,
        "torso_pitch": pitch,
        "com_y_relative": float(trace["com_y_relative"][index]),
        "support_slip_m": max(0.0, float(trace["support_foot_slip"][index])),
        "ball_lateral_error_m": float(trace["ball_lateral_error_m"][index]),
        "contact_phase": float(np.clip(trace["policy_phase"][index], 0.0, 1.0)),
        "energy_margin": float(np.clip(1.0 - ratio, 0.0, 1.0)),
        "sensor_quality": 1.0 if finite else 0.0,
    }


def _action(trace: dict[str, np.ndarray], index: int) -> dict[str, float]:
    residual = trace.get("combined_residual", trace.get("feedback_residual"))
    if residual is None:
        joint_values = np.zeros(3, dtype=np.float64)
    else:
        joint_values = np.asarray(residual[index], dtype=np.float64)[list(_ACTION_JOINT_INDICES)]
    phase_rate = (
        float(trace["feedback_phase_rate"][index]) if "feedback_phase_rate" in trace else 0.0
    )
    raw = (*map(float, joint_values), phase_rate)
    return {
        name: float(np.clip(value, -limit, limit))
        for name, value, limit in zip(
            G1_CONTINUAL_ACTIONS,
            raw,
            G1_CONTINUAL_ACTION_LIMITS,
            strict=True,
        )
    }


def _monotonic_phases(trace: dict[str, np.ndarray], count: int) -> tuple[SkillPhase, ...]:
    phases: list[SkillPhase] = []
    contact = np.asarray(trace["contact_impulse"], dtype=np.float64)
    for index in range(count):
        phase = float(np.clip(trace["policy_phase"][index], 0.0, 1.0))
        if phase < 0.05:
            value = SkillPhase.STAND
        elif phase < 0.25:
            value = SkillPhase.PREPARE
        elif phase < 0.38:
            value = SkillPhase.WEIGHT_TRANSFER
        elif contact[index] <= 0.0 and phase < 0.58:
            value = SkillPhase.SWING
        elif contact[index] > 0.0 and phase < 0.62:
            value = SkillPhase.CONTACT
        elif phase < 0.95:
            value = SkillPhase.RECOVERY
        else:
            value = SkillPhase.COMPLETE
        if phases and list(SkillPhase).index(value) < list(SkillPhase).index(phases[-1]):
            value = phases[-1]
        phases.append(value)
    return tuple(phases)


def _reward(
    episode: GoalForgeEpisode,
    trace: dict[str, np.ndarray],
    index: int,
    terminal: bool,
    observation: dict[str, float],
    action: dict[str, float],
) -> RewardVector:
    impulse = np.asarray(trace["contact_impulse"], dtype=np.float64)
    new_contact = impulse[index + 1] > impulse[index] + 1e-12
    attitude = math.hypot(observation["torso_roll"], observation["torso_pitch"])
    balance = float(np.clip((0.11 - abs(observation["com_y_relative"])) / 0.11, -1.0, 1.0))
    style = -sum(
        (action[name] / limit) ** 2
        for name, limit in zip(G1_CONTINUAL_ACTIONS, G1_CONTINUAL_ACTION_LIMITS, strict=True)
    ) / len(G1_CONTINUAL_ACTIONS)
    return RewardVector(
        task=1.0 if terminal and episode.result.success else 0.0,
        tracking=-attitude,
        balance=balance,
        contact=1.0 if new_contact else 0.0,
        learning=0.0,
        style=style,
    )


def _cost(
    episode: GoalForgeEpisode,
    trace: dict[str, np.ndarray],
    index: int,
    terminal: bool,
    observation: dict[str, float],
) -> CostVector:
    saturation = 0.0
    if "combined_residual_saturation" in trace:
        values = np.asarray(trace["combined_residual_saturation"], dtype=np.float64)
        saturation = max(0.0, float(values[index + 1] - values[index]))
    result = episode.result
    return CostVector(
        fall=1.0 if terminal and result.post_kick_fall else 0.0,
        joint_limit=1.0 if terminal and result.joint_limit_violation else 0.0,
        torque=(
            1.0
            if terminal and (result.torque_limit_violation or result.actuator_saturation)
            else 0.0
        ),
        slip=observation["support_slip_m"],
        energy=1.0 - observation["energy_margin"],
        stale=0.0,
        collision=(
            1.0 if terminal and result.status is GoalForgeStatus.WRONG_FOOT_CONTACT else 0.0
        ),
        feedback_saturation=saturation,
    )


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _trajectory_hash(trajectory: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(trajectory):
        value = np.ascontiguousarray(trajectory[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "G1_CONTINUAL_ACTIONS",
    "G1_CONTINUAL_ACTION_LIMITS",
    "G1_CONTINUAL_OBSERVATIONS",
    "G1PolicyLineage",
    "G1TrajectoryAdaptation",
    "adapt_goalforge_episode",
    "build_g1_policy_lineage",
]
