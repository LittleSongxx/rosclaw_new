"""SIM-only learned post-contact muscle memory for G1 GoalForge.

The frozen RoboNaldo prior remains responsible for the kick.  This module
loads a small, content-addressed proprioceptive policy that can add only
bounded joint-target residuals after observed ball contact and kick-foot
landing.  It contains no transport, torque, Registry, or hardware path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.profiles.g1 import g1_joint_residual_limits
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_json,
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 1024 * 1024

G1_MUSCLE_MEMORY_OBSERVATIONS = (
    "post_contact_time_sec",
    "policy_phase",
    "pelvis_velocity_x_m_s",
    "pelvis_velocity_y_m_s",
    "pelvis_velocity_z_m_s",
    "torso_roll_rad",
    "torso_pitch_rad",
    "torso_angular_velocity_x_rad_s",
    "torso_angular_velocity_y_rad_s",
    "torso_angular_velocity_z_rad_s",
    "com_y_relative_m",
    "left_support",
    "right_support",
    "left_ground_force_scale",
    "right_ground_force_scale",
    "contact_impulse_ns",
)

G1_MUSCLE_MEMORY_ACTIONS = (
    "sagittal_common",
    "sagittal_split",
    "lateral_common",
    "lateral_split",
    "leg_absorption",
    "waist_pitch",
    "waist_roll",
    "arm_pitch_counter",
    "arm_roll_counter",
)


@dataclass(frozen=True)
class G1MuscleMemoryArtifact:
    """Portable NumPy policy trained from post-contact MuJoCo returns."""

    body_hash: str
    motion_hash: str
    parent_recovery_config_hash: str
    training_dataset_hash: str
    observation_mean: tuple[float, ...]
    observation_scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    action_limits_rad: tuple[float, ...]
    training_episode_count: int
    training_seed: int
    output_smoothing_alpha: float = 0.35
    output_rate_limit_rad: float = 0.012
    maximum_feature_z: float = 6.0
    activation_duration_sec: float = 1.6
    fade_out_sec: float = 0.4
    sagittal_capture_deadband_m_s: float = 0.02
    sagittal_capture_full_scale_m_s: float = 0.20
    sagittal_minimum_impulse_ns: float = 1.75
    sagittal_impulse_ramp_ns: float = 0.40
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.muscle_memory_artifact.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("motion_hash", self.motion_hash),
            ("parent_recovery_config_hash", self.parent_recovery_config_hash),
            ("training_dataset_hash", self.training_dataset_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        observations = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
        actions = len(G1_MUSCLE_MEMORY_ACTIONS)
        if (
            len(self.observation_mean) != observations
            or len(self.observation_scale) != observations
        ):
            raise ValueError("muscle-memory observation normalization shape is invalid")
        if len(self.weights) != actions or any(len(row) != observations for row in self.weights):
            raise ValueError("muscle-memory weight matrix shape is invalid")
        if len(self.bias) != actions or len(self.action_limits_rad) != actions:
            raise ValueError("muscle-memory action shape is invalid")
        arrays = (
            np.asarray(self.observation_mean),
            np.asarray(self.observation_scale),
            np.asarray(self.weights),
            np.asarray(self.bias),
            np.asarray(self.action_limits_rad),
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("muscle-memory artifact contains non-finite values")
        if np.any(np.asarray(self.observation_scale) <= 0.0):
            raise ValueError("muscle-memory observation scales must be positive")
        limits = np.asarray(self.action_limits_rad)
        if np.any(limits <= 0.0) or np.any(limits > 0.08):
            raise ValueError("muscle-memory action limits must be in (0, 0.08]")
        if self.training_episode_count <= 0:
            raise ValueError("muscle-memory training requires at least one physics episode")
        if self.training_seed < 0:
            raise ValueError("muscle-memory training seed must be non-negative")
        if not 0.05 <= self.output_smoothing_alpha <= 1.0:
            raise ValueError("muscle-memory smoothing alpha must be in [0.05, 1]")
        if not 0.001 <= self.output_rate_limit_rad <= 0.04:
            raise ValueError("muscle-memory output rate limit must be in [0.001, 0.04]")
        if not 2.0 <= self.maximum_feature_z <= 12.0:
            raise ValueError("muscle-memory feature envelope must be in [2, 12]")
        if not 0.4 <= self.activation_duration_sec <= 4.0:
            raise ValueError("muscle-memory activation duration must be in [0.4, 4] seconds")
        if not 0.1 <= self.fade_out_sec <= min(1.0, self.activation_duration_sec):
            raise ValueError("muscle-memory fade out must fit inside the activation window")
        if not 0.0 <= self.sagittal_capture_deadband_m_s <= 0.20:
            raise ValueError("muscle-memory sagittal capture deadband is invalid")
        if not 0.05 <= self.sagittal_capture_full_scale_m_s <= 0.50:
            raise ValueError("muscle-memory sagittal capture scale is invalid")
        if not 0.0 <= self.sagittal_minimum_impulse_ns <= 4.0:
            raise ValueError("muscle-memory minimum contact impulse is invalid")
        if not 0.1 <= self.sagittal_impulse_ramp_ns <= 2.0:
            raise ValueError("muscle-memory contact impulse ramp is invalid")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("learned G1 muscle memory must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation_names"] = list(G1_MUSCLE_MEMORY_OBSERVATIONS)
        value["action_names"] = list(G1_MUSCLE_MEMORY_ACTIONS)
        value["observation_mean"] = list(self.observation_mean)
        value["observation_scale"] = list(self.observation_scale)
        value["weights"] = [list(row) for row in self.weights]
        value["bias"] = list(self.bias)
        value["action_limits_rad"] = list(self.action_limits_rad)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G1MuscleMemoryArtifact:
        if value.get("schema_version") != "rosclaw.g1_goalforge.muscle_memory_artifact.v1":
            raise ValueError("unsupported G1 muscle-memory artifact schema")
        if tuple(value.get("observation_names", ())) != G1_MUSCLE_MEMORY_OBSERVATIONS:
            raise ValueError("muscle-memory observation contract mismatch")
        if tuple(value.get("action_names", ())) != G1_MUSCLE_MEMORY_ACTIONS:
            raise ValueError("muscle-memory action contract mismatch")
        return cls(
            body_hash=str(value["body_hash"]),
            motion_hash=str(value["motion_hash"]),
            parent_recovery_config_hash=str(value["parent_recovery_config_hash"]),
            training_dataset_hash=str(value["training_dataset_hash"]),
            observation_mean=tuple(float(item) for item in value["observation_mean"]),
            observation_scale=tuple(float(item) for item in value["observation_scale"]),
            weights=tuple(tuple(float(item) for item in row) for row in value["weights"]),
            bias=tuple(float(item) for item in value["bias"]),
            action_limits_rad=tuple(float(item) for item in value["action_limits_rad"]),
            training_episode_count=int(value["training_episode_count"]),
            training_seed=int(value["training_seed"]),
            output_smoothing_alpha=float(value.get("output_smoothing_alpha", 0.35)),
            output_rate_limit_rad=float(value.get("output_rate_limit_rad", 0.012)),
            maximum_feature_z=float(value.get("maximum_feature_z", 6.0)),
            activation_duration_sec=float(value.get("activation_duration_sec", 1.6)),
            fade_out_sec=float(value.get("fade_out_sec", 0.4)),
            sagittal_capture_deadband_m_s=float(value.get("sagittal_capture_deadband_m_s", 0.02)),
            sagittal_capture_full_scale_m_s=float(
                value.get("sagittal_capture_full_scale_m_s", 0.20)
            ),
            sagittal_minimum_impulse_ns=float(value.get("sagittal_minimum_impulse_ns", 1.75)),
            sagittal_impulse_ramp_ns=float(value.get("sagittal_impulse_ramp_ns", 0.40)),
            activation_ceiling=str(value.get("activation_ceiling", "")),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class G1MuscleMemoryEffect:
    residual: np.ndarray
    synergy_actions: np.ndarray
    active: bool
    out_of_distribution: bool
    saturated_joint_count: int
    residual_rms_rad: float


@dataclass(frozen=True)
class G1MuscleMemoryReceipt:
    artifact_hash: str
    body_hash: str
    motion_hash: str
    parent_recovery_config_hash: str
    training_dataset_hash: str
    inference_count: int
    active_count: int
    out_of_distribution_count: int
    saturated_joint_count: int
    peak_residual_rms_rad: float
    peak_joint_residual_rad: float
    trace_hash: str
    activation_ceiling: str
    hardware_command_sent: bool = False
    evidence_domain: str = "SIM"
    schema_version: str = "rosclaw.g1_goalforge.muscle_memory_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class G1MuscleMemoryPolicy:
    """Allocation-light state feedback over fixed whole-body synergies."""

    def __init__(self, artifact: G1MuscleMemoryArtifact) -> None:
        self.artifact = artifact
        self._mean = np.asarray(artifact.observation_mean, dtype=np.float64)
        self._scale = np.asarray(artifact.observation_scale, dtype=np.float64)
        self._weights = np.asarray(artifact.weights, dtype=np.float64)
        self._bias = np.asarray(artifact.bias, dtype=np.float64)
        self._action_limits = np.asarray(artifact.action_limits_rad, dtype=np.float64)
        self._synergies = _muscle_synergies()
        self._joint_limits = np.asarray(
            g1_joint_residual_limits(G1_DDS_JOINT_NAMES), dtype=np.float64
        )
        self.reset()

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        parent_recovery_config_hash: str,
    ) -> None:
        if body_hash != self.artifact.body_hash:
            raise ValueError("muscle-memory Body hash mismatch")
        if motion_hash != self.artifact.motion_hash:
            raise ValueError("muscle-memory motion hash mismatch")
        if parent_recovery_config_hash != self.artifact.parent_recovery_config_hash:
            raise ValueError("muscle-memory parent recovery config hash mismatch")

    def reset(self) -> None:
        self._previous_residual = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
        self._activation_origin_sec: float | None = None
        self._inference_count = 0
        self._active_count = 0
        self._ood_count = 0
        self._saturated_joint_count = 0
        self._peak_residual_rms = 0.0
        self._peak_joint_residual = 0.0
        self._trace: list[dict[str, Any]] = []

    def infer(self, observation: Mapping[str, float]) -> G1MuscleMemoryEffect:
        missing = set(G1_MUSCLE_MEMORY_OBSERVATIONS).difference(observation)
        if missing:
            raise ValueError(f"muscle-memory inference is missing observations: {sorted(missing)}")
        ordered = np.asarray(
            [float(observation[name]) for name in G1_MUSCLE_MEMORY_OBSERVATIONS],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(ordered)):
            raise ValueError("muscle-memory observations must be finite")
        normalized = (ordered - self._mean) / self._scale
        ood = bool(np.max(np.abs(normalized)) > self.artifact.maximum_feature_z)
        if ood:
            raw_actions = np.zeros(len(G1_MUSCLE_MEMORY_ACTIONS), dtype=np.float64)
            raw_residual = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
            self._ood_count += 1
        else:
            post_contact_time_sec = ordered[
                G1_MUSCLE_MEMORY_OBSERVATIONS.index("post_contact_time_sec")
            ]
            if self._activation_origin_sec is None:
                self._activation_origin_sec = post_contact_time_sec
            envelope = self._activation_envelope(
                max(0.0, post_contact_time_sec - self._activation_origin_sec)
            )
            raw_actions = envelope * np.tanh(self._weights @ normalized + self._bias)
            velocity_x = ordered[G1_MUSCLE_MEMORY_OBSERVATIONS.index("pelvis_velocity_x_m_s")]
            capture_fraction = np.clip(
                (-velocity_x - self.artifact.sagittal_capture_deadband_m_s)
                / self.artifact.sagittal_capture_full_scale_m_s,
                0.0,
                1.0,
            )
            contact_impulse = ordered[G1_MUSCLE_MEMORY_OBSERVATIONS.index("contact_impulse_ns")]
            impulse_fraction = np.clip(
                (contact_impulse - self.artifact.sagittal_minimum_impulse_ns)
                / self.artifact.sagittal_impulse_ramp_ns,
                0.0,
                1.0,
            )
            raw_actions[G1_MUSCLE_MEMORY_ACTIONS.index("sagittal_common")] *= (
                capture_fraction * impulse_fraction
            )
            raw_residual = (raw_actions * self._action_limits) @ self._synergies
        clipped = np.clip(raw_residual, -self._joint_limits, self._joint_limits)
        saturated = int(np.count_nonzero(np.abs(clipped - raw_residual) > 1e-12))
        delta = np.clip(
            clipped - self._previous_residual,
            -self.artifact.output_rate_limit_rad,
            self.artifact.output_rate_limit_rad,
        )
        rate_limited = self._previous_residual + delta
        alpha = self.artifact.output_smoothing_alpha
        residual = self._previous_residual + alpha * (rate_limited - self._previous_residual)
        residual = np.clip(residual, -self._joint_limits, self._joint_limits)
        self._previous_residual = residual.copy()
        rms = float(np.sqrt(np.mean(np.square(residual))))
        active = bool(np.any(np.abs(residual) > 1e-9))
        self._inference_count += 1
        self._active_count += int(active)
        self._saturated_joint_count += saturated
        self._peak_residual_rms = max(self._peak_residual_rms, rms)
        self._peak_joint_residual = max(
            self._peak_joint_residual,
            float(np.max(np.abs(residual))),
        )
        self._trace.append(
            {
                "observation": list(map(float, ordered)),
                "synergy_actions": list(map(float, raw_actions)),
                "residual_rms_rad": rms,
                "out_of_distribution": ood,
            }
        )
        return G1MuscleMemoryEffect(
            residual=residual,
            synergy_actions=raw_actions,
            active=active,
            out_of_distribution=ood,
            saturated_joint_count=saturated,
            residual_rms_rad=rms,
        )

    def _activation_envelope(self, post_contact_time_sec: float) -> float:
        remaining = self.artifact.activation_duration_sec - post_contact_time_sec
        if remaining <= 0.0:
            return 0.0
        if remaining >= self.artifact.fade_out_sec:
            return 1.0
        linear = remaining / self.artifact.fade_out_sec
        return linear * linear * (3.0 - 2.0 * linear)

    def build_receipt(self) -> G1MuscleMemoryReceipt:
        return G1MuscleMemoryReceipt(
            artifact_hash=self.artifact.artifact_hash,
            body_hash=self.artifact.body_hash,
            motion_hash=self.artifact.motion_hash,
            parent_recovery_config_hash=self.artifact.parent_recovery_config_hash,
            training_dataset_hash=self.artifact.training_dataset_hash,
            inference_count=self._inference_count,
            active_count=self._active_count,
            out_of_distribution_count=self._ood_count,
            saturated_joint_count=self._saturated_joint_count,
            peak_residual_rms_rad=self._peak_residual_rms,
            peak_joint_residual_rad=self._peak_joint_residual,
            trace_hash=hash_json({"trace": self._trace}),
            activation_ceiling=self.artifact.activation_ceiling,
        )


def load_g1_muscle_memory_artifact(path: Path) -> G1MuscleMemoryArtifact:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("muscle-memory artifact is missing or exceeds 1 MiB")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("muscle-memory artifact is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("muscle-memory artifact root must be an object")
    return G1MuscleMemoryArtifact.from_dict(value)


def _muscle_synergies() -> np.ndarray:
    index = {name: position for position, name in enumerate(G1_DDS_JOINT_NAMES)}
    value = np.zeros(
        (len(G1_MUSCLE_MEMORY_ACTIONS), len(G1_DDS_JOINT_NAMES)),
        dtype=np.float64,
    )

    def set_joint(action: str, joint: str, weight: float) -> None:
        value[G1_MUSCLE_MEMORY_ACTIONS.index(action), index[joint]] = weight

    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        set_joint("sagittal_common", f"{side}_hip_pitch_joint", 1.0)
        set_joint("sagittal_common", f"{side}_knee_joint", -0.70)
        set_joint("sagittal_common", f"{side}_ankle_pitch_joint", 0.45)
        set_joint("sagittal_split", f"{side}_hip_pitch_joint", sign)
        set_joint("sagittal_split", f"{side}_knee_joint", -0.55 * sign)
        set_joint("sagittal_split", f"{side}_ankle_pitch_joint", 0.35 * sign)
        set_joint("lateral_common", f"{side}_hip_roll_joint", 1.0)
        set_joint("lateral_common", f"{side}_ankle_roll_joint", -0.65)
        set_joint("lateral_split", f"{side}_hip_roll_joint", sign)
        set_joint("lateral_split", f"{side}_ankle_roll_joint", -0.65 * sign)
        set_joint("leg_absorption", f"{side}_hip_pitch_joint", -0.45)
        set_joint("leg_absorption", f"{side}_knee_joint", 1.0)
        set_joint("leg_absorption", f"{side}_ankle_pitch_joint", -0.55)
    set_joint("waist_pitch", "waist_pitch_joint", 1.0)
    set_joint("waist_roll", "waist_roll_joint", 1.0)
    set_joint("arm_pitch_counter", "left_shoulder_pitch_joint", 1.0)
    set_joint("arm_pitch_counter", "right_shoulder_pitch_joint", -1.0)
    set_joint("arm_pitch_counter", "left_elbow_joint", -0.35)
    set_joint("arm_pitch_counter", "right_elbow_joint", 0.35)
    set_joint("arm_roll_counter", "left_shoulder_roll_joint", 1.0)
    set_joint("arm_roll_counter", "right_shoulder_roll_joint", -1.0)
    value.setflags(write=False)
    return value


__all__ = [
    "G1_MUSCLE_MEMORY_ACTIONS",
    "G1_MUSCLE_MEMORY_OBSERVATIONS",
    "G1MuscleMemoryArtifact",
    "G1MuscleMemoryEffect",
    "G1MuscleMemoryPolicy",
    "G1MuscleMemoryReceipt",
    "load_g1_muscle_memory_artifact",
]
