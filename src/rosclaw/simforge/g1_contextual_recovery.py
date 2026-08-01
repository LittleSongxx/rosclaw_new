"""SIM-only contextual motor-primitive memory for G1 post-kick recovery.

The learned object maps the proprioceptive landing state to one of a bounded
set of already validated recovery primitives.  It cannot emit torque, raw
joint targets, ROS/DDS commands, or a hardware authorization.  Out-of-
distribution states deterministically route to the retained parent primitive.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.g1_muscle_memory import G1_MUSCLE_MEMORY_OBSERVATIONS
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 1024 * 1024
G1_CONTEXTUAL_RECOVERY_FEATURES = G1_MUSCLE_MEMORY_OBSERVATIONS


@dataclass(frozen=True)
class G1ContextualRecoveryPrimitive:
    """Bounded parameters for one contact-and-landing recovery primitive."""

    start_policy_frame: int
    blend_frames: int
    settling_start_policy_frame: int
    settling_blend_frames: int
    settling_standing_pose_blend: float
    settling_waist_pitch_bias_rad: float
    target_smoothing_alpha: float
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_primitive.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "rosclaw.g1_goalforge.contextual_recovery_primitive.v1":
            raise ValueError("unsupported contextual recovery primitive schema")
        counts = (
            self.start_policy_frame,
            self.blend_frames,
            self.settling_start_policy_frame,
            self.settling_blend_frames,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("contextual recovery frame parameters must be integers")
        if self.start_policy_frame < 0 or self.blend_frames <= 0:
            raise ValueError("contextual recovery start/blend frames are invalid")
        if self.settling_start_policy_frame < self.start_policy_frame + self.blend_frames:
            raise ValueError("contextual settling cannot overlap the unloading blend")
        if self.settling_blend_frames <= 0:
            raise ValueError("contextual settling blend frames must be positive")
        values = (
            self.settling_standing_pose_blend,
            self.settling_waist_pitch_bias_rad,
            self.target_smoothing_alpha,
        )
        if any(isinstance(value, bool) for value in values) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            raise ValueError("contextual recovery parameters must be finite")
        if not 0.0 <= self.settling_standing_pose_blend <= 0.50:
            raise ValueError("contextual standing blend must be in [0, 0.50]")
        if not -0.12 <= self.settling_waist_pitch_bias_rad <= 0.12:
            raise ValueError("contextual waist pitch must be in [-0.12, 0.12]")
        if not 0.25 <= self.target_smoothing_alpha <= 1.0:
            raise ValueError("contextual smoothing alpha must be in [0.25, 1]")

    @property
    def primitive_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G1ContextualRecoveryPrimitive:
        expected = {
            "schema_version",
            "start_policy_frame",
            "blend_frames",
            "settling_start_policy_frame",
            "settling_blend_frames",
            "settling_standing_pose_blend",
            "settling_waist_pitch_bias_rad",
            "target_smoothing_alpha",
        }
        if set(value) != expected:
            raise ValueError("contextual recovery primitive fields are invalid")
        return cls(
            schema_version=str(value["schema_version"]),
            start_policy_frame=_strict_int(value["start_policy_frame"]),
            blend_frames=_strict_int(value["blend_frames"]),
            settling_start_policy_frame=_strict_int(value["settling_start_policy_frame"]),
            settling_blend_frames=_strict_int(value["settling_blend_frames"]),
            settling_standing_pose_blend=_strict_float(value["settling_standing_pose_blend"]),
            settling_waist_pitch_bias_rad=_strict_float(value["settling_waist_pitch_bias_rad"]),
            target_smoothing_alpha=_strict_float(value["target_smoothing_alpha"]),
        )


@dataclass(frozen=True)
class G1ContextualRecoveryArtifact:
    """Content-addressed nearest-prototype contextual recovery policy."""

    body_hash: str
    motion_hash: str
    baseline_recovery_config_hash: str
    fallback_recovery_config_hash: str
    training_dataset_hash: str
    observation_mean: tuple[float, ...]
    observation_scale: tuple[float, ...]
    regime_feature_names: tuple[str, ...]
    regime_prototypes: tuple[tuple[float, ...], ...]
    primitives: tuple[G1ContextualRecoveryPrimitive, ...]
    maximum_regime_distance: float
    maximum_feature_z: float
    training_episode_count: int
    training_seed: int
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_artifact.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("motion_hash", self.motion_hash),
            ("baseline_recovery_config_hash", self.baseline_recovery_config_hash),
            ("fallback_recovery_config_hash", self.fallback_recovery_config_hash),
            ("training_dataset_hash", self.training_dataset_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        if self.schema_version != "rosclaw.g1_goalforge.contextual_recovery_artifact.v1":
            raise ValueError("unsupported contextual recovery artifact schema")
        if self.regime_feature_names != G1_CONTEXTUAL_RECOVERY_FEATURES:
            raise ValueError("contextual recovery feature contract mismatch")
        observation_count = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
        feature_count = len(self.regime_feature_names)
        if (
            len(self.observation_mean) != observation_count
            or len(self.observation_scale) != observation_count
        ):
            raise ValueError("contextual recovery normalization shape is invalid")
        if not 1 <= len(self.regime_prototypes) <= 16:
            raise ValueError("contextual recovery requires 1 to 16 prototypes")
        if len(self.primitives) != len(self.regime_prototypes) or any(
            len(row) != feature_count for row in self.regime_prototypes
        ):
            raise ValueError("contextual recovery prototype/primitive shape is invalid")
        arrays = (
            np.asarray(self.observation_mean, dtype=np.float64),
            np.asarray(self.observation_scale, dtype=np.float64),
            np.asarray(self.regime_prototypes, dtype=np.float64),
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("contextual recovery artifact contains non-finite values")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("contextual recovery observation scales must be positive")
        thresholds = (self.maximum_regime_distance, self.maximum_feature_z)
        if any(isinstance(value, bool) for value in thresholds) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in thresholds
        ):
            raise ValueError("contextual recovery thresholds must be finite numbers")
        if not 0.01 <= self.maximum_regime_distance <= 2.0:
            raise ValueError("contextual recovery distance must be in [0.01, 2]")
        if not 2.0 <= self.maximum_feature_z <= 12.0:
            raise ValueError("contextual recovery feature envelope must be in [2, 12]")
        if isinstance(self.training_episode_count, bool) or not isinstance(
            self.training_episode_count, int
        ):
            raise ValueError("contextual recovery episode count must be an integer")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise ValueError("contextual recovery training seed must be an integer")
        if self.training_episode_count <= 0 or self.training_seed < 0:
            raise ValueError("contextual recovery training evidence is invalid")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("contextual recovery must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation_names"] = list(G1_MUSCLE_MEMORY_OBSERVATIONS)
        value["observation_mean"] = list(self.observation_mean)
        value["observation_scale"] = list(self.observation_scale)
        value["regime_feature_names"] = list(self.regime_feature_names)
        value["regime_prototypes"] = [list(row) for row in self.regime_prototypes]
        value["primitives"] = [item.to_dict() for item in self.primitives]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G1ContextualRecoveryArtifact:
        expected = {
            "schema_version",
            "body_hash",
            "motion_hash",
            "baseline_recovery_config_hash",
            "fallback_recovery_config_hash",
            "training_dataset_hash",
            "observation_names",
            "observation_mean",
            "observation_scale",
            "regime_feature_names",
            "regime_prototypes",
            "primitives",
            "maximum_regime_distance",
            "maximum_feature_z",
            "training_episode_count",
            "training_seed",
            "activation_ceiling",
        }
        if set(value) != expected:
            raise ValueError("contextual recovery artifact fields are invalid")
        observation_names = value["observation_names"]
        feature_names = value["regime_feature_names"]
        if (
            not isinstance(observation_names, list)
            or not all(isinstance(item, str) for item in observation_names)
            or tuple(observation_names) != G1_MUSCLE_MEMORY_OBSERVATIONS
        ):
            raise ValueError("contextual recovery observation contract mismatch")
        if not isinstance(feature_names, list) or not all(
            isinstance(item, str) for item in feature_names
        ):
            raise ValueError("contextual recovery feature names must be an array of strings")
        prototypes_value = value["regime_prototypes"]
        primitives_value = value["primitives"]
        if not isinstance(prototypes_value, list) or not isinstance(primitives_value, list):
            raise ValueError("contextual recovery prototypes/primitives must be arrays")
        if not all(isinstance(row, list) for row in prototypes_value):
            raise ValueError("contextual recovery prototypes must contain numeric arrays")
        if not all(isinstance(item, Mapping) for item in primitives_value):
            raise ValueError("contextual recovery primitives must contain objects")
        return cls(
            schema_version=str(value["schema_version"]),
            body_hash=str(value["body_hash"]),
            motion_hash=str(value["motion_hash"]),
            baseline_recovery_config_hash=str(value["baseline_recovery_config_hash"]),
            fallback_recovery_config_hash=str(value["fallback_recovery_config_hash"]),
            training_dataset_hash=str(value["training_dataset_hash"]),
            observation_mean=_float_tuple(value["observation_mean"]),
            observation_scale=_float_tuple(value["observation_scale"]),
            regime_feature_names=tuple(feature_names),
            regime_prototypes=tuple(_float_tuple(row) for row in prototypes_value),
            primitives=tuple(
                G1ContextualRecoveryPrimitive.from_dict(item) for item in primitives_value
            ),
            maximum_regime_distance=_strict_float(value["maximum_regime_distance"]),
            maximum_feature_z=_strict_float(value["maximum_feature_z"]),
            training_episode_count=_strict_int(value["training_episode_count"]),
            training_seed=_strict_int(value["training_seed"]),
            activation_ceiling=str(value["activation_ceiling"]),
        )


@dataclass(frozen=True)
class G1ContextualRecoverySelection:
    primitive_index: int | None
    nearest_distance: float
    out_of_distribution: bool


@dataclass(frozen=True)
class G1ContextualRecoveryReceipt:
    artifact_hash: str
    selected_primitive_index: int | None
    nearest_distance: float | None
    selection_count: int
    fallback_count: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class G1ContextualRecoveryPolicy:
    """Select a bounded primitive from normalized proprioceptive context."""

    def __init__(self, artifact: G1ContextualRecoveryArtifact) -> None:
        self.artifact = artifact
        self._mean = np.asarray(artifact.observation_mean, dtype=np.float64)
        self._scale = np.asarray(artifact.observation_scale, dtype=np.float64)
        self._feature_indices = np.asarray(
            [G1_MUSCLE_MEMORY_OBSERVATIONS.index(name) for name in artifact.regime_feature_names],
            dtype=np.int64,
        )
        self._prototypes = np.asarray(artifact.regime_prototypes, dtype=np.float64)
        self.reset()

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        baseline_recovery_config_hash: str,
        fallback_recovery_config_hash: str,
    ) -> None:
        expected = self.artifact
        if body_hash != expected.body_hash:
            raise ValueError("contextual recovery Body hash mismatch")
        if motion_hash != expected.motion_hash:
            raise ValueError("contextual recovery motion hash mismatch")
        if baseline_recovery_config_hash != expected.baseline_recovery_config_hash:
            raise ValueError("contextual recovery baseline config hash mismatch")
        if fallback_recovery_config_hash != expected.fallback_recovery_config_hash:
            raise ValueError("contextual recovery fallback config hash mismatch")

    def reset(self) -> None:
        self._selection: G1ContextualRecoverySelection | None = None
        self._selection_count = 0
        self._fallback_count = 0

    def select(self, observation: Mapping[str, float]) -> G1ContextualRecoverySelection:
        if self._selection is not None:
            return self._selection
        missing = set(G1_MUSCLE_MEMORY_OBSERVATIONS).difference(observation)
        if missing:
            return self._latch_fallback()
        try:
            ordered = np.asarray(
                [float(observation[name]) for name in G1_MUSCLE_MEMORY_OBSERVATIONS],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return self._latch_fallback()
        if not np.all(np.isfinite(ordered)):
            return self._latch_fallback()
        normalized = (ordered - self._mean) / self._scale
        feature = normalized[self._feature_indices]
        distances = np.linalg.norm(self._prototypes - feature, axis=1)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        ood = bool(
            np.max(np.abs(normalized)) > self.artifact.maximum_feature_z
            or distance > self.artifact.maximum_regime_distance
        )
        self._selection = G1ContextualRecoverySelection(
            primitive_index=None if ood else nearest,
            nearest_distance=distance,
            out_of_distribution=ood,
        )
        self._selection_count += 1
        self._fallback_count += int(ood)
        return self._selection

    def _latch_fallback(self) -> G1ContextualRecoverySelection:
        self._selection = G1ContextualRecoverySelection(
            primitive_index=None,
            nearest_distance=self.artifact.maximum_regime_distance + 1.0,
            out_of_distribution=True,
        )
        self._selection_count += 1
        self._fallback_count += 1
        return self._selection

    def build_receipt(self) -> G1ContextualRecoveryReceipt:
        return G1ContextualRecoveryReceipt(
            artifact_hash=self.artifact.artifact_hash,
            selected_primitive_index=(
                self._selection.primitive_index if self._selection is not None else None
            ),
            nearest_distance=(
                self._selection.nearest_distance if self._selection is not None else None
            ),
            selection_count=self._selection_count,
            fallback_count=self._fallback_count,
        )


def load_g1_contextual_recovery_artifact(path: Path) -> G1ContextualRecoveryArtifact:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("contextual recovery artifact is missing or exceeds 1 MiB")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("contextual recovery artifact is not readable JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("contextual recovery artifact root must be an object")
    return G1ContextualRecoveryArtifact.from_dict(value)


def _strict_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("contextual recovery numeric field has invalid type")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("contextual recovery numeric field must be finite")
    return result


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("contextual recovery integer field has invalid type")
    return value


def _float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("contextual recovery numeric vector must be an array")
    return tuple(_strict_float(item) for item in value)


__all__ = [
    "G1ContextualRecoveryArtifact",
    "G1_CONTEXTUAL_RECOVERY_FEATURES",
    "G1ContextualRecoveryPolicy",
    "G1ContextualRecoveryPrimitive",
    "G1ContextualRecoveryReceipt",
    "G1ContextualRecoverySelection",
    "load_g1_contextual_recovery_artifact",
]
