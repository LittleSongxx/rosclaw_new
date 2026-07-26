"""Body/regime-bound storage for trial-to-trial feedback trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ILCFeedforward:
    """Immutable, body/regime-bound joint feed-forward trajectory."""

    body_hash: str
    regime_hash: str
    joint_names: tuple[str, ...]
    values: np.ndarray
    residual_limit: float
    trial: int
    source_receipt_hashes: tuple[str, ...] = ()
    schema_version: str = "rosclaw.feedback.ilc_feedforward.v1"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        if not _SHA256.fullmatch(self.body_hash) or not _SHA256.fullmatch(self.regime_hash):
            raise ValueError("ILC feedforward requires body and regime sha256 hashes")
        if any(not _SHA256.fullmatch(value) for value in self.source_receipt_hashes):
            raise ValueError("ILC source receipt hashes must be sha256 content hashes")
        if values.ndim != 2 or not values.size or values.shape[1] != len(self.joint_names):
            raise ValueError("ILC feedforward must be non-empty [time, joint] data")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("ILC feedforward joint names must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("ILC feedforward must be finite")
        if self.residual_limit <= 0.0:
            raise ValueError("ILC feedforward residual_limit must be positive")
        if np.max(np.abs(values)) > self.residual_limit + 1e-12:
            raise ValueError("ILC feedforward exceeds its residual limit")
        if self.trial < 0:
            raise ValueError("ILC feedforward trial must be non-negative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def trajectory_hash(self) -> str:
        metadata = {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "regime_hash": self.regime_hash,
            "joint_names": list(self.joint_names),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "residual_limit": self.residual_limit,
            "trial": self.trial,
            "source_receipt_hashes": list(self.source_receipt_hashes),
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(self.values.tobytes(order="C"))
        return "sha256:" + digest.hexdigest()

    def require_compatible(
        self,
        *,
        body_hash: str,
        regime_hash: str,
        joint_names: tuple[str, ...],
    ) -> None:
        if self.body_hash != body_hash:
            raise ValueError("wrong-body ILC feedforward rejected")
        if self.regime_hash != regime_hash:
            raise ValueError("wrong-regime ILC feedforward rejected")
        if self.joint_names != joint_names:
            raise ValueError("ILC feedforward joint order mismatch")

    def value_at(self, index: int) -> np.ndarray:
        if not 0 <= index < self.values.shape[0]:
            raise IndexError("ILC feedforward frame is outside its pinned trajectory")
        return self.values[index].copy()

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trajectory_hash": self.trajectory_hash,
            "body_hash": self.body_hash,
            "regime_hash": self.regime_hash,
            "joint_names": list(self.joint_names),
            "shape": list(self.values.shape),
            "residual_limit": self.residual_limit,
            "residual_peak": float(np.max(np.abs(self.values))),
            "trial": self.trial,
            "source_receipt_hashes": list(self.source_receipt_hashes),
            "value_hash": canonical_hash(
                {
                    "sha256": hashlib.sha256(self.values.tobytes(order="C")).hexdigest(),
                }
            ),
        }


@dataclass(frozen=True)
class ILCTrajectory:
    receipt_hash: str
    body_hash: str
    regime_hash: str
    tracking_error: np.ndarray
    feedforward_residual: np.ndarray
    energy: float
    safety_interventions: int

    def __post_init__(self) -> None:
        error = np.asarray(self.tracking_error, dtype=np.float64).copy()
        residual = np.asarray(self.feedforward_residual, dtype=np.float64).copy()
        if error.ndim != 2 or error.shape != residual.shape or not error.size:
            raise ValueError(
                "ILC error and residual must have equal non-empty [time, signal] shape"
            )
        if not np.all(np.isfinite(error)) or not np.all(np.isfinite(residual)):
            raise ValueError("ILC trajectories must be finite")
        if self.energy < 0.0 or self.safety_interventions < 0:
            raise ValueError("energy and safety_interventions must be non-negative")
        error.setflags(write=False)
        residual.setflags(write=False)
        object.__setattr__(self, "tracking_error", error)
        object.__setattr__(self, "feedforward_residual", residual)

    @property
    def error_rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.tracking_error))))


class ILCTrajectoryMemory:
    """Reject wrong-body/wrong-regime reuse and keep a bounded trial window."""

    def __init__(self, *, body_hash: str, regime_hash: str, capacity: int = 20) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.body_hash = body_hash
        self.regime_hash = regime_hash
        self.capacity = capacity
        self._items: list[ILCTrajectory] = []
        self._shape: tuple[int, int] | None = None

    @property
    def items(self) -> tuple[ILCTrajectory, ...]:
        return tuple(self._items)

    def append(self, trajectory: ILCTrajectory) -> None:
        if trajectory.body_hash != self.body_hash:
            raise ValueError("wrong-body ILC trajectory rejected")
        if trajectory.regime_hash != self.regime_hash:
            raise ValueError("wrong-regime ILC trajectory rejected")
        shape = trajectory.tracking_error.shape
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError("ILC trajectory shape changed")
        self._items.append(trajectory)
        if len(self._items) > self.capacity:
            del self._items[0]
