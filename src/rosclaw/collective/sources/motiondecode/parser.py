"""Strict MotionDecode CSV parser and canonical motion episode."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.joint_mapping import (
    MOTIONDECODE_ROOT_COLUMNS,
    MotionDecodeJointMapping,
    mapping_from_header,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_MAX_FRAMES = 1_000_000
_MAX_CSV_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class CanonicalMotionEpisode:
    """One immutable, target-ordered motion reference.

    Arrays are read-only and are not serialized into control-plane receipts.
    The registered file hash is the commitment to their raw values.
    """

    source_manifest_hash: str
    source_file_hash: str
    target_body_hash: str
    mapping: MotionDecodeJointMapping
    sample_rate_hz: float
    time: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    implicit_timeline: bool
    derivation_manifest_hash: str | None = None
    ball_pose_available: bool = False
    action_semantics_available: bool = False
    reward_semantics_available: bool = False
    transition_semantics_available: bool = False
    schema_version: str = "rosclaw.collective.canonical_motion_episode.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("source_manifest_hash", self.source_manifest_hash),
            ("source_file_hash", self.source_file_hash),
            ("target_body_hash", self.target_body_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not isinstance(self.mapping, MotionDecodeJointMapping):
            raise ValueError("mapping must be MotionDecodeJointMapping")
        if self.derivation_manifest_hash is not None and (
            not self.derivation_manifest_hash.startswith("sha256:")
            or len(self.derivation_manifest_hash) != 71
        ):
            raise ValueError("derivation_manifest_hash must be a sha256: content hash")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        frames = int(self.time.shape[0])
        expected = {
            "time": (frames,),
            "root_position": (frames, 3),
            "root_quaternion": (frames, 4),
            "joint_position": (frames, len(self.mapping.target_joint_names)),
            "joint_velocity": (frames, len(self.mapping.target_joint_names)),
            "joint_acceleration": (frames, len(self.mapping.target_joint_names)),
        }
        if frames < 3:
            raise ValueError("a canonical motion episode requires at least three frames")
        for label, shape in expected.items():
            array = np.asarray(getattr(self, label), dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{label} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, label, array)
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time must be strictly increasing")

    @property
    def episode_hash(self) -> str:
        return canonical_hash(self.summary())

    @property
    def duration_seconds(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def hardware_authorized(self) -> bool:
        return False

    def summary(self) -> dict[str, Any]:
        summary = {
            "schema_version": self.schema_version,
            "source_manifest_hash": self.source_manifest_hash,
            "source_file_hash": self.source_file_hash,
            "target_body_hash": self.target_body_hash,
            "mapping_hash": self.mapping.mapping_hash,
            "joint_names": list(self.mapping.target_joint_names),
            "frame_count": int(self.time.shape[0]),
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "implicit_timeline": self.implicit_timeline,
            "root_position_unit": "m",
            "root_quaternion_order": "wxyz",
            "joint_position_unit": "rad",
            "ball_pose_available": self.ball_pose_available,
            "action_semantics_available": self.action_semantics_available,
            "reward_semantics_available": self.reward_semantics_available,
            "transition_semantics_available": self.transition_semantics_available,
            "hardware_authorized": self.hardware_authorized,
        }
        if self.derivation_manifest_hash is not None:
            summary["derivation_manifest_hash"] = self.derivation_manifest_hash
        return summary


def parse_motion_csv(
    path: Path,
    *,
    source_manifest_hash: str,
    expected_file_hash: str,
    target_body_hash: str,
    sample_rate_hz: float = 120.0,
    target_joint_names: tuple[str, ...] = G1_DDS_JOINT_NAMES,
    max_frames: int = _MAX_FRAMES,
) -> CanonicalMotionEpisode:
    """Parse a full registered CSV without guessing units or joint names."""

    if max_frames <= 0 or max_frames > _MAX_FRAMES:
        raise ValueError(f"max_frames must be in [1, {_MAX_FRAMES}]")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("motion CSV must be a non-symlink regular file")
    if resolved.stat().st_size > _MAX_CSV_BYTES:
        raise ValueError("motion CSV exceeds the 128 MB parser safety limit")
    payload = _read_stable_bytes(resolved)
    actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_file_hash:
        raise ValueError("motion CSV does not match its registered content hash")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("motion CSV must be UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("motion CSV is empty") from exc
        mapping, time_column = mapping_from_header(
            header,
            target_joint_names=target_joint_names,
        )
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(rows) >= max_frames:
                raise ValueError("motion CSV exceeds the bounded frame limit")
            if len(row) != len(header):
                raise ValueError(f"motion CSV row {line_number} has the wrong width")
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                raise ValueError(f"motion CSV row {line_number} contains non-numeric data") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"motion CSV row {line_number} contains NaN or Inf")
            rows.append(values)
    if len(rows) < 3:
        raise ValueError("motion CSV requires at least three data rows")
    matrix = np.asarray(rows, dtype=np.float64)
    offset = 1 if time_column is not None else 0
    if time_column is None:
        time = np.arange(matrix.shape[0], dtype=np.float64) / sample_rate_hz
        implicit_timeline = True
    else:
        time = matrix[:, 0].copy()
        implicit_timeline = False
        observed_steps = np.diff(time)
        inferred_rate = 1.0 / float(np.median(observed_steps))
        if not math.isclose(inferred_rate, sample_rate_hz, rel_tol=0.02):
            raise ValueError("explicit MotionDecode timeline is not consistent with declared rate")
    root_start = offset
    joint_start = root_start + len(MOTIONDECODE_ROOT_COLUMNS)
    root_position = matrix[:, root_start : root_start + 3].copy()
    root_quaternion = matrix[:, root_start + 3 : joint_start].copy()
    source_joints = matrix[:, joint_start:]
    joint_position = source_joints[:, mapping.source_indices_by_target].copy()
    joint_velocity = np.gradient(joint_position, time, axis=0, edge_order=2)
    joint_acceleration = np.gradient(joint_velocity, time, axis=0, edge_order=2)
    return CanonicalMotionEpisode(
        source_manifest_hash=source_manifest_hash,
        source_file_hash=expected_file_hash,
        target_body_hash=target_body_hash,
        mapping=mapping,
        sample_rate_hz=sample_rate_hz,
        time=time,
        root_position=root_position,
        root_quaternion=root_quaternion,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_acceleration=joint_acceleration,
        implicit_timeline=implicit_timeline,
    )


def _read_stable_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read(_MAX_CSV_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("motion CSV changed while it was read")
    if len(payload) > _MAX_CSV_BYTES:
        raise ValueError("motion CSV exceeds the 128 MB parser safety limit")
    return payload


__all__ = ["CanonicalMotionEpisode", "parse_motion_csv"]
