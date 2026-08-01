"""Strict CPU MuJoCo qualification for canonical motion references."""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.contracts import LicenseDecision
from rosclaw.collective.sources.motiondecode.audit import (
    AuditSeverity,
    MotionDecodeAuditThresholds,
    MotionQualificationLevel,
    load_g1_joint_contract,
)
from rosclaw.collective.sources.motiondecode.contact import (
    CanonicalContactTrace,
    ContactInferenceThresholds,
    MotionDecodeContactAudit,
    infer_motiondecode_contact_batch,
)
from rosclaw.collective.sources.motiondecode.manifest import MotionDecodeRegistration
from rosclaw.collective.sources.motiondecode.parser import CanonicalMotionEpisode
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
)

_MAX_SCENE_BYTES = 20 * 1024 * 1024
_HARD_FAILURES = frozenset(
    {
        "NONFINITE_PHYSICS",
        "INCOMPLETE_PHYSICS_REPLAY",
        "PELVIS_HEIGHT_FALL",
        "NON_FOOT_FLOOR_CONTACT",
        "TORQUE_LIMIT_EXCEEDED",
        "MUJOCO_WARNING",
    }
)


class PhysicsClipStatus(StrEnum):
    BLOCKED_LICENSE = "blocked_license"
    CONTACT_NOT_ELIGIBLE = "contact_not_eligible"
    PHYSICS_EXECUTED = "physics_executed"


@dataclass(frozen=True)
class PhysicsQualificationThresholds:
    minimum_duration_s: float = 1.0
    minimum_pelvis_height_m: float = 0.45
    maximum_joint_tracking_rmse_rad: float = 0.15
    maximum_root_position_rmse_m: float = 0.25
    maximum_root_orientation_rmse_rad: float = 0.35
    maximum_torque_ratio: float = 1.001
    maximum_torque_saturation_ratio: float = 0.10
    maximum_support_slip_p95_m_s: float = 0.25
    maximum_nonfoot_contact_steps: int = 0
    maximum_frames: int = 10_000
    maximum_simulation_seconds: float = 30.0

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError("physics qualification thresholds must be finite")
        if self.minimum_duration_s <= 0.0 or self.minimum_pelvis_height_m <= 0.0:
            raise ValueError("physics duration and pelvis height must be positive")
        if not isinstance(self.maximum_nonfoot_contact_steps, int) or not isinstance(
            self.maximum_frames, int
        ):
            raise ValueError("physics count thresholds must be integers")
        if self.maximum_frames < 3:
            raise ValueError("maximum_frames must retain at least three frames")

    @property
    def threshold_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_physics_thresholds.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, float | int]:
        return dict(vars(self))


@dataclass(frozen=True)
class PhysicsQualificationIssue:
    code: str
    observed: float
    threshold: float

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("physics qualification issue code must not be empty")
        if not math.isfinite(self.observed) or not math.isfinite(self.threshold):
            raise ValueError("physics qualification issue values must be finite")

    @property
    def hard_failure(self) -> bool:
        return self.code in _HARD_FAILURES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": AuditSeverity.ERROR.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "hard_failure": self.hard_failure,
        }


@dataclass(frozen=True)
class MotionPhysicsQualification:
    episode_hash: str
    contact_trace_hash: str
    target_body_hash: str
    target_model_file_hash: str
    scene_file_hash: str
    compiled_scene_hash: str
    threshold_hash: str
    qualification: MotionQualificationLevel
    metrics: dict[str, float | int | bool]
    issues: tuple[PhysicsQualificationIssue, ...]
    root_alignment_m: tuple[float, float, float]
    controller: str = "mujoco_model_position_reference_v1"
    backend: str = "mujoco_cpu"
    schema_version: str = "rosclaw.collective.motiondecode_physics_qualification.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("episode_hash", self.episode_hash),
            ("contact_trace_hash", self.contact_trace_hash),
            ("target_body_hash", self.target_body_hash),
            ("target_model_file_hash", self.target_model_file_hash),
            ("scene_file_hash", self.scene_file_hash),
            ("compiled_scene_hash", self.compiled_scene_hash),
            ("threshold_hash", self.threshold_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if self.backend != "mujoco_cpu" or self.controller != (
            "mujoco_model_position_reference_v1"
        ):
            raise ValueError("physics qualification backend/controller is unsupported")
        if self.qualification not in {
            MotionQualificationLevel.Q1_KINEMATIC_ONLY,
            MotionQualificationLevel.Q2_TRACKABLE_WITH_REPAIR,
            MotionQualificationLevel.Q3_PHYSICS_TRACKABLE,
        }:
            raise ValueError("physics qualification level is unsupported")
        if len(self.root_alignment_m) != 3 or not all(
            math.isfinite(value) for value in self.root_alignment_m
        ):
            raise ValueError("physics root alignment must be finite")
        if self.qualification is MotionQualificationLevel.Q3_PHYSICS_TRACKABLE and self.issues:
            raise ValueError("Q3 physics qualification cannot contain gate failures")
        if any(issue.hard_failure for issue in self.issues) and (
            self.qualification is not MotionQualificationLevel.Q1_KINEMATIC_ONLY
        ):
            raise ValueError("hard physics failures must remain Q1")

    @property
    def qualification_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def physics_trackable(self) -> bool:
        return self.qualification in {
            MotionQualificationLevel.Q3_PHYSICS_TRACKABLE,
            MotionQualificationLevel.Q4_ROBUST_TRACKABLE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_hash": self.episode_hash,
            "contact_trace_hash": self.contact_trace_hash,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "scene_file_hash": self.scene_file_hash,
            "compiled_scene_hash": self.compiled_scene_hash,
            "threshold_hash": self.threshold_hash,
            "backend": self.backend,
            "controller": self.controller,
            "qualification": self.qualification.value,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
            "root_alignment_m": list(self.root_alignment_m),
            "truth_level": "T1_CPU_MUJOCO",
            "physics_trackable": self.physics_trackable,
            "promotion_truth_eligible": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodePhysicsClip:
    relative_path: str
    source_file_hash: str
    contact_audit_hash: str
    status: PhysicsClipStatus
    qualification: MotionQualificationLevel
    blocker_codes: tuple[str, ...]
    result: MotionPhysicsQualification | None = None
    schema_version: str = "rosclaw.collective.motiondecode_physics_clip.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("source_file_hash", self.source_file_hash),
            ("contact_audit_hash", self.contact_audit_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not self.relative_path or not isinstance(self.status, PhysicsClipStatus):
            raise ValueError("physics clip identity/status is invalid")
        if not isinstance(self.qualification, MotionQualificationLevel):
            raise ValueError("physics clip qualification is unknown")
        if self.status is PhysicsClipStatus.PHYSICS_EXECUTED:
            if self.result is None or self.result.qualification is not self.qualification:
                raise ValueError("executed physics clip requires a matching result")
        elif self.result is not None:
            raise ValueError("non-executed physics clip cannot claim a result")

    @property
    def clip_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "source_file_hash": self.source_file_hash,
            "contact_audit_hash": self.contact_audit_hash,
            "status": self.status.value,
            "qualification": self.qualification.value,
            "blocker_codes": list(self.blocker_codes),
            "result": self.result.to_dict() if self.result is not None else None,
            "result_hash": (self.result.qualification_hash if self.result is not None else None),
            "physics_step_count": (
                int(self.result.metrics["physics_step_count"]) if self.result is not None else 0
            ),
            "training_eligible": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodeQualificationReport:
    registration_hash: str
    source_manifest_hash: str
    contact_report_hash: str
    target_body_hash: str
    target_model_file_hash: str
    scene_file_hash: str
    compiled_scene_hash: str
    license_decision: LicenseDecision
    thresholds: PhysicsQualificationThresholds
    clips: tuple[MotionDecodePhysicsClip, ...]
    schema_version: str = "rosclaw.collective.motiondecode_qualification_report.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("registration_hash", self.registration_hash),
            ("source_manifest_hash", self.source_manifest_hash),
            ("contact_report_hash", self.contact_report_hash),
            ("target_body_hash", self.target_body_hash),
            ("target_model_file_hash", self.target_model_file_hash),
            ("scene_file_hash", self.scene_file_hash),
            ("compiled_scene_hash", self.compiled_scene_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not isinstance(self.license_decision, LicenseDecision):
            raise ValueError("qualification report license decision is unknown")
        if not isinstance(self.thresholds, PhysicsQualificationThresholds):
            raise ValueError("qualification report thresholds are invalid")
        clips = tuple(self.clips)
        if any(not isinstance(clip, MotionDecodePhysicsClip) for clip in clips):
            raise ValueError("qualification report contains an invalid clip")
        paths = tuple(clip.relative_path for clip in clips)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("qualification report clips must be unique and sorted")
        for clip in clips:
            if clip.result is not None and (
                clip.result.target_body_hash != self.target_body_hash
                or clip.result.target_model_file_hash != self.target_model_file_hash
                or clip.result.scene_file_hash != self.scene_file_hash
                or clip.result.compiled_scene_hash != self.compiled_scene_hash
                or clip.result.threshold_hash != self.thresholds.threshold_hash
            ):
                raise ValueError("physics result lineage does not match its report")
        object.__setattr__(self, "clips", clips)

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def qualification_counts(self) -> dict[str, int]:
        counts = Counter(clip.qualification.value for clip in self.clips)
        return dict(sorted(counts.items()))

    @property
    def status_counts(self) -> dict[str, int]:
        counts = Counter(clip.status.value for clip in self.clips)
        return dict(sorted(counts.items()))

    @property
    def physics_executed_count(self) -> int:
        return sum(clip.status is PhysicsClipStatus.PHYSICS_EXECUTED for clip in self.clips)

    @property
    def q3_count(self) -> int:
        return sum(
            clip.qualification is MotionQualificationLevel.Q3_PHYSICS_TRACKABLE
            for clip in self.clips
        )

    @property
    def physics_step_count(self) -> int:
        return sum(
            int(clip.result.metrics["physics_step_count"])
            for clip in self.clips
            if clip.result is not None
        )

    @property
    def quality_commitment(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_physics_quality.v1",
                "contact_report_hash": self.contact_report_hash,
                "scene_file_hash": self.scene_file_hash,
                "compiled_scene_hash": self.compiled_scene_hash,
                "threshold_hash": self.thresholds.threshold_hash,
                "clip_hashes": [clip.clip_hash for clip in self.clips],
            }
        )

    @property
    def training_blockers(self) -> list[str]:
        blockers = [
            "SOURCE_COORDINATE_FRAME_UNSPECIFIED",
            "SYNCHRONIZED_BALL_POSE_ABSENT",
            "ACTION_REWARD_TRANSITION_SEMANTICS_ABSENT",
            "FRAME_LEVEL_CONTACT_TRACE_NOT_PERSISTED",
        ]
        if self.q3_count == 0:
            blockers.insert(0, "NO_Q3_PHYSICS_TRACKABLE_CLIPS")
        if self.physics_executed_count != len(self.clips):
            blockers.insert(0, "PHYSICS_QUALIFICATION_PARTIAL")
        if self.license_decision is not LicenseDecision.PERMITTED:
            blockers.insert(0, "LICENSE_NOT_PERMITTED")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_hash": self.registration_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "contact_report_hash": self.contact_report_hash,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "scene_file_hash": self.scene_file_hash,
            "compiled_scene_hash": self.compiled_scene_hash,
            "license_decision": self.license_decision.value,
            "thresholds": self.thresholds.to_dict(),
            "threshold_hash": self.thresholds.threshold_hash,
            "clips": [clip.to_dict() for clip in self.clips],
            "clip_count": len(self.clips),
            "status_counts": self.status_counts,
            "qualification_counts": self.qualification_counts,
            "physics_executed_count": self.physics_executed_count,
            "physics_step_count": self.physics_step_count,
            "q3_count": self.q3_count,
            "quality_commitment": self.quality_commitment,
            "backend": "mujoco_cpu",
            "controller": "mujoco_model_position_reference_v1",
            "physics_execution_authorized": (self.license_decision is LicenseDecision.PERMITTED),
            "coordinate_frame_verified": False,
            "training_eligible": False,
            "training_blockers": self.training_blockers,
            "promotion_truth_eligible": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


def qualify_motiondecode_snapshot(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    scene_path: Path,
    expected_ingest_report_hash: str,
    expected_repair_report_hash: str,
    expected_contact_report_hash: str,
    audit_thresholds: MotionDecodeAuditThresholds | None = None,
    contact_thresholds: ContactInferenceThresholds | None = None,
    physics_thresholds: PhysicsQualificationThresholds | None = None,
) -> MotionDecodeQualificationReport:
    """Replay contact evidence and run physics only when the source is permitted."""

    selected_physics = physics_thresholds or PhysicsQualificationThresholds()
    batch = infer_motiondecode_contact_batch(
        registration,
        dataset_root,
        target_model_path=target_model_path,
        expected_ingest_report_hash=expected_ingest_report_hash,
        expected_repair_report_hash=expected_repair_report_hash,
        audit_thresholds=audit_thresholds,
        contact_thresholds=contact_thresholds,
    )
    if batch.report.report_hash != expected_contact_report_hash:
        raise ValueError("contact report hash does not match replayed contact evidence")
    joint_limits, _, target_model_hash = load_g1_joint_contract(target_model_path)
    *_, scene_hash, compiled_scene_hash = _load_physics_scene(
        scene_path,
        expected_joint_limits=joint_limits,
        expected_target_model_hash=target_model_hash,
    )
    contact_by_path = {clip.relative_path: clip for clip in batch.report.clips}
    bundle_by_path = {bundle.relative_path: bundle for bundle in batch.bundles}
    clips: list[MotionDecodePhysicsClip] = []
    for contact in batch.report.clips:
        if not contact.phase_segmentation_candidate:
            clips.append(_contact_rejected_physics(contact))
            continue
        if registration.manifest.license_snapshot.decision is not LicenseDecision.PERMITTED:
            clips.append(_license_blocked_physics(contact))
            continue
        bundle = bundle_by_path[contact.relative_path]
        result = qualify_canonical_motion(
            bundle.episode,
            bundle.trace,
            target_model_path=target_model_path,
            scene_path=scene_path,
            license_decision=registration.manifest.license_snapshot.decision,
            thresholds=selected_physics,
        )
        clips.append(
            MotionDecodePhysicsClip(
                relative_path=contact.relative_path,
                source_file_hash=contact.source_file_hash,
                contact_audit_hash=contact.audit_hash,
                status=PhysicsClipStatus.PHYSICS_EXECUTED,
                qualification=result.qualification,
                blocker_codes=tuple(issue.code for issue in result.issues),
                result=result,
            )
        )
    if set(contact_by_path) != {clip.relative_path for clip in clips}:
        raise ValueError("physics qualification did not cover every contact audit")
    return MotionDecodeQualificationReport(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        contact_report_hash=batch.report.report_hash,
        target_body_hash=batch.report.target_body_hash,
        target_model_file_hash=batch.report.target_model_file_hash,
        scene_file_hash=scene_hash,
        compiled_scene_hash=compiled_scene_hash,
        license_decision=registration.manifest.license_snapshot.decision,
        thresholds=selected_physics,
        clips=tuple(clips),
    )


def qualify_canonical_motion(
    episode: CanonicalMotionEpisode,
    contact: CanonicalContactTrace,
    *,
    target_model_path: Path,
    scene_path: Path,
    license_decision: LicenseDecision,
    thresholds: PhysicsQualificationThresholds | None = None,
) -> MotionPhysicsQualification:
    """Advance real CPU MuJoCo physics with the scene's position actuators.

    The license gate is structural: no physics step runs for a source whose
    license is not PERMITTED, regardless of which entry point is used.
    """

    if license_decision is not LicenseDecision.PERMITTED:
        raise ValueError("physics qualification requires a PERMITTED license decision")
    selected = thresholds or PhysicsQualificationThresholds()
    if contact.episode_hash != episode.episode_hash:
        raise ValueError("contact trace does not match the motion episode")
    frames = int(episode.time.shape[0])
    if frames > selected.maximum_frames:
        raise ValueError("motion exceeds the bounded physics frame budget")
    if episode.duration_seconds > selected.maximum_simulation_seconds:
        raise ValueError("motion exceeds the bounded physics duration")
    joint_limits, target_body_hash, target_model_hash = load_g1_joint_contract(target_model_path)
    if episode.target_body_hash != target_body_hash:
        raise ValueError("motion episode target body does not match physics model")
    if contact.target_model_file_hash != target_model_hash:
        raise ValueError("contact trace target model does not match physics model")
    (
        model,
        data,
        qpos_addresses,
        dof_addresses,
        left_site,
        right_site,
        floor_geom,
        allowed_floor_bodies,
        scene_hash,
        compiled_scene_hash,
    ) = _load_physics_scene(
        scene_path,
        expected_joint_limits=joint_limits,
        expected_target_model_hash=target_model_hash,
    )
    aligned_root = episode.root_position.copy()
    alignment = (
        -float(aligned_root[0, 0]),
        -float(aligned_root[0, 1]),
        -float(contact.ground_height_m),
    )
    aligned_root[:, 0] += alignment[0]
    aligned_root[:, 1] += alignment[1]
    aligned_root[:, 2] += alignment[2]
    data.qpos[:3] = aligned_root[0]
    data.qpos[3:7] = _unit_quaternion(episode.root_quaternion[0])
    data.qpos[qpos_addresses] = episode.joint_position[0]
    data.qvel[:] = 0.0
    import mujoco

    mujoco.mj_forward(model, data)
    joint_errors: list[np.ndarray] = []
    root_errors: list[np.ndarray] = []
    orientation_errors: list[float] = []
    left_positions: list[np.ndarray] = []
    right_positions: list[np.ndarray] = []
    minimum_pelvis_height = float(data.qpos[2])
    maximum_torque_ratio = 0.0
    saturated_steps = 0
    nonfoot_contact_steps = 0
    physics_steps = 0
    finite = True
    completed_frames = 0
    torque_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    for frame in range(frames):
        data.ctrl[:] = episode.joint_position[frame]
        target_time = float(frame + 1) / episode.sample_rate_hz
        while data.time + 1e-12 < target_time:
            mujoco.mj_step(model, data)
            physics_steps += 1
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                finite = False
                break
            torque = np.abs(data.qfrc_actuator[dof_addresses])
            ratio = float(np.max(torque / torque_limits))
            maximum_torque_ratio = max(maximum_torque_ratio, ratio)
            saturated_steps += int(ratio >= 0.999)
            minimum_pelvis_height = min(minimum_pelvis_height, float(data.qpos[2]))
            nonfoot_contact_steps += int(
                _has_nonfoot_floor_contact(
                    model,
                    data,
                    floor_geom=floor_geom,
                    allowed_body_ids=allowed_floor_bodies,
                )
            )
        if not finite:
            break
        actual_joint = np.asarray(data.qpos[qpos_addresses], dtype=np.float64)
        joint_errors.append(actual_joint - episode.joint_position[frame])
        root_errors.append(np.asarray(data.qpos[:3]) - aligned_root[frame])
        orientation_errors.append(
            _quaternion_distance(
                np.asarray(data.qpos[3:7]),
                episode.root_quaternion[frame],
            )
        )
        left_positions.append(np.asarray(data.site_xpos[left_site]).copy())
        right_positions.append(np.asarray(data.site_xpos[right_site]).copy())
        completed_frames += 1
    joint_rmse = _array_rmse(joint_errors)
    root_rmse = _array_rmse(root_errors)
    orientation_rmse = (
        float(np.sqrt(np.mean(np.square(orientation_errors)))) if orientation_errors else math.inf
    )
    saturation_ratio = saturated_steps / max(1, physics_steps)
    support_slip_p95 = _support_slip_p95(
        left_positions,
        right_positions,
        contact,
        completed_frames=completed_frames,
        sample_rate_hz=episode.sample_rate_hz,
    )
    warning_count = int(np.sum(data.warning.number))
    issues = _physics_issues(
        finite=finite,
        completed_frames=completed_frames,
        expected_frames=frames,
        duration_s=episode.duration_seconds,
        minimum_pelvis_height=minimum_pelvis_height,
        joint_rmse=joint_rmse,
        root_rmse=root_rmse,
        orientation_rmse=orientation_rmse,
        maximum_torque_ratio=maximum_torque_ratio,
        saturation_ratio=saturation_ratio,
        support_slip_p95=support_slip_p95,
        nonfoot_contact_steps=nonfoot_contact_steps,
        warning_count=warning_count,
        thresholds=selected,
    )
    if any(issue.hard_failure for issue in issues):
        qualification = MotionQualificationLevel.Q1_KINEMATIC_ONLY
    elif issues:
        qualification = MotionQualificationLevel.Q2_TRACKABLE_WITH_REPAIR
    else:
        qualification = MotionQualificationLevel.Q3_PHYSICS_TRACKABLE
    metrics: dict[str, float | int | bool] = {
        "frame_count": frames,
        "completed_frame_count": completed_frames,
        "physics_step_count": physics_steps,
        "simulation_duration_s": float(data.time),
        "reference_duration_s": episode.duration_seconds,
        "finite": finite,
        "minimum_pelvis_height_m": minimum_pelvis_height,
        "joint_tracking_rmse_rad": joint_rmse,
        "root_position_rmse_m": root_rmse,
        "root_orientation_rmse_rad": orientation_rmse,
        "maximum_torque_ratio": maximum_torque_ratio,
        "torque_saturation_ratio": saturation_ratio,
        "support_slip_p95_m_s": support_slip_p95,
        "nonfoot_floor_contact_steps": nonfoot_contact_steps,
        "mujoco_warning_count": warning_count,
    }
    return MotionPhysicsQualification(
        episode_hash=episode.episode_hash,
        contact_trace_hash=contact.trace_hash,
        target_body_hash=target_body_hash,
        target_model_file_hash=target_model_hash,
        scene_file_hash=scene_hash,
        compiled_scene_hash=compiled_scene_hash,
        threshold_hash=selected.threshold_hash,
        qualification=qualification,
        metrics=metrics,
        issues=tuple(issues),
        root_alignment_m=alignment,
    )


def hash_scene_file(scene_path: Path) -> str:
    resolved = scene_path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("MuJoCo scene must be a non-symlink regular file")
    payload = _read_stable_scene(resolved)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _load_physics_scene(
    scene_path: Path,
    *,
    expected_joint_limits: dict[str, tuple[float, float]],
    expected_target_model_hash: str,
) -> tuple[
    Any,
    Any,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    frozenset[int],
    str,
    str,
]:
    scene_hash = hash_scene_file(scene_path)
    try:
        import mujoco
    except ImportError as exc:
        raise ValueError("MuJoCo is required for physics qualification") from exc
    resolved = scene_path.expanduser().resolve(strict=True)
    included_model = resolved.parent / "g1.xml"
    if (
        not included_model.is_file()
        or included_model.is_symlink()
        or "sha256:" + hashlib.sha256(_read_stable_scene(included_model)).hexdigest()
        != expected_target_model_hash
    ):
        raise ValueError("physics scene does not include the committed target G1 model")
    model = mujoco.MjModel.from_xml_path(str(resolved))
    data = mujoco.MjData(model)
    if model.nu != len(G1_DDS_JOINT_NAMES):
        raise ValueError("physics scene must define exactly 29 G1 actuators")
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_names: list[str] = []
    for index, name in enumerate(G1_DDS_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            index,
        )
        if joint_id < 0 or actuator_name != name:
            raise ValueError("physics scene G1 joint/actuator order is incompatible")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_names.append(str(actuator_name))
        observed = tuple(float(item) for item in model.jnt_range[joint_id])
        expected = expected_joint_limits[name]
        if not np.allclose(observed, expected, atol=1e-9, rtol=0.0):
            raise ValueError(f"physics scene joint range mismatch: {name}")
    if tuple(actuator_names) != G1_DDS_JOINT_NAMES:
        raise ValueError("physics scene actuator names do not match Unitree HG order")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    allowed_bodies = frozenset(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    if (
        left_site < 0
        or right_site < 0
        or floor_geom < 0
        or any(item < 0 for item in allowed_bodies)
    ):
        raise ValueError("physics scene lacks floor or canonical G1 foot contracts")
    return (
        model,
        data,
        np.asarray(qpos_addresses, dtype=np.int64),
        np.asarray(dof_addresses, dtype=np.int64),
        int(left_site),
        int(right_site),
        int(floor_geom),
        allowed_bodies,
        scene_hash,
        _compiled_model_hash(model),
    )


def _compiled_model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"rosclaw.collective.mujoco_compiled_scene.v1")
    for label, value in (
        ("body_mass", model.body_mass),
        ("body_inertia", model.body_inertia),
        ("body_pos", model.body_pos),
        ("body_quat", model.body_quat),
        ("jnt_type", model.jnt_type),
        ("jnt_axis", model.jnt_axis),
        ("jnt_pos", model.jnt_pos),
        ("jnt_range", model.jnt_range),
        ("dof_armature", model.dof_armature),
        ("dof_damping", model.dof_damping),
        ("dof_frictionloss", model.dof_frictionloss),
        ("geom_type", model.geom_type),
        ("geom_bodyid", model.geom_bodyid),
        ("geom_size", model.geom_size),
        ("geom_pos", model.geom_pos),
        ("geom_quat", model.geom_quat),
        ("geom_friction", model.geom_friction),
        ("geom_condim", model.geom_condim),
        ("mesh_vert", model.mesh_vert),
        ("mesh_face", model.mesh_face),
        ("actuator_trnid", model.actuator_trnid),
        ("actuator_gear", model.actuator_gear),
        ("actuator_gainprm", model.actuator_gainprm),
        ("actuator_biasprm", model.actuator_biasprm),
        ("actuator_ctrlrange", model.actuator_ctrlrange),
        ("actuator_forcerange", model.actuator_forcerange),
    ):
        array = np.ascontiguousarray(value)
        digest.update(label.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(
        np.asarray(
            [
                model.opt.timestep,
                *model.opt.gravity,
                float(model.opt.integrator),
            ],
            dtype="<f8",
        ).tobytes()
    )
    return "sha256:" + digest.hexdigest()


def _has_nonfoot_floor_contact(
    model: Any,
    data: Any,
    *,
    floor_geom: int,
    allowed_body_ids: frozenset[int],
) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == floor_geom:
            other = int(contact.geom2)
        elif contact.geom2 == floor_geom:
            other = int(contact.geom1)
        else:
            continue
        if int(model.geom_bodyid[other]) not in allowed_body_ids:
            return True
    return False


def _support_slip_p95(
    left_positions: list[np.ndarray],
    right_positions: list[np.ndarray],
    contact: CanonicalContactTrace,
    *,
    completed_frames: int,
    sample_rate_hz: float,
) -> float:
    if completed_frames < 3:
        return math.inf
    left = np.asarray(left_positions, dtype=np.float64)
    right = np.asarray(right_positions, dtype=np.float64)
    left_speed = np.linalg.norm(
        np.gradient(left, 1.0 / sample_rate_hz, axis=0, edge_order=2),
        axis=1,
    )
    right_speed = np.linalg.norm(
        np.gradient(right, 1.0 / sample_rate_hz, axis=0, edge_order=2),
        axis=1,
    )
    samples = np.concatenate(
        (
            left_speed[contact.left_contact[:completed_frames]],
            right_speed[contact.right_contact[:completed_frames]],
        )
    )
    return float(np.quantile(samples, 0.95)) if samples.size else math.inf


def _physics_issues(
    *,
    finite: bool,
    completed_frames: int,
    expected_frames: int,
    duration_s: float,
    minimum_pelvis_height: float,
    joint_rmse: float,
    root_rmse: float,
    orientation_rmse: float,
    maximum_torque_ratio: float,
    saturation_ratio: float,
    support_slip_p95: float,
    nonfoot_contact_steps: int,
    warning_count: int,
    thresholds: PhysicsQualificationThresholds,
) -> list[PhysicsQualificationIssue]:
    values = (
        (
            not finite,
            "NONFINITE_PHYSICS",
            float(not finite),
            0.0,
        ),
        (
            completed_frames != expected_frames,
            "INCOMPLETE_PHYSICS_REPLAY",
            float(completed_frames),
            float(expected_frames),
        ),
        (
            duration_s < thresholds.minimum_duration_s,
            "REFERENCE_DURATION_TOO_SHORT",
            duration_s,
            thresholds.minimum_duration_s,
        ),
        (
            minimum_pelvis_height < thresholds.minimum_pelvis_height_m,
            "PELVIS_HEIGHT_FALL",
            minimum_pelvis_height,
            thresholds.minimum_pelvis_height_m,
        ),
        (
            joint_rmse > thresholds.maximum_joint_tracking_rmse_rad,
            "JOINT_TRACKING_ERROR_HIGH",
            joint_rmse,
            thresholds.maximum_joint_tracking_rmse_rad,
        ),
        (
            root_rmse > thresholds.maximum_root_position_rmse_m,
            "ROOT_POSITION_ERROR_HIGH",
            root_rmse,
            thresholds.maximum_root_position_rmse_m,
        ),
        (
            orientation_rmse > thresholds.maximum_root_orientation_rmse_rad,
            "ROOT_ORIENTATION_ERROR_HIGH",
            orientation_rmse,
            thresholds.maximum_root_orientation_rmse_rad,
        ),
        (
            maximum_torque_ratio > thresholds.maximum_torque_ratio,
            "TORQUE_LIMIT_EXCEEDED",
            maximum_torque_ratio,
            thresholds.maximum_torque_ratio,
        ),
        (
            saturation_ratio > thresholds.maximum_torque_saturation_ratio,
            "TORQUE_SATURATION_HIGH",
            saturation_ratio,
            thresholds.maximum_torque_saturation_ratio,
        ),
        (
            support_slip_p95 > thresholds.maximum_support_slip_p95_m_s,
            "SUPPORT_SLIP_HIGH",
            support_slip_p95,
            thresholds.maximum_support_slip_p95_m_s,
        ),
        (
            nonfoot_contact_steps > thresholds.maximum_nonfoot_contact_steps,
            "NON_FOOT_FLOOR_CONTACT",
            float(nonfoot_contact_steps),
            float(thresholds.maximum_nonfoot_contact_steps),
        ),
        (
            warning_count > 0,
            "MUJOCO_WARNING",
            float(warning_count),
            0.0,
        ),
    )
    return [
        PhysicsQualificationIssue(code=code, observed=observed, threshold=threshold)
        for failed, code, observed, threshold in values
        if failed
    ]


def _contact_rejected_physics(
    contact: MotionDecodeContactAudit,
) -> MotionDecodePhysicsClip:
    qualification = (
        MotionQualificationLevel.Q1_KINEMATIC_ONLY
        if contact.upstream_q1
        else MotionQualificationLevel.Q0_INVALID
    )
    return MotionDecodePhysicsClip(
        relative_path=contact.relative_path,
        source_file_hash=contact.source_file_hash,
        contact_audit_hash=contact.audit_hash,
        status=PhysicsClipStatus.CONTACT_NOT_ELIGIBLE,
        qualification=qualification,
        blocker_codes=tuple(
            issue.code for issue in contact.issues if issue.severity is AuditSeverity.ERROR
        ),
    )


def _license_blocked_physics(
    contact: MotionDecodeContactAudit,
) -> MotionDecodePhysicsClip:
    return MotionDecodePhysicsClip(
        relative_path=contact.relative_path,
        source_file_hash=contact.source_file_hash,
        contact_audit_hash=contact.audit_hash,
        status=PhysicsClipStatus.BLOCKED_LICENSE,
        qualification=MotionQualificationLevel.Q1_KINEMATIC_ONLY,
        blocker_codes=("LICENSE_NOT_PERMITTED",),
    )


def _unit_quaternion(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("reference quaternion is invalid")
    return np.asarray(value, dtype=np.float64) / norm


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = _unit_quaternion(first)
    right = _unit_quaternion(second)
    cosine = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def _array_rmse(values: list[np.ndarray]) -> float:
    if not values:
        return math.inf
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _read_stable_scene(path: Path) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read(_MAX_SCENE_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("MuJoCo scene changed while it was read")
    if len(payload) > _MAX_SCENE_BYTES:
        raise ValueError("MuJoCo scene exceeds the 20 MB safety limit")
    return payload


__all__ = [
    "MotionDecodePhysicsClip",
    "MotionDecodeQualificationReport",
    "MotionPhysicsQualification",
    "PhysicsClipStatus",
    "PhysicsQualificationIssue",
    "PhysicsQualificationThresholds",
    "hash_scene_file",
    "qualify_canonical_motion",
    "qualify_motiondecode_snapshot",
]
