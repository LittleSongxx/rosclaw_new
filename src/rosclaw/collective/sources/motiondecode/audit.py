"""Kinematic qualification and collective-capsule closure for MotionDecode."""

from __future__ import annotations

import csv
import hashlib
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.contracts import (
    ApplicabilityAssessment,
    ApplicabilityDecision,
    CollectiveUse,
    ExperienceCapsule,
    LicenseDecision,
)
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeFileRecord,
    MotionDecodeRegistration,
    verify_registered_files,
)
from rosclaw.collective.sources.motiondecode.parser import (
    CanonicalMotionEpisode,
    parse_motion_csv,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import EvidenceLevel, EvidenceUsePolicy
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES


class MotionQualificationLevel(StrEnum):
    Q0_INVALID = "q0_invalid"
    Q1_KINEMATIC_ONLY = "q1_kinematic_only"
    Q2_TRACKABLE_WITH_REPAIR = "q2_trackable_with_repair"
    Q3_PHYSICS_TRACKABLE = "q3_physics_trackable"
    Q4_ROBUST_TRACKABLE = "q4_robust_trackable"


class AuditSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class MotionDecodeAuditThresholds:
    quaternion_norm_warning: float = 0.005
    quaternion_norm_error: float = 0.05
    root_step_warning_m: float = 0.05
    root_step_error_m: float = 0.10
    terminal_root_match_m: float = 0.002
    terminal_quaternion_match: float = 0.01
    terminal_joint_match_rad: float = 0.05
    loop_closure_search_frames: int = 64
    loop_closure_root_speed_error_m_s: float = 2.0
    loop_closure_joint_speed_error_rad_s: float = 10.0
    joint_limit_tolerance_rad: float = 0.02
    joint_velocity_warning_rad_s: float = 30.0
    joint_acceleration_warning_rad_s2: float = 1000.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("kinematic audit thresholds must be finite and positive")
        if not isinstance(self.loop_closure_search_frames, int):
            raise ValueError("loop_closure_search_frames must be an integer")
        if self.quaternion_norm_warning >= self.quaternion_norm_error:
            raise ValueError("quaternion warning threshold must be below error threshold")
        if self.root_step_warning_m >= self.root_step_error_m:
            raise ValueError("root-step warning threshold must be below error threshold")

    def to_dict(self) -> dict[str, float | int]:
        return dict(vars(self))


@dataclass(frozen=True)
class MotionDecodeAuditIssue:
    code: str
    severity: AuditSeverity
    observed: float
    threshold: float
    count: int

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("audit issue code must not be empty")
        if not isinstance(self.severity, AuditSeverity):
            raise ValueError("audit issue severity is unknown")
        if not math.isfinite(self.observed) or not math.isfinite(self.threshold):
            raise ValueError("audit issue values must be finite")
        if self.count <= 0:
            raise ValueError("audit issue count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "count": self.count,
        }


@dataclass(frozen=True)
class TerminalResetDetection:
    """One high-speed reset to the initial pose near a clip tail."""

    transition_frame: int
    reset_frame: int
    root_match_error_m: float
    quaternion_match_error: float
    joint_match_error_rad: float
    root_speed_m_s: float
    joint_speed_rad_s: float
    schema_version: str = "rosclaw.collective.motiondecode_terminal_reset.v1"

    def __post_init__(self) -> None:
        if self.transition_frame < 0 or self.reset_frame != self.transition_frame + 1:
            raise ValueError("terminal reset frame indices are inconsistent")
        values = (
            self.root_match_error_m,
            self.quaternion_match_error,
            self.joint_match_error_rad,
            self.root_speed_m_s,
            self.joint_speed_rad_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("terminal reset measurements must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transition_frame": self.transition_frame,
            "reset_frame": self.reset_frame,
            "root_match_error_m": self.root_match_error_m,
            "quaternion_match_error": self.quaternion_match_error,
            "joint_match_error_rad": self.joint_match_error_rad,
            "root_speed_m_s": self.root_speed_m_s,
            "joint_speed_rad_s": self.joint_speed_rad_s,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TerminalResetDetection:
        if value.get("schema_version") != cls.schema_version:
            raise ValueError("terminal reset schema version is unsupported")
        return cls(
            transition_frame=int(value["transition_frame"]),
            reset_frame=int(value["reset_frame"]),
            root_match_error_m=float(value["root_match_error_m"]),
            quaternion_match_error=float(value["quaternion_match_error"]),
            joint_match_error_rad=float(value["joint_match_error_rad"]),
            root_speed_m_s=float(value["root_speed_m_s"]),
            joint_speed_rad_s=float(value["joint_speed_rad_s"]),
        )


@dataclass(frozen=True)
class MotionDecodeClipAudit:
    relative_path: str
    source_file_hash: str
    qualification: MotionQualificationLevel
    issues: tuple[MotionDecodeAuditIssue, ...]
    episode_summary: dict[str, Any] | None
    parser_error: str | None = None
    schema_version: str = "rosclaw.collective.motiondecode_clip_audit.v1"

    @property
    def kinematic_valid(self) -> bool:
        return (
            self.qualification is MotionQualificationLevel.Q1_KINEMATIC_ONLY
            and self.parser_error is None
            and not any(issue.severity is AuditSeverity.ERROR for issue in self.issues)
        )

    @property
    def physics_training_eligible(self) -> bool:
        return False

    @property
    def audit_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "source_file_hash": self.source_file_hash,
            "qualification": self.qualification.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "episode_summary": self.episode_summary,
            "parser_error": self.parser_error,
            "kinematic_valid": self.kinematic_valid,
            "physics_training_eligible": self.physics_training_eligible,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodeIngestReport:
    registration_hash: str
    source_manifest_hash: str
    target_body_hash: str
    target_model_file_hash: str
    license_decision: LicenseDecision
    catalog_schema_valid: bool
    thresholds: MotionDecodeAuditThresholds
    clips: tuple[MotionDecodeClipAudit, ...]
    experience_capsule: ExperienceCapsule | None
    schema_version: str = "rosclaw.collective.motiondecode_ingest_report.v1"

    @property
    def quality_commitment(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_quality_commitment.v1",
                "thresholds": self.thresholds.to_dict(),
                "clip_audit_hashes": [clip.audit_hash for clip in self.clips],
            }
        )

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def kinematic_valid_count(self) -> int:
        return sum(clip.kinematic_valid for clip in self.clips)

    @property
    def qualification_counts(self) -> dict[str, int]:
        counts = Counter(clip.qualification.value for clip in self.clips)
        return dict(sorted(counts.items()))

    @property
    def issue_clip_counts(self) -> dict[str, int]:
        counts = Counter(issue.code for clip in self.clips for issue in clip.issues)
        return dict(sorted(counts.items()))

    @property
    def segmentation_repair_candidate_count(self) -> int:
        repair_codes = {
            "ROOT_LOOP_CLOSURE_DISCONTINUITY",
            "JOINT_LOOP_CLOSURE_DISCONTINUITY",
        }
        return sum(any(issue.code in repair_codes for issue in clip.issues) for clip in self.clips)

    @property
    def training_eligible(self) -> bool:
        return False

    @property
    def training_blockers(self) -> list[str]:
        blockers = [
            "Q3_OR_Q4_PHYSICS_QUALIFICATION_REQUIRED",
            "SOURCE_COORDINATE_FRAME_UNSPECIFIED",
            "SYNCHRONIZED_BALL_POSE_ABSENT",
            "ACTION_REWARD_TRANSITION_SEMANTICS_ABSENT",
        ]
        if self.kinematic_valid_count == 0:
            blockers.insert(0, "NO_KINEMATICALLY_VALID_CLIPS")
        elif self.kinematic_valid_count != len(self.clips):
            blockers.insert(0, "KINEMATIC_AUDIT_PARTIAL")
        if self.license_decision is not LicenseDecision.PERMITTED:
            blockers.insert(0, "LICENSE_NOT_PERMITTED")
        if not self.catalog_schema_valid:
            blockers.insert(0, "CATALOG_SCHEMA_INVALID")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_hash": self.registration_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "license_decision": self.license_decision.value,
            "catalog_schema_valid": self.catalog_schema_valid,
            "thresholds": self.thresholds.to_dict(),
            "clips": [clip.to_dict() for clip in self.clips],
            "clip_count": len(self.clips),
            "kinematic_valid_count": self.kinematic_valid_count,
            "qualification_counts": self.qualification_counts,
            "issue_clip_counts": self.issue_clip_counts,
            "segmentation_repair_candidate_count": (self.segmentation_repair_candidate_count),
            "manual_review_required_count": (
                len(self.clips) - self.segmentation_repair_candidate_count
            ),
            "quality_commitment": self.quality_commitment,
            "experience_capsule": (
                self.experience_capsule.to_dict() if self.experience_capsule else None
            ),
            "experience_capsule_hash": (
                self.experience_capsule.capsule_hash if self.experience_capsule else None
            ),
            "eligibility": {
                "motion_reference_discovery": self.kinematic_valid_count > 0,
                "motion_tracker_training": False,
                "football_contact_training": False,
                "behavior_cloning": False,
                "offline_rl": False,
                "promotion_truth": False,
            },
            "training_eligible": self.training_eligible,
            "training_blockers": self.training_blockers,
            "hardware_authorized": False,
        }


def audit_motiondecode_snapshot(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    thresholds: MotionDecodeAuditThresholds | None = None,
) -> MotionDecodeIngestReport:
    """Replay source hashes, parse registered clips and stop at Q1."""

    limits, target_body_hash, model_hash = load_g1_joint_contract(target_model_path)
    verified = verify_registered_files(registration, dataset_root)
    selected_thresholds = thresholds or MotionDecodeAuditThresholds()
    clips: list[MotionDecodeClipAudit] = []
    for record in registration.manifest.files:
        if not record.relative_path.startswith("samples/"):
            continue
        clips.append(
            _audit_record(
                record,
                verified[record.relative_path],
                source_manifest_hash=registration.manifest.manifest_hash,
                target_body_hash=target_body_hash,
                joint_limits=limits,
                thresholds=selected_thresholds,
                sample_rate_hz=registration.manifest.sample_rate_hz,
            )
        )
    quality_commitment = canonical_hash(
        {
            "schema_version": "rosclaw.collective.motiondecode_quality_commitment.v1",
            "thresholds": selected_thresholds.to_dict(),
            "clip_audit_hashes": [clip.audit_hash for clip in clips],
        }
    )
    capsule = _build_capsule(
        registration,
        tuple(clips),
        target_body_hash=target_body_hash,
        quality_commitment=quality_commitment,
    )
    return MotionDecodeIngestReport(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        target_body_hash=target_body_hash,
        target_model_file_hash=model_hash,
        license_decision=registration.manifest.license_snapshot.decision,
        catalog_schema_valid=registration.catalog_audit.schema_valid,
        thresholds=selected_thresholds,
        clips=tuple(clips),
        experience_capsule=capsule,
    )


def _audit_record(
    record: MotionDecodeFileRecord,
    path: Path,
    *,
    source_manifest_hash: str,
    target_body_hash: str,
    joint_limits: dict[str, tuple[float, float]],
    thresholds: MotionDecodeAuditThresholds,
    sample_rate_hz: float,
) -> MotionDecodeClipAudit:
    try:
        episode = parse_motion_csv(
            path,
            source_manifest_hash=source_manifest_hash,
            expected_file_hash=record.content_hash,
            target_body_hash=target_body_hash,
            sample_rate_hz=sample_rate_hz,
        )
        issues = audit_canonical_episode(episode, joint_limits, thresholds)
    except (OSError, ValueError, csv.Error) as exc:
        return MotionDecodeClipAudit(
            relative_path=record.relative_path,
            source_file_hash=record.content_hash,
            qualification=MotionQualificationLevel.Q0_INVALID,
            issues=(),
            episode_summary=None,
            parser_error=f"{type(exc).__name__}: {exc}",
        )
    qualification = (
        MotionQualificationLevel.Q0_INVALID
        if any(issue.severity is AuditSeverity.ERROR for issue in issues)
        else MotionQualificationLevel.Q1_KINEMATIC_ONLY
    )
    return MotionDecodeClipAudit(
        relative_path=record.relative_path,
        source_file_hash=record.content_hash,
        qualification=qualification,
        issues=tuple(issues),
        episode_summary=episode.summary(),
    )


def audit_canonical_episode(
    episode: CanonicalMotionEpisode,
    joint_limits: dict[str, tuple[float, float]],
    thresholds: MotionDecodeAuditThresholds,
) -> list[MotionDecodeAuditIssue]:
    """Apply the reusable Q1 kinematic checks to a canonical episode."""

    issues: list[MotionDecodeAuditIssue] = []
    norms = np.linalg.norm(episode.root_quaternion, axis=1)
    quaternion_error = np.abs(norms - 1.0)
    maximum_quaternion_error = float(np.max(quaternion_error))
    if maximum_quaternion_error > thresholds.quaternion_norm_error:
        issues.append(
            MotionDecodeAuditIssue(
                code="QUATERNION_NORM_ERROR",
                severity=AuditSeverity.ERROR,
                observed=maximum_quaternion_error,
                threshold=thresholds.quaternion_norm_error,
                count=int(np.count_nonzero(quaternion_error > thresholds.quaternion_norm_error)),
            )
        )
    elif maximum_quaternion_error > thresholds.quaternion_norm_warning:
        issues.append(
            MotionDecodeAuditIssue(
                code="QUATERNION_NORM_WARNING",
                severity=AuditSeverity.WARNING,
                observed=maximum_quaternion_error,
                threshold=thresholds.quaternion_norm_warning,
                count=int(np.count_nonzero(quaternion_error > thresholds.quaternion_norm_warning)),
            )
        )
    root_steps = np.linalg.norm(np.diff(episode.root_position, axis=0), axis=1)
    maximum_root_step = float(np.max(root_steps))
    issues.extend(_loop_closure_issues(episode, thresholds))
    if maximum_root_step > thresholds.root_step_error_m:
        issues.append(
            MotionDecodeAuditIssue(
                code="ROOT_STEP_ERROR",
                severity=AuditSeverity.ERROR,
                observed=maximum_root_step,
                threshold=thresholds.root_step_error_m,
                count=int(np.count_nonzero(root_steps > thresholds.root_step_error_m)),
            )
        )
    elif maximum_root_step > thresholds.root_step_warning_m:
        issues.append(
            MotionDecodeAuditIssue(
                code="ROOT_STEP_WARNING",
                severity=AuditSeverity.WARNING,
                observed=maximum_root_step,
                threshold=thresholds.root_step_warning_m,
                count=int(np.count_nonzero(root_steps > thresholds.root_step_warning_m)),
            )
        )
    limit_excesses: list[np.ndarray] = []
    for index, joint_name in enumerate(episode.mapping.target_joint_names):
        lower, upper = joint_limits[joint_name]
        values = episode.joint_position[:, index]
        limit_excesses.append(np.maximum(lower - values, values - upper))
    excess = np.maximum(np.column_stack(limit_excesses), 0.0)
    maximum_excess = float(np.max(excess))
    if maximum_excess > thresholds.joint_limit_tolerance_rad:
        issues.append(
            MotionDecodeAuditIssue(
                code="JOINT_LIMIT_ERROR",
                severity=AuditSeverity.ERROR,
                observed=maximum_excess,
                threshold=thresholds.joint_limit_tolerance_rad,
                count=int(np.count_nonzero(excess > thresholds.joint_limit_tolerance_rad)),
            )
        )
    maximum_velocity = float(np.max(np.abs(episode.joint_velocity)))
    if maximum_velocity > thresholds.joint_velocity_warning_rad_s:
        issues.append(
            MotionDecodeAuditIssue(
                code="JOINT_VELOCITY_WARNING",
                severity=AuditSeverity.WARNING,
                observed=maximum_velocity,
                threshold=thresholds.joint_velocity_warning_rad_s,
                count=int(
                    np.count_nonzero(
                        np.abs(episode.joint_velocity) > thresholds.joint_velocity_warning_rad_s
                    )
                ),
            )
        )
    maximum_acceleration = float(np.max(np.abs(episode.joint_acceleration)))
    if maximum_acceleration > thresholds.joint_acceleration_warning_rad_s2:
        issues.append(
            MotionDecodeAuditIssue(
                code="JOINT_ACCELERATION_WARNING",
                severity=AuditSeverity.WARNING,
                observed=maximum_acceleration,
                threshold=thresholds.joint_acceleration_warning_rad_s2,
                count=int(
                    np.count_nonzero(
                        np.abs(episode.joint_acceleration)
                        > thresholds.joint_acceleration_warning_rad_s2
                    )
                ),
            )
        )
    return issues


def _loop_closure_issues(
    episode: CanonicalMotionEpisode,
    thresholds: MotionDecodeAuditThresholds,
) -> list[MotionDecodeAuditIssue]:
    """Find an appended reset-to-start frame near a non-cyclic clip tail."""

    detections = detect_terminal_resets(episode, thresholds)
    root_speeds = [
        item.root_speed_m_s
        for item in detections
        if item.root_speed_m_s > thresholds.loop_closure_root_speed_error_m_s
    ]
    joint_speeds = [
        item.joint_speed_rad_s
        for item in detections
        if item.joint_speed_rad_s > thresholds.loop_closure_joint_speed_error_rad_s
    ]
    issues: list[MotionDecodeAuditIssue] = []
    if root_speeds:
        issues.append(
            MotionDecodeAuditIssue(
                code="ROOT_LOOP_CLOSURE_DISCONTINUITY",
                severity=AuditSeverity.ERROR,
                observed=max(root_speeds),
                threshold=thresholds.loop_closure_root_speed_error_m_s,
                count=len(root_speeds),
            )
        )
    if joint_speeds:
        issues.append(
            MotionDecodeAuditIssue(
                code="JOINT_LOOP_CLOSURE_DISCONTINUITY",
                severity=AuditSeverity.ERROR,
                observed=max(joint_speeds),
                threshold=thresholds.loop_closure_joint_speed_error_rad_s,
                count=len(joint_speeds),
            )
        )
    return issues


def detect_terminal_resets(
    episode: CanonicalMotionEpisode,
    thresholds: MotionDecodeAuditThresholds,
) -> tuple[TerminalResetDetection, ...]:
    """Return bounded reset-to-initial discontinuities near the clip tail."""

    transition_count = episode.time.shape[0] - 1
    start = max(0, transition_count - thresholds.loop_closure_search_frames)
    detections: list[TerminalResetDetection] = []
    for index in range(start, transition_count):
        post = index + 1
        root_match = float(np.linalg.norm(episode.root_position[post] - episode.root_position[0]))
        quaternion_match = float(
            np.linalg.norm(episode.root_quaternion[post] - episode.root_quaternion[0])
        )
        joint_match = float(
            np.max(np.abs(episode.joint_position[post] - episode.joint_position[0]))
        )
        if (
            root_match > thresholds.terminal_root_match_m
            or quaternion_match > thresholds.terminal_quaternion_match
            or joint_match > thresholds.terminal_joint_match_rad
        ):
            continue
        step_seconds = float(episode.time[post] - episode.time[index])
        root_speed = float(
            np.linalg.norm(episode.root_position[post] - episode.root_position[index])
            / step_seconds
        )
        joint_speed = float(
            np.max(np.abs(episode.joint_position[post] - episode.joint_position[index]))
            / step_seconds
        )
        if (
            root_speed <= thresholds.loop_closure_root_speed_error_m_s
            and joint_speed <= thresholds.loop_closure_joint_speed_error_rad_s
        ):
            continue
        detections.append(
            TerminalResetDetection(
                transition_frame=index,
                reset_frame=post,
                root_match_error_m=root_match,
                quaternion_match_error=quaternion_match,
                joint_match_error_rad=joint_match,
                root_speed_m_s=root_speed,
                joint_speed_rad_s=joint_speed,
            )
        )
    return tuple(detections)


def _build_capsule(
    registration: MotionDecodeRegistration,
    clips: tuple[MotionDecodeClipAudit, ...],
    *,
    target_body_hash: str,
    quality_commitment: str,
) -> ExperienceCapsule | None:
    valid = tuple(clip for clip in clips if clip.kinematic_valid and clip.episode_summary)
    if not valid:
        return None
    mapping_hashes = {
        str(clip.episode_summary["mapping_hash"])
        for clip in valid
        if clip.episode_summary is not None
    }
    if len(mapping_hashes) != 1:
        raise ValueError("kinematically valid clips do not share one target mapping")
    mapping_hash = next(iter(mapping_hashes))
    applicability = ApplicabilityAssessment(
        target_body_hash=target_body_hash,
        target_mapping_hash=mapping_hash,
        body_score=1.0,
        task_score=0.5,
        regime_score=0.0,
        confidence=0.5,
        evidence_hashes=(quality_commitment,),
        decision=ApplicabilityDecision.PENDING,
    )
    valid_paths = {clip.relative_path for clip in valid}
    observed_families = sorted(
        {
            record.family.value
            for record in registration.manifest.files
            if record.relative_path in valid_paths
        }
    )
    task_semantics_hash = canonical_hash(
        {
            "schema_version": "rosclaw.collective.motiondecode_task_semantics.v1",
            "families": observed_families,
            "football_contact_semantics": False,
        }
    )
    observation_semantics_hash = canonical_hash(
        {
            "schema_version": "rosclaw.collective.motiondecode_observation_semantics.v1",
            "root_position": {"unit": "m", "frame": "source_world_unspecified"},
            "root_quaternion": {"order": "wxyz", "frame": "source_world_unspecified"},
            "joint_position": {
                "unit": "rad",
                "order": list(G1_DDS_JOINT_NAMES),
            },
            "sample_rate_hz": registration.manifest.sample_rate_hz,
        }
    )
    return ExperienceCapsule(
        source=registration.manifest.source_identity,
        applicability=applicability,
        task_semantics_hash=task_semantics_hash,
        observation_semantics_hash=observation_semantics_hash,
        modalities=("root_pose", "joint_position"),
        requested_uses=(CollectiveUse.MOTION_REFERENCE,),
        quality_report_hash=quality_commitment,
        evidence_policy=EvidenceUsePolicy(EvidenceLevel.EXTERNAL_UNVERIFIED),
        source_episode_count=len(valid),
    )


def load_g1_joint_contract(
    model_path: Path,
) -> tuple[dict[str, tuple[float, float]], str, str]:
    resolved = model_path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("target model must be a non-symlink regular file")
    if resolved.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("target model exceeds the 10 MB parser safety limit")
    payload = resolved.read_bytes()
    file_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    root = ET.fromstring(payload)
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if name is None or name not in G1_DDS_JOINT_NAMES:
            continue
        raw_range = joint.attrib.get("range")
        if raw_range is None:
            raise ValueError(f"target model joint lacks a range: {name}")
        values = tuple(float(item) for item in raw_range.split())
        if len(values) != 2 or not all(math.isfinite(item) for item in values):
            raise ValueError(f"target model joint has an invalid range: {name}")
        limits[name] = (values[0], values[1])
    if set(limits) != set(G1_DDS_JOINT_NAMES):
        missing = sorted(set(G1_DDS_JOINT_NAMES) - set(limits))
        raise ValueError(f"target model does not define the full G1 joint contract: {missing}")
    target_body_hash = canonical_hash(
        {
            "schema_version": "rosclaw.body.unitree_g1_joint_contract.v1",
            "model_file_hash": file_hash,
            "joint_names": list(G1_DDS_JOINT_NAMES),
            "joint_limits": {name: list(limits[name]) for name in G1_DDS_JOINT_NAMES},
        }
    )
    return limits, target_body_hash, file_hash


__all__ = [
    "AuditSeverity",
    "MotionDecodeAuditIssue",
    "MotionDecodeAuditThresholds",
    "MotionDecodeClipAudit",
    "MotionDecodeIngestReport",
    "MotionQualificationLevel",
    "TerminalResetDetection",
    "audit_canonical_episode",
    "audit_motiondecode_snapshot",
    "detect_terminal_resets",
    "load_g1_joint_contract",
]
