"""Deterministic kinematic contact and support-phase inference."""

from __future__ import annotations

import hashlib
import math
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
    load_g1_joint_contract,
)
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeRegistration,
    verify_registered_files,
)
from rosclaw.collective.sources.motiondecode.parser import (
    CanonicalMotionEpisode,
    parse_motion_csv,
)
from rosclaw.collective.sources.motiondecode.repair import (
    MotionDecodeRepairResult,
    MotionRepairDisposition,
    repair_motiondecode_snapshot,
    replay_segmentation_repair,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_MAX_MODEL_BYTES = 10 * 1024 * 1024


class SupportPhase(StrEnum):
    FLIGHT = "flight"
    LEFT = "left"
    RIGHT = "right"
    DOUBLE = "double"


_PHASE_CODE = {
    SupportPhase.FLIGHT: 0,
    SupportPhase.LEFT: 1,
    SupportPhase.RIGHT: 2,
    SupportPhase.DOUBLE: 3,
}


@dataclass(frozen=True)
class ContactInferenceThresholds:
    ground_quantile: float = 0.05
    sole_offset_m: float = 0.03
    contact_enter_height_m: float = 0.025
    contact_exit_height_m: float = 0.04
    contact_enter_speed_m_s: float = 0.30
    contact_exit_speed_m_s: float = 0.45
    minimum_contact_frames: int = 3
    maximum_gap_frames: int = 2
    minimum_supported_ratio: float = 0.75
    maximum_flight_run_s: float = 0.50
    maximum_skating_ratio: float = 0.10
    maximum_near_ground_speed_p95_m_s: float = 0.50

    def __post_init__(self) -> None:
        numeric = tuple(vars(self).values())
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in numeric
        ):
            raise ValueError("contact inference thresholds must be finite and positive")
        if not 0.0 < self.ground_quantile < 0.5:
            raise ValueError("ground_quantile must be in (0, 0.5)")
        if not 0.0 < self.minimum_supported_ratio <= 1.0:
            raise ValueError("minimum_supported_ratio must be in (0, 1]")
        if not 0.0 < self.maximum_skating_ratio <= 1.0:
            raise ValueError("maximum_skating_ratio must be in (0, 1]")
        if not isinstance(self.minimum_contact_frames, int) or not isinstance(
            self.maximum_gap_frames, int
        ):
            raise ValueError("contact frame thresholds must be integers")
        if self.contact_enter_height_m >= self.contact_exit_height_m:
            raise ValueError("contact enter height must be below exit height")
        if self.contact_enter_speed_m_s >= self.contact_exit_speed_m_s:
            raise ValueError("contact enter speed must be below exit speed")

    @property
    def threshold_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_contact_thresholds.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, float | int]:
        return dict(vars(self))


@dataclass(frozen=True)
class ContactInferenceIssue:
    code: str
    severity: AuditSeverity
    observed: float
    threshold: float

    def __post_init__(self) -> None:
        if not self.code or not isinstance(self.severity, AuditSeverity):
            raise ValueError("contact inference issue is invalid")
        if not math.isfinite(self.observed) or not math.isfinite(self.threshold):
            raise ValueError("contact inference issue values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class CanonicalContactTrace:
    """Read-only derived arrays; only their hash and summary enter receipts."""

    episode_hash: str
    target_model_file_hash: str
    threshold_hash: str
    ground_height_m: float
    left_sole_position: np.ndarray
    right_sole_position: np.ndarray
    center_of_mass: np.ndarray
    left_foot_speed: np.ndarray
    right_foot_speed: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    phase_code: np.ndarray
    schema_version: str = "rosclaw.collective.motiondecode_contact_trace.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("episode_hash", self.episode_hash),
            ("target_model_file_hash", self.target_model_file_hash),
            ("threshold_hash", self.threshold_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not math.isfinite(self.ground_height_m):
            raise ValueError("ground height must be finite")
        frames = int(self.phase_code.shape[0])
        expected = {
            "left_sole_position": ((frames, 3), np.float64),
            "right_sole_position": ((frames, 3), np.float64),
            "center_of_mass": ((frames, 3), np.float64),
            "left_foot_speed": ((frames,), np.float64),
            "right_foot_speed": ((frames,), np.float64),
            "left_contact": ((frames,), np.bool_),
            "right_contact": ((frames,), np.bool_),
            "phase_code": ((frames,), np.uint8),
        }
        if frames < 3:
            raise ValueError("contact trace requires at least three frames")
        for label, (shape, dtype) in expected.items():
            array = np.asarray(getattr(self, label), dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{label} must have shape {shape}")
            if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
                raise ValueError(f"{label} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, label, array)
        if np.any(self.phase_code > max(_PHASE_CODE.values())):
            raise ValueError("contact trace contains an unknown support phase")

    @property
    def trace_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema_version.encode("ascii"))
        digest.update(self.episode_hash.encode("ascii"))
        digest.update(self.target_model_file_hash.encode("ascii"))
        digest.update(self.threshold_hash.encode("ascii"))
        digest.update(np.asarray([self.ground_height_m], dtype="<f8").tobytes())
        for array in (
            self.left_sole_position,
            self.right_sole_position,
            self.center_of_mass,
            self.left_foot_speed,
            self.right_foot_speed,
        ):
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.left_contact, dtype=np.uint8).tobytes())
        digest.update(np.ascontiguousarray(self.right_contact, dtype=np.uint8).tobytes())
        digest.update(np.ascontiguousarray(self.phase_code, dtype=np.uint8).tobytes())
        return "sha256:" + digest.hexdigest()

    @property
    def phases(self) -> tuple[SupportPhase, ...]:
        by_code = {code: phase for phase, code in _PHASE_CODE.items()}
        return tuple(by_code[int(value)] for value in self.phase_code)


@dataclass(frozen=True)
class MotionDecodeContactAudit:
    relative_path: str
    source_file_hash: str
    repair_manifest_hash: str | None
    episode_hash: str | None
    trace_hash: str | None
    metrics: dict[str, float | int] | None
    issues: tuple[ContactInferenceIssue, ...]
    upstream_q1: bool
    schema_version: str = "rosclaw.collective.motiondecode_contact_audit.v1"

    def __post_init__(self) -> None:
        if not self.relative_path:
            raise ValueError("contact audit relative path must not be empty")
        for label, value in (
            ("source_file_hash", self.source_file_hash),
            ("repair_manifest_hash", self.repair_manifest_hash),
            ("episode_hash", self.episode_hash),
            ("trace_hash", self.trace_hash),
        ):
            if value is not None and (not value.startswith("sha256:") or len(value) != 71):
                raise ValueError(f"{label} must be a sha256: content hash")
        if not self.issues or any(
            not isinstance(issue, ContactInferenceIssue) for issue in self.issues
        ):
            raise ValueError("contact audit requires inference issues")
        if self.upstream_q1:
            if self.episode_hash is None or self.trace_hash is None or self.metrics is None:
                raise ValueError("upstream Q1 contact audit requires derived summary evidence")
        elif any(
            value is not None
            for value in (
                self.repair_manifest_hash,
                self.episode_hash,
                self.trace_hash,
                self.metrics,
            )
        ):
            raise ValueError("upstream rejected contact audit cannot claim derived evidence")

    @property
    def phase_segmentation_candidate(self) -> bool:
        return (
            self.upstream_q1
            and self.trace_hash is not None
            and not any(issue.severity is AuditSeverity.ERROR for issue in self.issues)
        )

    @property
    def audit_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "source_file_hash": self.source_file_hash,
            "repair_manifest_hash": self.repair_manifest_hash,
            "episode_hash": self.episode_hash,
            "trace_hash": self.trace_hash,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
            "upstream_q1": self.upstream_q1,
            "phase_segmentation_candidate": self.phase_segmentation_candidate,
            "frame_level_trace_persisted": False,
            "training_eligible": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodeContactReport:
    registration_hash: str
    source_manifest_hash: str
    ingest_report_hash: str
    repair_report_hash: str
    target_body_hash: str
    target_model_file_hash: str
    license_decision: LicenseDecision
    thresholds: ContactInferenceThresholds
    clips: tuple[MotionDecodeContactAudit, ...]
    schema_version: str = "rosclaw.collective.motiondecode_contact_report.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("registration_hash", self.registration_hash),
            ("source_manifest_hash", self.source_manifest_hash),
            ("ingest_report_hash", self.ingest_report_hash),
            ("repair_report_hash", self.repair_report_hash),
            ("target_body_hash", self.target_body_hash),
            ("target_model_file_hash", self.target_model_file_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not isinstance(self.license_decision, LicenseDecision):
            raise ValueError("contact report license decision is unknown")
        if not isinstance(self.thresholds, ContactInferenceThresholds):
            raise ValueError("contact report thresholds are invalid")
        clips = tuple(self.clips)
        if any(not isinstance(clip, MotionDecodeContactAudit) for clip in clips):
            raise ValueError("contact report contains an invalid clip")
        paths = tuple(clip.relative_path for clip in clips)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("contact report clips must be unique and sorted")
        object.__setattr__(self, "clips", clips)

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def inferred_count(self) -> int:
        return sum(clip.trace_hash is not None for clip in self.clips)

    @property
    def phase_candidate_count(self) -> int:
        return sum(clip.phase_segmentation_candidate for clip in self.clips)

    @property
    def issue_clip_counts(self) -> dict[str, int]:
        counts = Counter(issue.code for clip in self.clips for issue in clip.issues)
        return dict(sorted(counts.items()))

    @property
    def quality_commitment(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_contact_quality.v1",
                "repair_report_hash": self.repair_report_hash,
                "threshold_hash": self.thresholds.threshold_hash,
                "clip_audit_hashes": [clip.audit_hash for clip in self.clips],
            }
        )

    @property
    def training_blockers(self) -> list[str]:
        blockers = [
            "Q3_OR_Q4_PHYSICS_QUALIFICATION_REQUIRED",
            "SOURCE_COORDINATE_FRAME_UNSPECIFIED",
            "SYNCHRONIZED_BALL_POSE_ABSENT",
            "ACTION_REWARD_TRANSITION_SEMANTICS_ABSENT",
            "FRAME_LEVEL_CONTACT_TRACE_NOT_PERSISTED",
        ]
        if self.phase_candidate_count != self.inferred_count:
            blockers.insert(0, "CONTACT_INFERENCE_PARTIAL")
        if self.inferred_count != len(self.clips):
            blockers.insert(0, "UPSTREAM_Q1_PARTIAL")
        if self.license_decision is not LicenseDecision.PERMITTED:
            blockers.insert(0, "LICENSE_NOT_PERMITTED")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_hash": self.registration_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "ingest_report_hash": self.ingest_report_hash,
            "repair_report_hash": self.repair_report_hash,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "license_decision": self.license_decision.value,
            "thresholds": self.thresholds.to_dict(),
            "threshold_hash": self.thresholds.threshold_hash,
            "clips": [clip.to_dict() for clip in self.clips],
            "clip_count": len(self.clips),
            "inferred_count": self.inferred_count,
            "phase_candidate_count": self.phase_candidate_count,
            "issue_clip_counts": self.issue_clip_counts,
            "quality_commitment": self.quality_commitment,
            "coordinate_frame": {
                "source_declared": False,
                "ground_estimation": "per_clip_lower_foot_quantile",
                "verified": False,
            },
            "eligibility": {
                "support_phase_discovery": self.phase_candidate_count > 0,
                "mujoco_qualification_candidate": self.phase_candidate_count > 0,
                "mujoco_qualification_authorized": (
                    self.phase_candidate_count > 0
                    and self.license_decision is LicenseDecision.PERMITTED
                ),
                "motion_tracker_training": False,
                "promotion_truth": False,
            },
            "frame_level_trace_persisted": False,
            "training_eligible": False,
            "training_blockers": self.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodeContactBundle:
    relative_path: str
    episode: CanonicalMotionEpisode
    trace: CanonicalContactTrace

    def __post_init__(self) -> None:
        if not self.relative_path:
            raise ValueError("contact bundle relative path must not be empty")
        if self.trace.episode_hash != self.episode.episode_hash:
            raise ValueError("contact bundle trace does not match its episode")


@dataclass(frozen=True)
class MotionDecodeContactBatch:
    report: MotionDecodeContactReport
    bundles: tuple[MotionDecodeContactBundle, ...]

    def __post_init__(self) -> None:
        paths = tuple(bundle.relative_path for bundle in self.bundles)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("contact bundles must be unique and sorted")
        expected = {clip.relative_path for clip in self.report.clips if clip.trace_hash is not None}
        if set(paths) != expected:
            raise ValueError("contact bundles do not match inferred report clips")


def infer_motiondecode_contacts(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    expected_ingest_report_hash: str,
    expected_repair_report_hash: str,
    audit_thresholds: MotionDecodeAuditThresholds | None = None,
    contact_thresholds: ContactInferenceThresholds | None = None,
) -> MotionDecodeContactReport:
    """Return the serializable report while keeping frame traces in memory."""

    return infer_motiondecode_contact_batch(
        registration,
        dataset_root,
        target_model_path=target_model_path,
        expected_ingest_report_hash=expected_ingest_report_hash,
        expected_repair_report_hash=expected_repair_report_hash,
        audit_thresholds=audit_thresholds,
        contact_thresholds=contact_thresholds,
    ).report


def infer_motiondecode_contact_batch(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    expected_ingest_report_hash: str,
    expected_repair_report_hash: str,
    audit_thresholds: MotionDecodeAuditThresholds | None = None,
    contact_thresholds: ContactInferenceThresholds | None = None,
) -> MotionDecodeContactBatch:
    """Replay repaired Q1 clips and infer contact without persisting frame labels."""

    selected_audit = audit_thresholds or MotionDecodeAuditThresholds()
    selected_contact = contact_thresholds or ContactInferenceThresholds()
    repair_report = repair_motiondecode_snapshot(
        registration,
        dataset_root,
        target_model_path=target_model_path,
        expected_ingest_report_hash=expected_ingest_report_hash,
        thresholds=selected_audit,
    )
    if repair_report.report_hash != expected_repair_report_hash:
        raise ValueError("repair report hash does not match replayed repair evidence")
    _, target_body_hash, model_hash = load_g1_joint_contract(target_model_path)
    model, data, qpos_addresses, left_site, right_site = _load_contact_model(
        target_model_path,
        expected_model_hash=model_hash,
    )
    verified = verify_registered_files(registration, dataset_root)
    record_by_path = {
        record.relative_path: record
        for record in registration.manifest.files
        if record.relative_path.startswith("samples/")
    }
    clips: list[MotionDecodeContactAudit] = []
    bundles: list[MotionDecodeContactBundle] = []
    for result in repair_report.results:
        if not result.q1_after:
            clips.append(_upstream_rejected_contact(result))
            continue
        record = record_by_path[result.relative_path]
        episode = _episode_for_contact(
            registration,
            result,
            verified[result.relative_path],
            dataset_root=dataset_root,
            target_model_path=target_model_path,
            target_body_hash=target_body_hash,
            audit_thresholds=selected_audit,
        )
        trace = infer_contact_trace(
            episode,
            model=model,
            data=data,
            qpos_addresses=qpos_addresses,
            left_site_id=left_site,
            right_site_id=right_site,
            target_model_file_hash=model_hash,
            thresholds=selected_contact,
        )
        bundles.append(
            MotionDecodeContactBundle(
                relative_path=result.relative_path,
                episode=episode,
                trace=trace,
            )
        )
        clips.append(
            _audit_contact_trace(
                result,
                trace,
                sample_rate_hz=episode.sample_rate_hz,
                thresholds=selected_contact,
                expected_source_hash=record.content_hash,
            )
        )
    report = MotionDecodeContactReport(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        ingest_report_hash=expected_ingest_report_hash,
        repair_report_hash=repair_report.report_hash,
        target_body_hash=target_body_hash,
        target_model_file_hash=model_hash,
        license_decision=registration.manifest.license_snapshot.decision,
        thresholds=selected_contact,
        clips=tuple(clips),
    )
    return MotionDecodeContactBatch(report=report, bundles=tuple(bundles))


def infer_contact_trace(
    episode: CanonicalMotionEpisode,
    *,
    model: Any,
    data: Any,
    qpos_addresses: np.ndarray,
    left_site_id: int,
    right_site_id: int,
    target_model_file_hash: str,
    thresholds: ContactInferenceThresholds | None = None,
) -> CanonicalContactTrace:
    """Run MuJoCo forward kinematics only; no physics step or actuator command."""

    selected = thresholds or ContactInferenceThresholds()
    frames = int(episode.time.shape[0])
    left: np.ndarray = np.empty((frames, 3), dtype=np.float64)
    right: np.ndarray = np.empty((frames, 3), dtype=np.float64)
    com: np.ndarray = np.empty((frames, 3), dtype=np.float64)
    import mujoco

    for frame in range(frames):
        data.qpos[:3] = episode.root_position[frame]
        quaternion = episode.root_quaternion[frame]
        data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        data.qpos[qpos_addresses] = episode.joint_position[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        left[frame] = data.site_xpos[left_site_id]
        right[frame] = data.site_xpos[right_site_id]
        com[frame] = data.subtree_com[0]
    left[:, 2] -= selected.sole_offset_m
    right[:, 2] -= selected.sole_offset_m
    left_velocity = np.gradient(left, episode.time, axis=0, edge_order=2)
    right_velocity = np.gradient(right, episode.time, axis=0, edge_order=2)
    left_speed = np.linalg.norm(left_velocity, axis=1)
    right_speed = np.linalg.norm(right_velocity, axis=1)
    ground = float(
        np.quantile(
            np.minimum(left[:, 2], right[:, 2]),
            selected.ground_quantile,
        )
    )
    left_height = left[:, 2] - ground
    right_height = right[:, 2] - ground
    left_contact = _contact_hysteresis(left_height, left_speed, selected)
    right_contact = _contact_hysteresis(right_height, right_speed, selected)
    phase: np.ndarray = np.zeros(frames, dtype=np.uint8)
    phase[left_contact & ~right_contact] = _PHASE_CODE[SupportPhase.LEFT]
    phase[~left_contact & right_contact] = _PHASE_CODE[SupportPhase.RIGHT]
    phase[left_contact & right_contact] = _PHASE_CODE[SupportPhase.DOUBLE]
    return CanonicalContactTrace(
        episode_hash=episode.episode_hash,
        target_model_file_hash=target_model_file_hash,
        threshold_hash=selected.threshold_hash,
        ground_height_m=ground,
        left_sole_position=left,
        right_sole_position=right,
        center_of_mass=com,
        left_foot_speed=left_speed,
        right_foot_speed=right_speed,
        left_contact=left_contact,
        right_contact=right_contact,
        phase_code=phase,
    )


def _contact_hysteresis(
    height: np.ndarray,
    speed: np.ndarray,
    thresholds: ContactInferenceThresholds,
) -> np.ndarray:
    contact = np.zeros(height.shape[0], dtype=np.bool_)
    active = False
    for index in range(height.shape[0]):
        if active:
            active = bool(
                height[index] <= thresholds.contact_exit_height_m
                and speed[index] <= thresholds.contact_exit_speed_m_s
            )
        else:
            active = bool(
                height[index] <= thresholds.contact_enter_height_m
                and speed[index] <= thresholds.contact_enter_speed_m_s
            )
        contact[index] = active
    _remove_short_true_runs(contact, thresholds.minimum_contact_frames)
    _fill_short_false_gaps(contact, thresholds.maximum_gap_frames)
    return contact


def _remove_short_true_runs(values: np.ndarray, minimum: int) -> None:
    start: int | None = None
    for index in range(values.shape[0] + 1):
        active = index < values.shape[0] and bool(values[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < minimum:
                values[start:index] = False
            start = None


def _fill_short_false_gaps(values: np.ndarray, maximum: int) -> None:
    start: int | None = None
    for index in range(values.shape[0] + 1):
        inactive = index < values.shape[0] and not bool(values[index])
        if inactive and start is None:
            start = index
        elif not inactive and start is not None:
            if start > 0 and index < values.shape[0] and index - start <= maximum:
                values[start:index] = True
            start = None


def _load_contact_model(
    model_path: Path,
    *,
    expected_model_hash: str,
) -> tuple[Any, Any, np.ndarray, int, int]:
    resolved = model_path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("contact model must be a non-symlink regular file")
    if resolved.stat().st_size > _MAX_MODEL_BYTES:
        raise ValueError("contact model exceeds the 10 MB safety limit")
    actual_hash = "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual_hash != expected_model_hash:
        raise ValueError("contact model hash changed before kinematic inference")
    try:
        import mujoco
    except ImportError as exc:
        raise ValueError("MuJoCo is required for contact forward kinematics") from exc
    model = mujoco.MjModel.from_xml_path(str(resolved))
    data = mujoco.MjData(model)
    addresses: list[int] = []
    for name in G1_DDS_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"contact model is missing target joint: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    if left_site < 0 or right_site < 0:
        raise ValueError("contact model must define left_foot and right_foot sites")
    return (
        model,
        data,
        np.asarray(addresses, dtype=np.int64),
        int(left_site),
        int(right_site),
    )


def _episode_for_contact(
    registration: MotionDecodeRegistration,
    result: MotionDecodeRepairResult,
    source_path: Path,
    *,
    dataset_root: Path,
    target_model_path: Path,
    target_body_hash: str,
    audit_thresholds: MotionDecodeAuditThresholds,
) -> CanonicalMotionEpisode:
    if result.disposition is MotionRepairDisposition.REPAIRED_Q1:
        if result.repair_manifest is None:
            raise ValueError("repaired Q1 result lacks a repair manifest")
        return replay_segmentation_repair(
            registration,
            dataset_root,
            target_model_path=target_model_path,
            manifest=result.repair_manifest,
            thresholds=audit_thresholds,
        )
    if result.disposition is not MotionRepairDisposition.NOT_REQUIRED_Q1:
        raise ValueError("contact inference received a non-Q1 repair result")
    return parse_motion_csv(
        source_path,
        source_manifest_hash=registration.manifest.manifest_hash,
        expected_file_hash=result.source_file_hash,
        target_body_hash=target_body_hash,
        sample_rate_hz=registration.manifest.sample_rate_hz,
    )


def _audit_contact_trace(
    result: MotionDecodeRepairResult,
    trace: CanonicalContactTrace,
    *,
    sample_rate_hz: float,
    thresholds: ContactInferenceThresholds,
    expected_source_hash: str,
) -> MotionDecodeContactAudit:
    if result.source_file_hash != expected_source_hash:
        raise ValueError("contact source hash does not match repair evidence")
    supported = trace.left_contact | trace.right_contact
    near_left = (
        trace.left_sole_position[:, 2] - trace.ground_height_m <= thresholds.contact_enter_height_m
    )
    near_right = (
        trace.right_sole_position[:, 2] - trace.ground_height_m <= thresholds.contact_enter_height_m
    )
    skating = (near_left & ~trace.left_contact) | (near_right & ~trace.right_contact)
    near_speeds = np.concatenate(
        (
            trace.left_foot_speed[near_left],
            trace.right_foot_speed[near_right],
        )
    )
    near_speed_p95 = float(np.quantile(near_speeds, 0.95)) if near_speeds.size else 0.0
    supported_ratio = float(np.mean(supported))
    skating_ratio = float(np.mean(skating))
    maximum_flight_run_s = _maximum_false_run(supported) / sample_rate_hz
    phase_switches = int(np.count_nonzero(np.diff(trace.phase_code) != 0))
    issues = [
        ContactInferenceIssue(
            code="SOURCE_COORDINATE_FRAME_UNSPECIFIED",
            severity=AuditSeverity.WARNING,
            observed=0.0,
            threshold=1.0,
        )
    ]
    if supported_ratio < thresholds.minimum_supported_ratio:
        issues.append(
            ContactInferenceIssue(
                code="SUPPORT_COVERAGE_LOW",
                severity=AuditSeverity.ERROR,
                observed=supported_ratio,
                threshold=thresholds.minimum_supported_ratio,
            )
        )
    if maximum_flight_run_s > thresholds.maximum_flight_run_s:
        issues.append(
            ContactInferenceIssue(
                code="FLIGHT_RUN_TOO_LONG",
                severity=AuditSeverity.ERROR,
                observed=maximum_flight_run_s,
                threshold=thresholds.maximum_flight_run_s,
            )
        )
    if skating_ratio > thresholds.maximum_skating_ratio:
        issues.append(
            ContactInferenceIssue(
                code="FOOT_SKATING_RATIO_HIGH",
                severity=AuditSeverity.ERROR,
                observed=skating_ratio,
                threshold=thresholds.maximum_skating_ratio,
            )
        )
    if near_speed_p95 > thresholds.maximum_near_ground_speed_p95_m_s:
        issues.append(
            ContactInferenceIssue(
                code="NEAR_GROUND_FOOT_SPEED_HIGH",
                severity=AuditSeverity.ERROR,
                observed=near_speed_p95,
                threshold=thresholds.maximum_near_ground_speed_p95_m_s,
            )
        )
    metrics: dict[str, float | int] = {
        "frame_count": int(trace.phase_code.shape[0]),
        "ground_height_m": trace.ground_height_m,
        "supported_ratio": supported_ratio,
        "left_contact_ratio": float(np.mean(trace.left_contact)),
        "right_contact_ratio": float(np.mean(trace.right_contact)),
        "double_support_ratio": float(np.mean(trace.left_contact & trace.right_contact)),
        "flight_ratio": float(np.mean(~supported)),
        "maximum_flight_run_s": maximum_flight_run_s,
        "skating_ratio": skating_ratio,
        "near_ground_foot_speed_p95_m_s": near_speed_p95,
        "phase_switch_count": phase_switches,
        "center_of_mass_height_min_m": float(np.min(trace.center_of_mass[:, 2])),
        "center_of_mass_height_max_m": float(np.max(trace.center_of_mass[:, 2])),
    }
    return MotionDecodeContactAudit(
        relative_path=result.relative_path,
        source_file_hash=result.source_file_hash,
        repair_manifest_hash=(
            result.repair_manifest.manifest_hash if result.repair_manifest is not None else None
        ),
        episode_hash=trace.episode_hash,
        trace_hash=trace.trace_hash,
        metrics=metrics,
        issues=tuple(issues),
        upstream_q1=True,
    )


def _upstream_rejected_contact(
    result: MotionDecodeRepairResult,
) -> MotionDecodeContactAudit:
    return MotionDecodeContactAudit(
        relative_path=result.relative_path,
        source_file_hash=result.source_file_hash,
        repair_manifest_hash=None,
        episode_hash=None,
        trace_hash=None,
        metrics=None,
        issues=(
            ContactInferenceIssue(
                code="UPSTREAM_Q1_REQUIRED",
                severity=AuditSeverity.ERROR,
                observed=0.0,
                threshold=1.0,
            ),
        ),
        upstream_q1=False,
    )


def _maximum_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = 0 if bool(value) else current + 1
        longest = max(longest, current)
    return longest


__all__ = [
    "CanonicalContactTrace",
    "ContactInferenceIssue",
    "ContactInferenceThresholds",
    "MotionDecodeContactAudit",
    "MotionDecodeContactBatch",
    "MotionDecodeContactBundle",
    "MotionDecodeContactReport",
    "SupportPhase",
    "infer_contact_trace",
    "infer_motiondecode_contact_batch",
    "infer_motiondecode_contacts",
]
