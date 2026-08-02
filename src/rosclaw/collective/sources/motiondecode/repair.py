"""Content-addressed, dry-run repair of terminal MotionDecode resets."""

from __future__ import annotations

import csv
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
    MotionDecodeClipAudit,
    MotionQualificationLevel,
    TerminalResetDetection,
    audit_canonical_episode,
    audit_motiondecode_snapshot,
    detect_terminal_resets,
    load_g1_joint_contract,
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

_DETECTOR_VERSION = "terminal_reset_tail_trim_v1"
_REPAIRABLE_ERROR_CODES = frozenset(
    {
        "ROOT_LOOP_CLOSURE_DISCONTINUITY",
        "JOINT_LOOP_CLOSURE_DISCONTINUITY",
        "ROOT_STEP_ERROR",
    }
)
_SEMANTIC_INVARIANTS = (
    "motion_family",
    "key_pose_prefix",
    "left_right_joint_semantics",
    "source_attribution",
    "source_revision",
)


class MotionRepairDisposition(StrEnum):
    NOT_REQUIRED_Q1 = "not_required_q1"
    REPAIRED_Q1 = "repaired_q1"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SegmentationRepairManifest:
    """A bounded plan that trims only an appended reset and its tail."""

    registration_hash: str
    source_manifest_hash: str
    source_file_hash: str
    relative_path: str
    motion_family: str
    target_body_hash: str
    target_model_file_hash: str
    before_audit_hash: str
    detector_hash: str
    detection: TerminalResetDetection
    original_frame_count: int
    retained_frame_count: int
    removed_frame_count: int
    sample_rate_hz: float
    operation: str = "trim_tail_before_terminal_reset"
    semantic_invariants: tuple[str, ...] = _SEMANTIC_INVARIANTS
    schema_version: str = "rosclaw.collective.motiondecode_segmentation_repair.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("registration_hash", self.registration_hash),
            ("source_manifest_hash", self.source_manifest_hash),
            ("source_file_hash", self.source_file_hash),
            ("target_body_hash", self.target_body_hash),
            ("target_model_file_hash", self.target_model_file_hash),
            ("before_audit_hash", self.before_audit_hash),
            ("detector_hash", self.detector_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not self.relative_path or not self.motion_family:
            raise ValueError("repair source identity must not be empty")
        if self.operation != "trim_tail_before_terminal_reset":
            raise ValueError("only terminal tail trimming is supported")
        if self.original_frame_count < 3 or self.retained_frame_count < 3:
            raise ValueError("repair must retain at least three frames")
        if self.removed_frame_count <= 0:
            raise ValueError("repair must remove at least one frame")
        if self.retained_frame_count + self.removed_frame_count != self.original_frame_count:
            raise ValueError("repair frame accounting is inconsistent")
        if self.detection.reset_frame != self.retained_frame_count:
            raise ValueError("repair boundary does not match the detected reset")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("repair sample rate must be finite and positive")
        if tuple(self.semantic_invariants) != _SEMANTIC_INVARIANTS:
            raise ValueError("repair semantic invariants cannot be weakened")

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_hash": self.registration_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "source_file_hash": self.source_file_hash,
            "relative_path": self.relative_path,
            "motion_family": self.motion_family,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "before_audit_hash": self.before_audit_hash,
            "detector_hash": self.detector_hash,
            "detection": self.detection.to_dict(),
            "operation": self.operation,
            "original_frame_count": self.original_frame_count,
            "retained_frame_range": [0, self.retained_frame_count - 1],
            "retained_frame_count": self.retained_frame_count,
            "removed_frame_count": self.removed_frame_count,
            "sample_rate_hz": self.sample_rate_hz,
            "semantic_invariants": list(self.semantic_invariants),
            "raw_motion_embedded": False,
            "hardware_authorized": False,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SegmentationRepairManifest:
        if value.get("schema_version") != cls.schema_version:
            raise ValueError("segmentation repair schema version is unsupported")
        if value.get("raw_motion_embedded") is not False:
            raise ValueError("segmentation repair cannot embed raw motion")
        if value.get("hardware_authorized") is not False:
            raise ValueError("segmentation repair cannot authorize hardware")
        detection_value = value.get("detection")
        invariants_value = value.get("semantic_invariants")
        retained_range = value.get("retained_frame_range")
        if not isinstance(detection_value, dict):
            raise ValueError("segmentation repair lacks a reset detection")
        if not isinstance(invariants_value, list):
            raise ValueError("segmentation repair semantic invariants must be an array")
        manifest = cls(
            registration_hash=str(value["registration_hash"]),
            source_manifest_hash=str(value["source_manifest_hash"]),
            source_file_hash=str(value["source_file_hash"]),
            relative_path=str(value["relative_path"]),
            motion_family=str(value["motion_family"]),
            target_body_hash=str(value["target_body_hash"]),
            target_model_file_hash=str(value["target_model_file_hash"]),
            before_audit_hash=str(value["before_audit_hash"]),
            detector_hash=str(value["detector_hash"]),
            detection=TerminalResetDetection.from_dict(detection_value),
            original_frame_count=int(value["original_frame_count"]),
            retained_frame_count=int(value["retained_frame_count"]),
            removed_frame_count=int(value["removed_frame_count"]),
            sample_rate_hz=float(value["sample_rate_hz"]),
            operation=str(value["operation"]),
            semantic_invariants=tuple(str(item) for item in invariants_value),
        )
        if retained_range != [0, manifest.retained_frame_count - 1]:
            raise ValueError("segmentation repair retained frame range is inconsistent")
        return manifest


@dataclass(frozen=True)
class MotionDecodeRepairResult:
    relative_path: str
    source_file_hash: str
    disposition: MotionRepairDisposition
    reason_codes: tuple[str, ...]
    before_audit: MotionDecodeClipAudit
    repair_manifest: SegmentationRepairManifest | None = None
    after_audit: MotionDecodeClipAudit | None = None
    schema_version: str = "rosclaw.collective.motiondecode_repair_result.v1"

    def __post_init__(self) -> None:
        if not self.source_file_hash.startswith("sha256:") or len(self.source_file_hash) != 71:
            raise ValueError("repair result source hash must be a sha256: content hash")
        if not isinstance(self.disposition, MotionRepairDisposition):
            raise ValueError("repair result disposition is unknown")
        if self.relative_path != self.before_audit.relative_path:
            raise ValueError("repair result path does not match its before audit")
        if self.source_file_hash != self.before_audit.source_file_hash:
            raise ValueError("repair result source hash does not match its before audit")
        if not self.reason_codes or any(not item for item in self.reason_codes):
            raise ValueError("repair result requires non-empty reason codes")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("repair result reason codes must be unique")
        if (self.repair_manifest is None) != (self.after_audit is None):
            raise ValueError("repair manifest and after audit must be present together")
        if self.repair_manifest is not None and (
            self.repair_manifest.relative_path != self.relative_path
            or self.repair_manifest.source_file_hash != self.source_file_hash
        ):
            raise ValueError("repair manifest does not match its result source")
        if self.after_audit is not None and (
            self.after_audit.relative_path != self.relative_path
            or self.after_audit.source_file_hash != self.source_file_hash
        ):
            raise ValueError("after audit does not match its result source")
        if self.disposition is MotionRepairDisposition.REPAIRED_Q1:
            if self.repair_manifest is None or self.after_audit is None:
                raise ValueError("a repaired result requires a manifest and after audit")
            if not self.after_audit.kinematic_valid:
                raise ValueError("a repaired Q1 result must pass the after audit")
            summary = self.after_audit.episode_summary
            if (
                summary is None
                or summary.get("derivation_manifest_hash") != self.repair_manifest.manifest_hash
            ):
                raise ValueError("repaired episode is not bound to its repair manifest")
        if self.disposition is MotionRepairDisposition.NOT_REQUIRED_Q1:
            if not self.before_audit.kinematic_valid:
                raise ValueError("not-required disposition requires an existing Q1 clip")
            if self.repair_manifest is not None or self.after_audit is not None:
                raise ValueError("an existing Q1 clip must not have a repair")

    @property
    def result_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def q1_after(self) -> bool:
        return self.disposition in {
            MotionRepairDisposition.NOT_REQUIRED_Q1,
            MotionRepairDisposition.REPAIRED_Q1,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "source_file_hash": self.source_file_hash,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "before_audit": self.before_audit.to_dict(),
            "before_audit_hash": self.before_audit.audit_hash,
            "repair_manifest": (
                self.repair_manifest.to_dict() if self.repair_manifest is not None else None
            ),
            "repair_manifest_hash": (
                self.repair_manifest.manifest_hash if self.repair_manifest is not None else None
            ),
            "after_audit": self.after_audit.to_dict() if self.after_audit is not None else None,
            "after_audit_hash": (
                self.after_audit.audit_hash if self.after_audit is not None else None
            ),
            "q1_after": self.q1_after,
            "raw_motion_persisted": False,
            "training_eligible": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class MotionDecodeRepairReport:
    registration_hash: str
    source_manifest_hash: str
    original_ingest_report_hash: str
    target_body_hash: str
    target_model_file_hash: str
    license_decision: LicenseDecision
    thresholds: MotionDecodeAuditThresholds
    detector_hash: str
    results: tuple[MotionDecodeRepairResult, ...]
    schema_version: str = "rosclaw.collective.motiondecode_repair_report.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("registration_hash", self.registration_hash),
            ("source_manifest_hash", self.source_manifest_hash),
            ("original_ingest_report_hash", self.original_ingest_report_hash),
            ("target_body_hash", self.target_body_hash),
            ("target_model_file_hash", self.target_model_file_hash),
            ("detector_hash", self.detector_hash),
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} must be a sha256: content hash")
        if not isinstance(self.license_decision, LicenseDecision):
            raise ValueError("repair report license decision is unknown")
        if not isinstance(self.thresholds, MotionDecodeAuditThresholds):
            raise ValueError("repair report thresholds are invalid")
        if self.detector_hash != _detector_hash(self.thresholds):
            raise ValueError("repair report detector hash does not replay")
        results = tuple(self.results)
        if any(not isinstance(item, MotionDecodeRepairResult) for item in results):
            raise ValueError("repair report contains an invalid result")
        paths = tuple(item.relative_path for item in results)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("repair report results must be unique and sorted")
        for result in results:
            manifest = result.repair_manifest
            if manifest is None:
                continue
            if (
                manifest.registration_hash != self.registration_hash
                or manifest.source_manifest_hash != self.source_manifest_hash
                or manifest.target_body_hash != self.target_body_hash
                or manifest.target_model_file_hash != self.target_model_file_hash
                or manifest.detector_hash != self.detector_hash
            ):
                raise ValueError("repair manifest lineage does not match its report")
        object.__setattr__(self, "results", results)

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def disposition_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(result.disposition.value for result in self.results).items()))

    @property
    def repaired_q1_count(self) -> int:
        return sum(
            result.disposition is MotionRepairDisposition.REPAIRED_Q1 for result in self.results
        )

    @property
    def q1_after_count(self) -> int:
        return sum(result.q1_after for result in self.results)

    @property
    def rejected_count(self) -> int:
        return sum(
            result.disposition is MotionRepairDisposition.REJECTED for result in self.results
        )

    @property
    def reason_clip_counts(self) -> dict[str, int]:
        counts = Counter(reason for result in self.results for reason in result.reason_codes)
        return dict(sorted(counts.items()))

    @property
    def quality_commitment(self) -> str:
        return canonical_hash(
            {
                "schema_version": "rosclaw.collective.motiondecode_repair_quality.v1",
                "original_ingest_report_hash": self.original_ingest_report_hash,
                "detector_hash": self.detector_hash,
                "result_hashes": [result.result_hash for result in self.results],
            }
        )

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
            "REPAIRED_MOTION_NOT_PERSISTED",
        ]
        if self.q1_after_count != len(self.results):
            blockers.insert(0, "KINEMATIC_REPAIR_PARTIAL")
        if self.license_decision is not LicenseDecision.PERMITTED:
            blockers.insert(0, "LICENSE_NOT_PERMITTED")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_hash": self.registration_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "original_ingest_report_hash": self.original_ingest_report_hash,
            "target_body_hash": self.target_body_hash,
            "target_model_file_hash": self.target_model_file_hash,
            "license_decision": self.license_decision.value,
            "thresholds": self.thresholds.to_dict(),
            "detector_version": _DETECTOR_VERSION,
            "detector_hash": self.detector_hash,
            "results": [result.to_dict() for result in self.results],
            "clip_count": len(self.results),
            "disposition_counts": self.disposition_counts,
            "repaired_q1_count": self.repaired_q1_count,
            "q1_after_count": self.q1_after_count,
            "rejected_count": self.rejected_count,
            "reason_clip_counts": self.reason_clip_counts,
            "quality_commitment": self.quality_commitment,
            "dry_run_only": True,
            "raw_motion_persisted": False,
            "eligibility": {
                "q1_motion_reference": self.q1_after_count > 0,
                "mujoco_qualification_candidate": self.q1_after_count > 0,
                "mujoco_qualification_authorized": (
                    self.q1_after_count > 0 and self.license_decision is LicenseDecision.PERMITTED
                ),
                "motion_tracker_training": False,
                "behavior_cloning": False,
                "offline_rl": False,
                "promotion_truth": False,
            },
            "training_eligible": self.training_eligible,
            "training_blockers": self.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


def repair_motiondecode_snapshot(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    expected_ingest_report_hash: str,
    thresholds: MotionDecodeAuditThresholds | None = None,
) -> MotionDecodeRepairReport:
    """Plan terminal trims in memory and re-run Q1 audit on retained frames."""

    selected_thresholds = thresholds or MotionDecodeAuditThresholds()
    before_report = audit_motiondecode_snapshot(
        registration,
        dataset_root,
        target_model_path=target_model_path,
        thresholds=selected_thresholds,
    )
    if before_report.report_hash != expected_ingest_report_hash:
        raise ValueError("ingest report hash does not match the replayed source audit")
    joint_limits, target_body_hash, model_hash = load_g1_joint_contract(target_model_path)
    if (
        before_report.target_body_hash != target_body_hash
        or before_report.target_model_file_hash != model_hash
    ):
        raise ValueError("target model contract changed during repair audit")
    verified = verify_registered_files(registration, dataset_root)
    before_by_path = {clip.relative_path: clip for clip in before_report.clips}
    detector_hash = _detector_hash(selected_thresholds)
    results: list[MotionDecodeRepairResult] = []
    for record in registration.manifest.files:
        if not record.relative_path.startswith("samples/"):
            continue
        results.append(
            _repair_record(
                registration,
                record,
                verified[record.relative_path],
                before=before_by_path[record.relative_path],
                target_body_hash=target_body_hash,
                target_model_file_hash=model_hash,
                joint_limits=joint_limits,
                thresholds=selected_thresholds,
                detector_hash=detector_hash,
            )
        )
    return MotionDecodeRepairReport(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        original_ingest_report_hash=before_report.report_hash,
        target_body_hash=target_body_hash,
        target_model_file_hash=model_hash,
        license_decision=registration.manifest.license_snapshot.decision,
        thresholds=selected_thresholds,
        detector_hash=detector_hash,
        results=tuple(results),
    )


def replay_segmentation_repair(
    registration: MotionDecodeRegistration,
    dataset_root: Path,
    *,
    target_model_path: Path,
    manifest: SegmentationRepairManifest,
    thresholds: MotionDecodeAuditThresholds | None = None,
) -> CanonicalMotionEpisode:
    """Reconstruct one repaired prefix from immutable source and manifest evidence."""

    selected_thresholds = thresholds or MotionDecodeAuditThresholds()
    if (
        manifest.registration_hash != registration.registration_hash
        or manifest.source_manifest_hash != registration.manifest.manifest_hash
    ):
        raise ValueError("repair manifest source lineage does not match registration")
    if manifest.detector_hash != _detector_hash(selected_thresholds):
        raise ValueError("repair manifest detector does not match audit thresholds")
    joint_limits, target_body_hash, model_hash = load_g1_joint_contract(target_model_path)
    if (
        manifest.target_body_hash != target_body_hash
        or manifest.target_model_file_hash != model_hash
    ):
        raise ValueError("repair manifest target model lineage does not replay")
    records = {
        item.relative_path: item
        for item in registration.manifest.files
        if item.relative_path.startswith("samples/")
    }
    record = records.get(manifest.relative_path)
    if record is None or record.content_hash != manifest.source_file_hash:
        raise ValueError("repair manifest source file does not replay")
    verified = verify_registered_files(registration, dataset_root)
    episode = parse_motion_csv(
        verified[record.relative_path],
        source_manifest_hash=registration.manifest.manifest_hash,
        expected_file_hash=record.content_hash,
        target_body_hash=target_body_hash,
        sample_rate_hz=registration.manifest.sample_rate_hz,
    )
    before_issues = tuple(audit_canonical_episode(episode, joint_limits, selected_thresholds))
    before = MotionDecodeClipAudit(
        relative_path=record.relative_path,
        source_file_hash=record.content_hash,
        qualification=(
            MotionQualificationLevel.Q0_INVALID
            if any(issue.severity is AuditSeverity.ERROR for issue in before_issues)
            else MotionQualificationLevel.Q1_KINEMATIC_ONLY
        ),
        issues=before_issues,
        episode_summary=episode.summary(),
    )
    error_codes = {issue.code for issue in before.issues if issue.severity is AuditSeverity.ERROR}
    unsupported = sorted(error_codes - _REPAIRABLE_ERROR_CODES)
    if before.kinematic_valid or unsupported:
        raise ValueError("repair manifest source is not an eligible segmentation failure")
    if not episode.implicit_timeline:
        raise ValueError("repair manifest cannot rewrite an explicit timeline")
    detections = detect_terminal_resets(episode, selected_thresholds)
    if len(detections) != 1 or detections[0] != manifest.detection:
        raise ValueError("repair manifest reset detection does not replay")
    if not _root_errors_are_at_reset(episode, selected_thresholds, detections[0]):
        raise ValueError("repair manifest cannot hide a root error outside its boundary")
    expected = SegmentationRepairManifest(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        source_file_hash=record.content_hash,
        relative_path=record.relative_path,
        motion_family=record.family.value,
        target_body_hash=target_body_hash,
        target_model_file_hash=model_hash,
        before_audit_hash=before.audit_hash,
        detector_hash=manifest.detector_hash,
        detection=detections[0],
        original_frame_count=int(episode.time.shape[0]),
        retained_frame_count=detections[0].reset_frame,
        removed_frame_count=int(episode.time.shape[0]) - detections[0].reset_frame,
        sample_rate_hz=episode.sample_rate_hz,
    )
    if expected.manifest_hash != manifest.manifest_hash:
        raise ValueError("repair manifest content does not replay")
    repaired = _apply_tail_trim(episode, manifest)
    after_issues = audit_canonical_episode(repaired, joint_limits, selected_thresholds)
    if any(issue.severity is AuditSeverity.ERROR for issue in after_issues):
        raise ValueError("replayed repair does not pass the Q1 audit")
    return repaired


def _repair_record(
    registration: MotionDecodeRegistration,
    record: MotionDecodeFileRecord,
    path: Path,
    *,
    before: MotionDecodeClipAudit,
    target_body_hash: str,
    target_model_file_hash: str,
    joint_limits: dict[str, tuple[float, float]],
    thresholds: MotionDecodeAuditThresholds,
    detector_hash: str,
) -> MotionDecodeRepairResult:
    if before.kinematic_valid:
        return MotionDecodeRepairResult(
            relative_path=record.relative_path,
            source_file_hash=record.content_hash,
            disposition=MotionRepairDisposition.NOT_REQUIRED_Q1,
            reason_codes=("ALREADY_Q1",),
            before_audit=before,
        )
    if before.parser_error is not None:
        return _rejected(record, before, "SOURCE_PARSE_FAILED")
    error_codes = {issue.code for issue in before.issues if issue.severity is AuditSeverity.ERROR}
    unsupported = sorted(error_codes - _REPAIRABLE_ERROR_CODES)
    if unsupported:
        return _rejected(
            record,
            before,
            *(f"UNREPAIRABLE_{code}" for code in unsupported),
        )
    try:
        episode = parse_motion_csv(
            path,
            source_manifest_hash=registration.manifest.manifest_hash,
            expected_file_hash=record.content_hash,
            target_body_hash=target_body_hash,
            sample_rate_hz=registration.manifest.sample_rate_hz,
        )
    except (OSError, ValueError, csv.Error):
        return _rejected(record, before, "SOURCE_CHANGED_OR_PARSE_FAILED")
    if not episode.implicit_timeline:
        return _rejected(record, before, "EXPLICIT_TIMELINE_REPAIR_UNSUPPORTED")
    detections = detect_terminal_resets(episode, thresholds)
    if not detections:
        return _rejected(record, before, "NO_TERMINAL_RESET")
    if len(detections) != 1:
        return _rejected(record, before, "AMBIGUOUS_TERMINAL_RESETS")
    detection = detections[0]
    if detection.reset_frame < 3:
        return _rejected(record, before, "INSUFFICIENT_RETAINED_FRAMES")
    if not _root_errors_are_at_reset(episode, thresholds, detection):
        return _rejected(record, before, "ROOT_STEP_OUTSIDE_RESET_BOUNDARY")
    manifest = SegmentationRepairManifest(
        registration_hash=registration.registration_hash,
        source_manifest_hash=registration.manifest.manifest_hash,
        source_file_hash=record.content_hash,
        relative_path=record.relative_path,
        motion_family=record.family.value,
        target_body_hash=target_body_hash,
        target_model_file_hash=target_model_file_hash,
        before_audit_hash=before.audit_hash,
        detector_hash=detector_hash,
        detection=detection,
        original_frame_count=int(episode.time.shape[0]),
        retained_frame_count=detection.reset_frame,
        removed_frame_count=int(episode.time.shape[0]) - detection.reset_frame,
        sample_rate_hz=episode.sample_rate_hz,
    )
    repaired = _apply_tail_trim(episode, manifest)
    issues = tuple(audit_canonical_episode(repaired, joint_limits, thresholds))
    qualification = (
        MotionQualificationLevel.Q0_INVALID
        if any(issue.severity is AuditSeverity.ERROR for issue in issues)
        else MotionQualificationLevel.Q1_KINEMATIC_ONLY
    )
    after = MotionDecodeClipAudit(
        relative_path=record.relative_path,
        source_file_hash=record.content_hash,
        qualification=qualification,
        issues=issues,
        episode_summary=repaired.summary(),
    )
    if not after.kinematic_valid:
        return MotionDecodeRepairResult(
            relative_path=record.relative_path,
            source_file_hash=record.content_hash,
            disposition=MotionRepairDisposition.REJECTED,
            reason_codes=("AFTER_REPAIR_Q0",),
            before_audit=before,
            repair_manifest=manifest,
            after_audit=after,
        )
    return MotionDecodeRepairResult(
        relative_path=record.relative_path,
        source_file_hash=record.content_hash,
        disposition=MotionRepairDisposition.REPAIRED_Q1,
        reason_codes=("TERMINAL_RESET_TAIL_TRIMMED",),
        before_audit=before,
        repair_manifest=manifest,
        after_audit=after,
    )


def _apply_tail_trim(
    episode: CanonicalMotionEpisode,
    manifest: SegmentationRepairManifest,
) -> CanonicalMotionEpisode:
    stop = manifest.retained_frame_count
    time = np.arange(stop, dtype=np.float64) / episode.sample_rate_hz
    joint_position = episode.joint_position[:stop].copy()
    joint_velocity = np.gradient(joint_position, time, axis=0, edge_order=2)
    joint_acceleration = np.gradient(joint_velocity, time, axis=0, edge_order=2)
    return CanonicalMotionEpisode(
        source_manifest_hash=episode.source_manifest_hash,
        source_file_hash=episode.source_file_hash,
        target_body_hash=episode.target_body_hash,
        mapping=episode.mapping,
        sample_rate_hz=episode.sample_rate_hz,
        time=time,
        root_position=episode.root_position[:stop].copy(),
        root_quaternion=episode.root_quaternion[:stop].copy(),
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_acceleration=joint_acceleration,
        implicit_timeline=True,
        derivation_manifest_hash=manifest.manifest_hash,
        ball_pose_available=episode.ball_pose_available,
        action_semantics_available=episode.action_semantics_available,
        reward_semantics_available=episode.reward_semantics_available,
        transition_semantics_available=episode.transition_semantics_available,
    )


def _root_errors_are_at_reset(
    episode: CanonicalMotionEpisode,
    thresholds: MotionDecodeAuditThresholds,
    detection: TerminalResetDetection,
) -> bool:
    root_steps = np.linalg.norm(np.diff(episode.root_position, axis=0), axis=1)
    error_indices = {
        int(item) for item in np.flatnonzero(root_steps > thresholds.root_step_error_m)
    }
    return not error_indices or error_indices == {detection.transition_frame}


def _rejected(
    record: MotionDecodeFileRecord,
    before: MotionDecodeClipAudit,
    *reason_codes: str,
) -> MotionDecodeRepairResult:
    return MotionDecodeRepairResult(
        relative_path=record.relative_path,
        source_file_hash=record.content_hash,
        disposition=MotionRepairDisposition.REJECTED,
        reason_codes=tuple(reason_codes),
        before_audit=before,
    )


def _detector_hash(thresholds: MotionDecodeAuditThresholds) -> str:
    return canonical_hash(
        {
            "schema_version": "rosclaw.collective.motiondecode_repair_detector.v1",
            "detector_version": _DETECTOR_VERSION,
            "thresholds": thresholds.to_dict(),
            "supported_operation": "trim_tail_before_terminal_reset",
            "maximum_tail_frames": thresholds.loop_closure_search_frames,
        }
    )


def clean_motiondecode_spans(
    episode: CanonicalMotionEpisode,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    minimum_frames: int = 32,
) -> tuple[tuple[int, int], ...]:
    """Return half-open, physically continuous spans for representation learning.

    This does not repair or relabel frames.  It excludes reset boundaries,
    non-unit quaternions, joint-limit excursions, and implausible derivatives,
    so a long clip can contribute clean windows without teaching a learner its
    capture discontinuities.
    """

    frame_count = int(episode.time.shape[0])
    if minimum_frames < 2:
        raise ValueError("clean MotionDecode span must contain at least two frames")
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.shape != (episode.joint_position.shape[1],):
        raise ValueError("clean MotionDecode span joint limits do not match the episode")
    quaternion_error = np.abs(np.linalg.norm(episode.root_quaternion, axis=1) - 1.0)
    in_limits = np.all(
        (episode.joint_position >= lower - 1e-4)
        & (episode.joint_position <= upper + 1e-4),
        axis=1,
    )
    frame_valid = (quaternion_error <= 0.02) & in_limits
    root_step_speed = (
        np.linalg.norm(np.diff(episode.root_position, axis=0), axis=1) * episode.sample_rate_hz
    )
    joint_step_speed = (
        np.max(np.abs(np.diff(episode.joint_position, axis=0)), axis=1)
        * episode.sample_rate_hz
    )
    edge_valid = (root_step_speed <= 8.0) & (joint_step_speed <= 25.0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(frame_count):
        valid = bool(frame_valid[frame]) and (frame == 0 or bool(edge_valid[frame - 1]))
        if valid and start is None:
            start = frame
        if not valid and start is not None:
            if frame - start >= minimum_frames:
                spans.append((start, frame))
            start = None
    if start is not None and frame_count - start >= minimum_frames:
        spans.append((start, frame_count))
    return tuple(spans)


__all__ = [
    "MotionDecodeRepairReport",
    "MotionDecodeRepairResult",
    "MotionRepairDisposition",
    "SegmentationRepairManifest",
    "clean_motiondecode_spans",
    "repair_motiondecode_snapshot",
    "replay_segmentation_repair",
]
