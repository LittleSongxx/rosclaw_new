from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rosclaw.collective.cli import dispatch_collective_argv
from rosclaw.collective.contracts import LicenseDecision, LicenseUse
from rosclaw.collective.sources.motiondecode.audit import (
    MotionQualificationLevel,
    audit_motiondecode_snapshot,
)
from rosclaw.collective.sources.motiondecode.contact import (
    ContactInferenceThresholds,
    infer_motiondecode_contact_batch,
    infer_motiondecode_contacts,
)
from rosclaw.collective.sources.motiondecode.joint_mapping import (
    MOTIONDECODE_ROOT_COLUMNS,
)
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeRegistration,
    register_motiondecode_source,
)
from rosclaw.collective.sources.motiondecode.qualification import (
    PhysicsClipStatus,
    qualify_canonical_motion,
    qualify_motiondecode_snapshot,
)
from rosclaw.collective.sources.motiondecode.repair import (
    repair_motiondecode_snapshot,
)
from rosclaw.collective.sources.motiondecode.taxonomy import (
    MOTIONDECODE_INDEX_COLUMNS,
    MotionFamily,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

ROOT = Path(__file__).parents[2]
G1_MODEL = ROOT / "e-urdf-zoo/g1/robot.mjcf.xml"
G1_SCENE = ROOT / "e-urdf-zoo/g1/scene.xml"
REVISION = "c460808e801b35c683eb4cf4c8338ce61481a9bb"


def _dataset(
    root: Path,
    *,
    root_step_m: float = 0.0,
    frames: int = 12,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    license_payload: bytes = b"",
) -> None:
    metadata = root / "metadata"
    sample_dir = root / "samples/3.3.Ball_Game_Interaction/3.3.1.Football"
    metadata.mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    with (metadata / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MOTIONDECODE_INDEX_COLUMNS)
        writer.writerow(
            (
                "3.3.1",
                "Ball_Game_Interaction",
                "Football",
                "Instep_Kick",
                "csv",
                "Unitree_G1",
            )
        )
    (root / "LICENSE").write_bytes(license_payload)
    header = MOTIONDECODE_ROOT_COLUMNS + tuple(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES)
    with (sample_dir / "contact.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame in range(frames):
            writer.writerow(
                [
                    root_step_m * frame,
                    0.0,
                    0.793,
                    *quaternion,
                    *([0.0] * len(G1_DDS_JOINT_NAMES)),
                ]
            )


def _registration(
    root: Path,
    *,
    license_decision: LicenseDecision = LicenseDecision.PENDING,
) -> MotionDecodeRegistration:
    return register_motiondecode_source(
        root,
        revision=REVISION,
        requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
        license_decision=license_decision,
        terms_path=root / "LICENSE",
        terms_uri=(f"https://huggingface.co/datasets/CMRobot/MotionDecode/blob/{REVISION}/LICENSE"),
        families=(MotionFamily.FOOTBALL,),
        limit=40,
    )


def _reports(
    root: Path,
    *,
    license_decision: LicenseDecision = LicenseDecision.PENDING,
) -> tuple[MotionDecodeRegistration, object, object]:
    registration = _registration(root, license_decision=license_decision)
    ingest = audit_motiondecode_snapshot(
        registration,
        root,
        target_model_path=G1_MODEL,
    )
    repair = repair_motiondecode_snapshot(
        registration,
        root,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
    )
    return registration, ingest, repair


def test_static_g1_reference_yields_deterministic_double_support(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    registration, ingest, repair = _reports(tmp_path)

    first = infer_motiondecode_contacts(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )
    second = infer_motiondecode_contacts(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )

    assert first.report_hash == second.report_hash
    assert first.inferred_count == 1
    assert first.phase_candidate_count == 1
    clip = first.clips[0]
    assert clip.trace_hash is not None
    assert clip.metrics is not None
    assert clip.metrics["supported_ratio"] == pytest.approx(1.0)
    assert clip.metrics["double_support_ratio"] == pytest.approx(1.0)
    assert clip.metrics["maximum_flight_run_s"] == 0.0
    assert clip.phase_segmentation_candidate is True
    result = first.to_dict()
    assert result["frame_level_trace_persisted"] is False
    assert result["eligibility"]["mujoco_qualification_authorized"] is False
    assert result["training_eligible"] is False
    assert "LICENSE_NOT_PERMITTED" in result["training_blockers"]


def test_fast_near_ground_slide_is_preserved_as_a_contact_counterexample(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path, root_step_m=0.01)
    registration, ingest, repair = _reports(tmp_path)

    report = infer_motiondecode_contacts(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )

    assert report.inferred_count == 1
    assert report.phase_candidate_count == 0
    codes = {issue.code for issue in report.clips[0].issues}
    assert "SUPPORT_COVERAGE_LOW" in codes
    assert "FOOT_SKATING_RATIO_HIGH" in codes
    assert "NEAR_GROUND_FOOT_SPEED_HIGH" in codes


def test_contact_threshold_contract_rejects_weakened_hysteresis() -> None:
    with pytest.raises(ValueError, match="enter height"):
        ContactInferenceThresholds(
            contact_enter_height_m=0.05,
            contact_exit_height_m=0.04,
        )
    with pytest.raises(ValueError, match="frame thresholds"):
        ContactInferenceThresholds(minimum_contact_frames=1.5)  # type: ignore[arg-type]


def test_contact_replays_the_exact_repair_commitment(tmp_path: Path) -> None:
    _dataset(tmp_path)
    registration, ingest, _ = _reports(tmp_path)

    with pytest.raises(ValueError, match="repair report hash"):
        infer_motiondecode_contacts(
            registration,
            tmp_path,
            target_model_path=G1_MODEL,
            expected_ingest_report_hash=ingest.report_hash,
            expected_repair_report_hash="sha256:" + "0" * 64,
        )


def test_pending_license_blocks_physics_before_any_mujoco_step(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path, frames=240)
    registration, ingest, repair = _reports(tmp_path)
    contact = infer_motiondecode_contacts(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )

    report = qualify_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        scene_path=G1_SCENE,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
        expected_contact_report_hash=contact.report_hash,
    )

    assert report.physics_executed_count == 0
    assert report.physics_step_count == 0
    assert report.q3_count == 0
    assert report.clips[0].status is PhysicsClipStatus.BLOCKED_LICENSE
    assert report.clips[0].blocker_codes == ("LICENSE_NOT_PERMITTED",)


def test_permitted_static_reference_advances_physics_and_reaches_q3(
    tmp_path: Path,
) -> None:
    _dataset(
        tmp_path,
        frames=240,
        license_payload=b"explicit synthetic research permission",
    )
    registration, ingest, repair = _reports(
        tmp_path,
        license_decision=LicenseDecision.PERMITTED,
    )
    contact = infer_motiondecode_contacts(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )

    report = qualify_motiondecode_snapshot(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        scene_path=G1_SCENE,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
        expected_contact_report_hash=contact.report_hash,
    )

    assert report.physics_executed_count == 1
    assert report.physics_step_count >= 900
    assert report.q3_count == 1
    result = report.clips[0].result
    assert result is not None
    assert result.qualification is MotionQualificationLevel.Q3_PHYSICS_TRACKABLE
    assert result.metrics["finite"] is True
    assert result.metrics["minimum_pelvis_height_m"] > 0.75
    assert result.metrics["nonfoot_floor_contact_steps"] == 0


def test_sideways_reference_is_a_hard_q1_physics_failure(tmp_path: Path) -> None:
    _dataset(
        tmp_path,
        frames=240,
        quaternion=(2**-0.5, 0.0, 2**-0.5, 0.0),
        license_payload=b"explicit synthetic research permission",
    )
    registration, ingest, repair = _reports(
        tmp_path,
        license_decision=LicenseDecision.PERMITTED,
    )
    batch = infer_motiondecode_contact_batch(
        registration,
        tmp_path,
        target_model_path=G1_MODEL,
        expected_ingest_report_hash=ingest.report_hash,
        expected_repair_report_hash=repair.report_hash,
    )
    bundle = batch.bundles[0]

    result = qualify_canonical_motion(
        bundle.episode,
        bundle.trace,
        target_model_path=G1_MODEL,
        scene_path=G1_SCENE,
    )

    assert result.qualification is MotionQualificationLevel.Q1_KINEMATIC_ONLY
    codes = {issue.code for issue in result.issues}
    assert "PELVIS_HEIGHT_FALL" in codes or "NON_FOOT_FLOOR_CONTACT" in codes
    assert result.metrics["physics_step_count"] > 0


def test_cli_contact_receipt_contains_no_frame_level_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    registration, ingest, repair = _reports(dataset)
    registration_path = tmp_path / "registration.json"
    ingest_path = tmp_path / "ingest.json"
    repair_path = tmp_path / "repair.json"
    output_path = tmp_path / "contact.json"
    qualification_path = tmp_path / "qualification.json"
    registration_path.write_text(
        json.dumps(
            {
                "registration": registration.to_dict(),
                "registration_hash": registration.registration_hash,
            }
        ),
        encoding="utf-8",
    )
    ingest_path.write_text(
        json.dumps({"report": ingest.to_dict(), "report_hash": ingest.report_hash}),
        encoding="utf-8",
    )
    repair_path.write_text(
        json.dumps({"report": repair.to_dict(), "report_hash": repair.report_hash}),
        encoding="utf-8",
    )

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "contact",
                "motiondecode",
                "--registration",
                str(registration_path),
                "--ingest-report",
                str(ingest_path),
                "--repair-report",
                str(repair_path),
                "--dataset-root",
                str(dataset),
                "--target-model",
                str(G1_MODEL),
                "--output",
                str(output_path),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["inferred_count"] == 1
    assert receipt["phase_candidate_count"] == 1
    assert receipt["frame_level_trace_persisted"] is False
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["report_hash"] == canonical_hash(artifact["report"])
    serialized = output_path.read_text(encoding="utf-8")
    assert '"left_contact"' not in serialized
    assert '"right_contact"' not in serialized
    assert '"phase_code"' not in serialized

    assert (
        dispatch_collective_argv(
            [
                "collective",
                "qualify",
                "motiondecode",
                "--registration",
                str(registration_path),
                "--ingest-report",
                str(ingest_path),
                "--repair-report",
                str(repair_path),
                "--contact-report",
                str(output_path),
                "--dataset-root",
                str(dataset),
                "--target-model",
                str(G1_MODEL),
                "--scene",
                str(G1_SCENE),
                "--output",
                str(qualification_path),
                "--source-checkout",
                str(ROOT),
            ]
        )
        == 1
    )
    qualification_receipt = json.loads(capsys.readouterr().out)
    assert qualification_receipt["physics_executed_count"] == 0
    assert qualification_receipt["physics_step_count"] == 0
    assert qualification_receipt["q3_count"] == 0
    assert qualification_receipt["hardware_authorized"] is False
