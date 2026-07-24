"""Evidence chain tests (PR-EVO-HW-1, 真机自进化v2 §2.8/§Phase 0.6)."""

from __future__ import annotations

import json

import pytest

from rosclaw.evolution.hardware.evidence import EvidenceManifest, file_sha256


def test_manifest_roundtrip_and_append(tmp_path) -> None:
    manifest = EvidenceManifest.open(tmp_path, "exp_1", "hash_1")
    manifest.record("prepare", ok=True)
    manifest.record("baseline_session", practice_id="prac_x", verify={"rc": 0})
    reopened = EvidenceManifest.open(tmp_path, "exp_1", "hash_1")
    assert len(reopened.entries) == 2
    assert reopened.by_kind("baseline_session")[0]["practice_id"] == "prac_x"
    summary = reopened.summary()
    assert summary["by_kind"] == {"prepare": 1, "baseline_session": 1}


def test_manifest_rejects_foreign_experiment_and_config_mutation(tmp_path) -> None:
    EvidenceManifest.open(tmp_path, "exp_1", "hash_1")
    with pytest.raises(ValueError, match="belongs to"):
        EvidenceManifest.open(tmp_path, "exp_2", "hash_1")
    with pytest.raises(ValueError, match="config hash changed"):
        EvidenceManifest.open(tmp_path, "exp_1", "hash_2")


def test_manifest_persists_json_on_disk(tmp_path) -> None:
    manifest = EvidenceManifest.open(tmp_path, "exp_1", "hash_1")
    manifest.record("storage_gate", db_doctor_rc=0)
    blob = json.loads((tmp_path / "evidence_manifest.json").read_text())
    assert blob["schema"] == "rosclaw.acceptance.evidence.v1"
    assert blob["entries"][0]["kind"] == "storage_gate"


def test_file_sha256_stable(tmp_path) -> None:
    path = tmp_path / "x.py"
    path.write_text("print('hello')\n")
    first = file_sha256(path)
    assert file_sha256(path) == first
    path.write_text("print('bye')\n")
    assert file_sha256(path) != first
