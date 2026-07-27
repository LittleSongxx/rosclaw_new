from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from rosclaw.continual.contracts import PolicyVersion
from rosclaw.continual.inference import (
    ResidualCandidatePolicy,
    build_g1_candidate_runtime,
    load_residual_candidate,
)
from rosclaw.continual.inference.version_lock import CandidateVersionLock

OBSERVATIONS = (
    "torso_roll",
    "torso_pitch",
    "com_y_relative",
    "support_slip_m",
    "ball_lateral_error_m",
    "contact_phase",
    "energy_margin",
    "sensor_quality",
)
ACTIONS = (
    "waist_roll_residual",
    "right_hip_roll_residual",
    "right_hip_yaw_residual",
    "kick_phase_rate",
)
LIMITS = (0.04, 0.08, 0.035, 0.08)


def _digest(value: bytes | str) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _artifact() -> bytes:
    metadata = {
        "schema_version": "rosclaw.continual.residual_sac_artifact.v1",
        "observation_names": list(OBSERVATIONS),
        "action_names": list(ACTIONS),
        "action_limits": list(LIMITS),
        "hidden_dims": [3, 2],
        "update_index": 7,
    }
    tensors = {
        "action_limits": np.asarray(LIMITS, dtype=np.float32),
        "backbone.hidden0.bias": np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
        "backbone.hidden0.weight": np.zeros((3, 8), dtype=np.float32),
        "backbone.hidden1.bias": np.zeros(2, dtype=np.float32),
        "backbone.hidden1.weight": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "backbone.output.bias": np.zeros(8, dtype=np.float32),
        "backbone.output.weight": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    chunks = [json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()]
    for name, value in sorted(tensors.items()):
        name_bytes = name.encode()
        dtype_bytes = str(value.dtype).encode()
        chunks.extend(
            (
                struct.pack("!I", len(name_bytes)),
                name_bytes,
                struct.pack("!I", len(dtype_bytes)),
                dtype_bytes,
                struct.pack("!I", value.ndim),
                struct.pack("!" + "Q" * value.ndim, *value.shape),
                value.tobytes(),
            )
        )
    return b"".join(chunks)


def _policies(artifact: bytes) -> tuple[PolicyVersion, PolicyVersion]:
    common = {
        "controller_snapshot_hash": _digest("controller"),
        "body_hash": _digest("body"),
        "safety_kernel_hash": _digest("safety"),
        "observation_names": OBSERVATIONS,
        "residual_action_names": ACTIONS,
    }
    parent = PolicyVersion(
        version=2, artifact_hash=_digest("parent"), parent_version_hash=_digest("v1"), **common
    )
    candidate = PolicyVersion(
        version=3,
        artifact_hash=_digest(artifact),
        parent_version_hash=parent.version_hash,
        **common,
    )
    return parent, candidate


def _write_candidate(tmp_path: Path) -> tuple[Path, PolicyVersion, PolicyVersion]:
    artifact = _artifact()
    parent, candidate = _policies(artifact)
    path = tmp_path / "candidate.bin"
    path.write_bytes(artifact)
    return path, parent, candidate


def test_read_only_candidate_loader_executes_bounded_numpy_actor(tmp_path: Path) -> None:
    path, parent, candidate = _write_candidate(tmp_path)

    loaded = load_residual_candidate(
        path,
        policy=candidate,
        parent=parent,
        expected_body_hash=candidate.body_hash,
    )
    runtime = ResidualCandidatePolicy(loaded)
    action = runtime.infer(dict.fromkeys(OBSERVATIONS, 0.0))
    receipt = runtime.build_receipt()

    assert action["waist_roll_residual"] == pytest.approx(np.tanh(0.2) * 0.04)
    assert all(abs(action[name]) <= limit for name, limit in zip(ACTIONS, LIMITS, strict=True))
    assert loaded.tensors["backbone.output.weight"].flags.writeable is False
    assert receipt.inference_count == 1
    assert receipt.actions_bounded
    assert receipt.action_rms > 0.0
    assert receipt.maximum_action_limit_ratio <= 1.0
    assert receipt.contact_timing_enabled_count == 0
    assert receipt.version_switch_count == 0
    assert receipt.registry_write_count == 0
    assert receipt.dds_opened is False


def test_candidate_loader_rejects_tampering_and_wrong_parent(tmp_path: Path) -> None:
    path, parent, candidate = _write_candidate(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="artifact hash"):
        load_residual_candidate(path, policy=candidate, parent=parent)

    artifact = _artifact()
    path.write_bytes(artifact)
    wrong_parent = PolicyVersion(
        version=2,
        artifact_hash=_digest("wrong-parent"),
        parent_version_hash=_digest("v1"),
        controller_snapshot_hash=parent.controller_snapshot_hash,
        body_hash=parent.body_hash,
        safety_kernel_hash=parent.safety_kernel_hash,
        observation_names=OBSERVATIONS,
        residual_action_names=ACTIONS,
    )
    with pytest.raises(ValueError, match="parent version hash"):
        load_residual_candidate(path, policy=candidate, parent=wrong_parent)


def test_g1_candidate_runtime_maps_actions_and_pins_version(tmp_path: Path) -> None:
    path, parent, candidate = _write_candidate(tmp_path)
    loaded = load_residual_candidate(path, policy=candidate, parent=parent)

    runtime, policy = build_g1_candidate_runtime(loaded, rate_hz=100.0)
    lock = CandidateVersionLock.pin(candidate)

    assert runtime.spec.body_hash == candidate.body_hash
    assert runtime.spec.output_limits["joint:waist_roll_joint"] == LIMITS[0]
    assert runtime.spec.output_limits["skill:kick_phase_rate"] == LIMITS[-1]
    lock.verify(policy.artifact.policy)
