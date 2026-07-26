from __future__ import annotations

from types import MappingProxyType

import pytest

from rosclaw.feedback.contracts import ControllerSnapshot, FallbackMode, FeedbackLoopSpec
from rosclaw.feedback.controllers.base import ZeroResidualController


def test_loop_spec_is_hashed_and_immutable() -> None:
    controller = ZeroResidualController()
    spec = FeedbackLoopSpec(
        loop_id="test/reflex",
        body_hash="sha256:" + "1" * 64,
        controller_hash=controller.controller_hash,
        reference_signals=("roll",),
        observation_signals=("roll",),
        output_limits={"joint:waist_roll_joint": 0.1},
    )

    assert spec.spec_hash.startswith("sha256:")
    assert len(spec.spec_hash) == 71
    assert isinstance(spec.output_limits, MappingProxyType)
    with pytest.raises(TypeError):
        spec.output_limits["joint:waist_roll_joint"] = 0.2  # type: ignore[index]


def test_loop_spec_rejects_unpinned_identity_and_impossible_deadline() -> None:
    with pytest.raises(ValueError, match="body_hash"):
        FeedbackLoopSpec(
            loop_id="bad",
            body_hash="latest",
            controller_hash="sha256:" + "2" * 64,
            reference_signals=("roll",),
            observation_signals=("roll",),
            output_limits={"joint": 0.1},
        )
    with pytest.raises(ValueError, match="deadline"):
        FeedbackLoopSpec(
            loop_id="bad",
            body_hash="sha256:" + "1" * 64,
            controller_hash="sha256:" + "2" * 64,
            reference_signals=("roll",),
            observation_signals=("roll",),
            output_limits={"joint": 0.1},
            rate_hz=200.0,
            deadline_ms=5.1,
        )


def test_fallback_modes_are_explicit_contract_values() -> None:
    assert FallbackMode.BASE_POLICY_ONLY.value == "base_policy_only"


def test_controller_snapshot_deep_freezes_config_before_hashing() -> None:
    config = {"gains": {"kp": 1.0}, "outputs": ["joint:a"]}
    snapshot = ControllerSnapshot(
        controller_id="controller",
        controller_type="test",
        body_hash="sha256:" + "1" * 64,
        loop_spec_hash="sha256:" + "2" * 64,
        config=config,
    )
    before = snapshot.snapshot_hash
    config["gains"]["kp"] = 2.0  # type: ignore[index]
    config["outputs"].append("joint:b")  # type: ignore[union-attr]

    assert snapshot.snapshot_hash == before
    assert snapshot.config["gains"]["kp"] == 1.0  # type: ignore[index]
