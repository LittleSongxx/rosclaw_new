"""Evo-RPS contract validation tests (PR-EVO-HW-1, 真机自进化v2 §4)."""

from __future__ import annotations

import copy

import pytest

from rosclaw.evolution.hardware.contracts import (
    CandidateSpace,
    ValidationError,
    load_config,
)

CONFIG_PATH = "configs/acceptance/evo_rps_v1.yaml"


def _config(**overrides):
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        if "." in dotted:
            section, key = dotted.split(".", 1)
            raw.setdefault(section, {})[key] = value
        else:
            raw[dotted] = value
    return raw


def _write(tmp_path, raw) -> str:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return str(path)


def test_shipped_config_is_valid() -> None:
    config = load_config(CONFIG_PATH)
    assert config.experiment_id == "evo_rps_2026_01"
    assert config.allow_mock_camera is False
    assert config.allow_fixture_execution is False
    assert config.unattended_real_execution is False
    assert config.namespace["database"] != "rosclaw"
    assert config.config_hash


def test_cooldown_bound_exceeded_rejected(tmp_path) -> None:
    raw = _config(**{"candidate_space": {"inter_round_cooldown_sec": [0, 10]}})
    with pytest.raises(ValidationError, match="outside"):
        load_config(_write(tmp_path, raw))


def test_shared_database_rejected(tmp_path) -> None:
    raw = _config(**{"namespace": {"database": "rosclaw"}})
    with pytest.raises(ValidationError, match="isolated database"):
        load_config(_write(tmp_path, raw))


def test_mock_camera_true_rejected_at_preflight_not_load(tmp_path) -> None:
    """allow_mock_camera=true loads (dev contracts exist) but preflight
    must flag it as a contract violation in formal acceptance (§2.2)."""
    raw = _config(**{"experiment.allow_mock_camera": True})
    config = load_config(_write(tmp_path, raw))
    assert config.allow_mock_camera is True


def test_forbidden_overlap_rejected(tmp_path) -> None:
    raw = _config(forbidden_parameters=["inter_round_cooldown_sec"])
    with pytest.raises(ValidationError, match="overlaps"):
        load_config(_write(tmp_path, raw))


def test_candidate_space_validation() -> None:
    space = CandidateSpace(
        inter_round_cooldown_sec=(0.0, 2.0, 4.0),
        cooldown_every_n_rounds=(0, 5, 10),
        neutral_pose_between_blocks=(False, True),
        rehome_between_blocks=(False, True),
        telemetry_hz=(1, 5),
    )
    assert space.validate_against_bounds() == []
    assert space.validate_candidate({"inter_round_cooldown_sec": 2.0}) == []
    # Unknown parameter, out-of-space value, cooldown stacking.
    assert "unknown candidate parameter" in space.validate_candidate({"servo_speed_scale": 0.5})[0]
    assert space.validate_candidate({"inter_round_cooldown_sec": 3.0})
    stacking = space.validate_candidate(
        {"inter_round_cooldown_sec": 2.0, "cooldown_every_n_rounds": 5}
    )
    assert any("non-stackable" in e for e in stacking)
