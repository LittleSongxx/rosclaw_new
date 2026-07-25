"""AUTO bounded candidate generation tests (PR-EVO-HW-3, §7.12/§Phase 4)."""

from __future__ import annotations

import pytest

from rosclaw.evolution.hardware.candidates import (
    CANDIDATE_TEMPLATE,
    MAX_CANDIDATES,
    CandidateError,
    generate_candidates,
)
from rosclaw.evolution.hardware.contracts import load_config

CONFIG_PATH = "configs/acceptance/evo_rps_v1.yaml"


def _config():
    return load_config(CONFIG_PATH)


def test_template_candidates_all_in_space() -> None:
    config = _config()
    candidates = generate_candidates(
        config, source_failure="右手 剪刀 joint_not_reached 失败 恢复", current_regime="THERMAL_DRIFT"
    )
    assert len(candidates) <= MAX_CANDIDATES
    assert len(candidates) >= len(CANDIDATE_TEMPLATE)
    for candidate in candidates:
        assert config.candidate_space.validate_candidate(candidate.changes) == []
        assert candidate.constraints["no_servo_speed_change"] is True
        assert candidate.constraints["no_trajectory_change"] is True
        assert candidate.constraints["max_round_duration_ms"] == 20000
        assert candidate.source_failure
        assert candidate.current_regime == "THERMAL_DRIFT"
        assert candidate.candidate_id.startswith("cand_evo_rps_2026_01_")


def test_c0_is_the_empty_baseline_identity() -> None:
    candidates = generate_candidates(_config(), source_failure="f", current_regime="r")
    assert candidates[0].changes == {}


def test_cooldown_class_never_stacks() -> None:
    candidates = generate_candidates(_config(), source_failure="f", current_regime="r")
    for candidate in candidates:
        active = [
            name
            for name in ("inter_round_cooldown_sec", "cooldown_every_n_rounds")
            if candidate.changes.get(name)
        ]
        assert len(active) <= 1, f"cooldown stacking in {candidate.changes}"


def test_max_candidates_cap_enforced() -> None:
    with pytest.raises(CandidateError, match="explainable"):
        generate_candidates(
            _config(), source_failure="f", current_regime="r", max_candidates=99
        )


def test_candidate_ids_are_deterministic() -> None:
    config = _config()
    a = generate_candidates(config, source_failure="f", current_regime="r")
    b = generate_candidates(config, source_failure="f", current_regime="r")
    assert [c.candidate_id for c in a] == [c.candidate_id for c in b]
