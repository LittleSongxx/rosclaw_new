"""Candidate gate pipeline tests (PR-EVO-HW-3, §Phase 5)."""

from __future__ import annotations

import json

from rosclaw.evolution.hardware.candidate_gate import (
    applicability_gate,
    evaluate_candidate,
    resource_budget_gate,
    round_durations_from_events,
    safety_invariants_gate,
    schema_gate,
    shadow_gate,
    timeline_replay_gate,
)
from rosclaw.evolution.hardware.candidates import Candidate, generate_candidates
from rosclaw.evolution.hardware.contracts import load_config
from rosclaw.how.choreography import ChoreographyValidator, load_contract

CONFIG_PATH = "configs/acceptance/evo_rps_v1.yaml"
CONTRACT_PATH = "configs/choreography/rh56_rps_v1.yaml"


def _config():
    return load_config(CONFIG_PATH)


def _validator():
    return ChoreographyValidator(load_contract(CONTRACT_PATH))


def _candidate(changes, **kw):
    return Candidate(
        candidate_id=kw.get("candidate_id", "cand_test"),
        changes=changes,
        source_failure=kw.get("source_failure", "右手 剪刀 joint_not_reached 失败 恢复"),
        current_regime=kw.get("current_regime", "THERMAL_DRIFT"),
    )


def test_schema_gate_rejects_missing_constraints() -> None:
    candidate = _candidate({"inter_round_cooldown_sec": 2.0})
    candidate.constraints.pop("no_servo_speed_change")
    verdict = schema_gate(candidate)
    assert not verdict.passed
    assert "no_servo_speed_change" in verdict.detail


def test_applicability_gate_rejects_forbidden_and_stacked() -> None:
    config = _config()
    bad = _candidate({"servo_speed_scale": 0.5})
    assert not applicability_gate(bad, config).passed
    stacked = _candidate({"inter_round_cooldown_sec": 2.0, "cooldown_every_n_rounds": 5})
    verdict = applicability_gate(stacked, config)
    assert not verdict.passed
    assert "non-stackable" in verdict.detail


def test_choreography_gate_blocks_run1_patch_allows_cooldown() -> None:
    from rosclaw.how.choreography.timing import build_timing_model

    contract = load_contract(CONTRACT_PATH)
    validator = ChoreographyValidator(contract)
    model = build_timing_model(contract, [])
    run1 = _candidate({"servo_speed_scale": 0.6, "per_phase_delay_ms": 400})
    # applicability would refuse these first; check the validator directly.
    from rosclaw.evolution.hardware.candidate_gate import choreography_gate

    verdict = choreography_gate(run1, validator, model)
    assert not verdict.passed
    cooldown = _candidate({"inter_round_cooldown_sec": 4.0})
    assert choreography_gate(cooldown, validator, model).passed


def test_timeline_replay_fits_and_reports_overhead() -> None:
    verdict = timeline_replay_gate(
        _candidate({"inter_round_cooldown_sec": 2.0}),
        [5000.0, 5200.0, 5100.0],
        max_round_duration_ms=20000.0,
    )
    assert verdict.passed
    assert verdict.metrics["replayed_max_ms"] == 7200.0
    assert verdict.metrics["cooldown_overhead_s_per_session"] == 6.0


def test_timeline_replay_blocks_budget_breach() -> None:
    verdict = timeline_replay_gate(
        _candidate({"inter_round_cooldown_sec": 8.0}),
        [15000.0, 16000.0],
        max_round_duration_ms=20000.0,
    )
    assert not verdict.passed
    assert "exceeds budget" in verdict.detail


def test_shadow_gate_requires_zero_hardware_actions() -> None:
    candidate = _candidate({"inter_round_cooldown_sec": 2.0})
    assert not shadow_gate(candidate, None).passed
    bad = shadow_gate(
        candidate,
        {"hardware_actions_executed": 3, "rounds_completed": 12, "candidate_lifecycle": {}},
    )
    assert not bad.passed
    missing_lifecycle = shadow_gate(
        candidate,
        {"hardware_actions_executed": 0, "rounds_completed": 12, "candidate_lifecycle": {}},
    )
    assert not missing_lifecycle.passed
    good = shadow_gate(
        candidate,
        {
            "hardware_actions_executed": 0,
            "rounds_completed": 12,
            "candidate_lifecycle": {"cooldown_applied": True},
        },
    )
    assert good.passed


def test_resource_budget_blocks_stalling_candidates() -> None:
    verdict = resource_budget_gate(
        _candidate({"inter_round_cooldown_sec": 8.0}),
        baseline_runtime_s=100.0,
        cooldown_overhead_s=320.0,
    )
    assert not verdict.passed


def test_safety_invariants_blocks_telemetry_starvation() -> None:
    verdict = safety_invariants_gate(_candidate({"telemetry_hz": 0.5}), _config())
    assert not verdict.passed


def test_full_pipeline_validates_cooldown_candidate() -> None:
    config = _config()
    candidates = generate_candidates(config, source_failure="f", current_regime="THERMAL_DRIFT")
    cooldown = next(c for c in candidates if c.changes.get("inter_round_cooldown_sec") == 2.0)
    from rosclaw.how.choreography.timing import build_timing_model

    contract = load_contract(CONTRACT_PATH)
    evaluation = evaluate_candidate(
        cooldown,
        config,
        validator=ChoreographyValidator(contract),
        timing_model=build_timing_model(contract, []),
        round_durations_ms=[5000.0] * 40,
        baseline_runtime_s=300.0,
        shadow_run={
            "hardware_actions_executed": 0,
            "rounds_completed": 12,
            "candidate_lifecycle": {"cooldown_applied": True},
        },
    )
    assert evaluation.passed
    assert evaluation.failed_gate is None
    assert [v.gate for v in evaluation.verdicts] == [
        "schema", "applicability", "choreography", "timeline_replay",
        "shadow", "resource_budget", "safety_invariants",
    ]


def test_round_durations_from_events(tmp_path) -> None:
    """Round periods are start → next start (the contract's budget unit)."""
    events = tmp_path / "events.jsonl"
    rows = [
        {"event_type": "rps.stress.round.started", "timestamp_ns": 100_000_000_000, "payload": {"round_id": "r1"}},
        {"event_type": "rps.stress.round.resolved", "timestamp_ns": 105_200_000_000, "payload": {"round_id": "r1"}},
        {"event_type": "rps.stress.round.started", "timestamp_ns": 106_000_000_000, "payload": {"round_id": "r2"}},
        {"event_type": "rps.stress.round.resolved", "timestamp_ns": 111_800_000_000, "payload": {"round_id": "r2"}},
        {"event_type": "rps.stress.round.started", "timestamp_ns": 113_500_000_000, "payload": {"round_id": "r3"}},
    ]
    events.write_text("\n".join(json.dumps(r) for r in rows))
    durations = round_durations_from_events(events)
    assert durations == [6000.0, 7500.0]
