"""Dashboard evolution page data assembly tests (PR-EVO-HW-5 §十三)."""

from __future__ import annotations

import json

from rosclaw.evolution.hardware.dashboard_data import (
    assemble_evolution_page,
    render_evolution_page_html,
)


def _manifest(tmp_path) -> None:
    entries = [
        {"kind": "baseline_session", "invalid_rate": 0.2, "verified_rate": 0.8, "peak_temperature": 48},
        {"kind": "distill_gate", "memory_verify_rc": 0},
        {"kind": "propose", "candidates": ["cand_1"]},
        {"kind": "candidate_evaluated", "candidate_id": "cand_1", "state": "VALIDATED"},
        {"kind": "canary_session", "arm": "A_no_memory", "block": 0, "invalid_rate": 0.3, "verified_rate": 0.7, "peak_temperature": 50, "practice_id": "prac_a", "verify": {"rc": 0}},
        {"kind": "canary_session", "arm": "C_candidate_canary", "block": 0, "invalid_rate": 0.4, "verified_rate": 0.6, "peak_temperature": 50, "practice_id": "prac_c", "verify": {"rc": 0}, "candidate_lifecycle": {"cooldown_applied": True}},
        {"kind": "canary_aborted", "arm": "B_fixed_cooldown", "peak_temperature": 52},
        {"kind": "promotion_decision", "candidate_id": "cand_1", "decision": "ROLLED_BACK", "scope": "none", "checks": []},
        {"kind": "recurrence_blocked", "reason": "no promoted rule"},
    ]
    (tmp_path / "evidence_manifest.json").write_text(
        json.dumps({"experiment_id": "exp_1", "config_hash": "abc123", "entries": entries})
    )


def test_page_assembles_all_six_sections_with_honest_states(tmp_path) -> None:
    _manifest(tmp_path)
    page = assemble_evolution_page(
        experiment_id="exp_1",
        evidence_root=tmp_path,
        candidates=[
            {
                "candidate_id": "cand_1",
                "state": "ROLLED_BACK",
                "changes": '{"inter_round_cooldown_sec": 2.0}',
                "failed_gate": None,
                "gate_verdicts": "[]",
            }
        ],
        promoted_rules=[],
    )
    s = page["sections"]
    assert set(s) == {
        "physical_system", "flywheel", "candidate_comparison",
        "decision_evidence", "physical_execution", "promotion_state",
    }
    assert s["physical_system"]["safety_aborts"] == 1
    done_stages = {f["stage"] for f in s["flywheel"] if f["done"]}
    assert "Recurrence" not in done_stages  # honestly not reached
    assert s["candidate_comparison"]["A"][0]["invalid_rate"] == 0.3
    assert s["candidate_comparison"]["C"][0]["invalid_rate"] == 0.4
    assert s["decision_evidence"][0]["state"] == "ROLLED_BACK"  # shown, not hidden
    assert s["decision_evidence"][0]["changes"] == {"inter_round_cooldown_sec": 2.0}
    assert s["physical_execution"]["canary_sessions"][1]["candidate_lifecycle"] == {"cooldown_applied": True}
    assert s["promotion_state"]["decisions"][0]["decision"] == "ROLLED_BACK"
    assert s["promotion_state"]["promoted_rules"] == []
    assert s["promotion_state"]["recurrence_blocked"]


def test_html_template_wires_the_api_path() -> None:
    html = render_evolution_page_html("exp_1")
    assert "/api/evolution/evo-rps/exp_1" in html
    assert "Evo-RPS" in html
