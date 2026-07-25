"""Candidate state machine + registry tests (PR-EVO-HW-3)."""

from __future__ import annotations

from rosclaw.evolution.hardware.candidate_gate import CandidateEvaluation, GateVerdict
from rosclaw.evolution.hardware.promotion import (
    COLLECTION,
    CandidateRecord,
    CandidateRegistry,
    CandidateState,
)
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore


def _evaluation(passed: bool, failed_gate: str | None = None) -> CandidateEvaluation:
    verdicts = [GateVerdict("schema", True, "ok")]
    if not passed:
        verdicts.append(GateVerdict(failed_gate or "choreography", False, "blocked by contract"))
    return CandidateEvaluation(
        candidate_id="cand_1", passed=passed, failed_gate=failed_gate, verdicts=tuple(verdicts)
    )


def test_record_advances_to_validated() -> None:
    record = CandidateRecord(
        candidate_id="cand_1", experiment_id="exp", changes={}, source_failure="f", current_regime="r"
    )
    state = record.advance(_evaluation(True))
    assert state is CandidateState.VALIDATED
    assert record.failed_gate is None
    assert record.gate_verdicts[0]["gate"] == "schema"


def test_rejection_is_terminal_with_gate_named() -> None:
    record = CandidateRecord(
        candidate_id="cand_1", experiment_id="exp",
        changes={"servo_speed_scale": 0.5}, source_failure="f", current_regime="r",
    )
    state = record.advance(_evaluation(False, "applicability"))
    assert state is CandidateState.REJECTED
    assert record.failed_gate == "applicability"
    assert record.rejection_reason == "blocked by contract"
    blob = record.to_record()
    assert blob["state"] == "REJECTED"
    assert blob["changes"] == {"servo_speed_scale": 0.5}


def test_registry_roundtrip_in_namespace_store() -> None:
    store = InMemoryKnowledgeStore()
    store.connect()
    registry = CandidateRegistry(store)
    record = CandidateRecord(
        candidate_id="cand_x", experiment_id="exp",
        changes={"inter_round_cooldown_sec": 2.0}, source_failure="f", current_regime="r",
    )
    registry.upsert(record)
    fetched = registry.get("cand_x")
    assert fetched is not None
    assert fetched["state"] == "PROPOSED"
    record.advance(_evaluation(True))
    registry.upsert(record)
    assert registry.get("cand_x")["state"] == "VALIDATED"
    assert len(registry.by_state(CandidateState.VALIDATED)) == 1
    # upsert replaces, never duplicates
    assert len(store.query(COLLECTION, filters=None, limit=10)) == 1
