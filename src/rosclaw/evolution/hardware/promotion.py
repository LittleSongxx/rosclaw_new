"""Candidate state machine + registry (PR-EVO-HW-3, v4 §9 lineage).

    PROPOSED → SCHEMA → APPLICABILITY → CHOREOGRAPHY → TIMELINE_REPLAY
    → SHADOW → VALIDATED | REJECTED(gate, reason)

Registry rows persist into the EXPERIMENT NAMESPACE knowledge store
(``evo_candidates`` collection) — never the shared database — and every
transition lands in the evidence manifest.  A REJECTED candidate is
terminal; a VALIDATED one becomes eligible for the HW-4 canary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .candidate_gate import CandidateEvaluation


class CandidateState(StrEnum):
    PROPOSED = "PROPOSED"
    SCHEMA = "SCHEMA"
    APPLICABILITY = "APPLICABILITY"
    CHOREOGRAPHY = "CHOREOGRAPHY"
    TIMELINE_REPLAY = "TIMELINE_REPLAY"
    SHADOW = "SHADOW"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


ORDER = (
    CandidateState.PROPOSED,
    CandidateState.SCHEMA,
    CandidateState.APPLICABILITY,
    CandidateState.CHOREOGRAPHY,
    CandidateState.TIMELINE_REPLAY,
    CandidateState.SHADOW,
    CandidateState.VALIDATED,
)


@dataclass
class CandidateRecord:
    candidate_id: str
    experiment_id: str
    changes: dict[str, Any]
    source_failure: str
    current_regime: str
    state: CandidateState = CandidateState.PROPOSED
    failed_gate: str | None = None
    rejection_reason: str = ""
    gate_verdicts: list[dict[str, Any]] = field(default_factory=list)
    baseline_practice_id: str | None = None
    updated_at: float = field(default_factory=time.time)

    def advance(self, evaluation: CandidateEvaluation) -> CandidateState:
        """Fold one gate-pipeline evaluation into the state machine."""
        self.gate_verdicts = [
            {"gate": v.gate, "passed": v.passed, "detail": v.detail, "metrics": v.metrics}
            for v in evaluation.verdicts
        ]
        self.updated_at = time.time()
        if evaluation.passed:
            self.state = CandidateState.VALIDATED
            self.failed_gate = None
            self.rejection_reason = ""
        else:
            self.state = CandidateState.REJECTED
            self.failed_gate = evaluation.failed_gate
            last = evaluation.verdicts[-1]
            self.rejection_reason = last.detail
        return self.state

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "changes": self.changes,
            "source_failure": self.source_failure,
            "current_regime": self.current_regime,
            "state": self.state.value,
            "failed_gate": self.failed_gate,
            "rejection_reason": self.rejection_reason,
            "gate_verdicts": self.gate_verdicts,
            "baseline_practice_id": self.baseline_practice_id,
            "updated_at": self.updated_at,
        }


COLLECTION = "evo_candidates"


class CandidateRegistry:
    """Namespace-scoped persistence for candidate records."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def upsert(self, record: CandidateRecord) -> None:
        existing = self._store.query(COLLECTION, filters={"id": record.candidate_id}, limit=1)
        if existing:
            self._store.delete(COLLECTION, record.candidate_id)
        self._store.insert(COLLECTION, record.to_record())

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        rows = self._store.query(COLLECTION, filters={"id": candidate_id}, limit=1)
        return rows[0] if rows else None

    def by_state(self, state: CandidateState) -> list[dict[str, Any]]:
        return self._store.query(COLLECTION, filters={"state": state.value}, limit=100)

    def all(self) -> list[dict[str, Any]]:
        return self._store.query(COLLECTION, filters=None, limit=200)
