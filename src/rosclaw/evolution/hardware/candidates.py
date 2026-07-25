"""AUTO bounded candidate generation (PR-EVO-HW-3, 真机自进化v2 §7.12/§Phase 4).

AUTO v1 never writes code and never touches trajectories, servo speeds, or
safety limits: it emits CONFIG candidates from the contract's bounded
space only.  The deterministic C0–C6 template (§Phase 4) comes first —
every candidate is explainable by construction; bounded enumeration from
the config's candidate_space fills the remainder up to ``max_candidates``.

Every candidate carries its provenance (source failure + regime) and its
constraints (round budget, no-servo, no-trajectory) — the gate pipeline
refuses anything without them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import EvoRpsConfig

CONSTRAINTS = {
    "max_round_duration_ms": 20000,
    "no_servo_speed_change": True,
    "no_trajectory_change": True,
}

# §Phase 4 deterministic template — C0..C6 (cooldown classes never stack).
CANDIDATE_TEMPLATE: list[dict[str, Any]] = [
    {},  # C0: no patch (baseline identity)
    {"inter_round_cooldown_sec": 2.0},  # C1
    {"inter_round_cooldown_sec": 4.0},  # C2
    {"cooldown_every_n_rounds": 5},  # C3
    {"neutral_pose_between_blocks": True},  # C4
    {"rehome_between_blocks": True},  # C5
    {"cooldown_every_n_rounds": 5, "neutral_pose_between_blocks": True},  # C6
]

MAX_CANDIDATES = 8


class CandidateError(ValueError):
    pass


@dataclass
class Candidate:
    candidate_id: str
    changes: dict[str, Any]
    source_failure: str
    current_regime: str
    constraints: dict[str, Any] = field(default_factory=lambda: dict(CONSTRAINTS))
    created_at: float = field(default_factory=time.time)
    ordinal: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "changes": self.changes,
            "source_failure": self.source_failure,
            "current_regime": self.current_regime,
            "constraints": self.constraints,
            "created_at": self.created_at,
        }


def _candidate_id(experiment_id: str, ordinal: int, changes: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(changes, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"cand_{experiment_id}_{ordinal:03d}_{digest}"


def generate_candidates(
    config: EvoRpsConfig,
    *,
    source_failure: str,
    current_regime: str,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Bounded, explainable candidate generation (≤ max_candidates)."""
    if max_candidates > MAX_CANDIDATES:
        raise CandidateError(
            f"max_candidates {max_candidates} > {MAX_CANDIDATES} — the search "
            "space must stay explainable (§Phase 4)"
        )
    space = config.candidate_space
    seen: set[str] = set()
    raw_changes: list[dict[str, Any]] = []

    for template in CANDIDATE_TEMPLATE:
        key = json.dumps(template, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            raw_changes.append(dict(template))

    # Bounded enumeration: single-parameter points from the config space,
    # then (one cooldown-class param) × (one pose param) combos — never two
    # cooldown-class params together (§五A non-stacking).
    singles: list[dict[str, Any]] = []
    for value in space.inter_round_cooldown_sec:
        singles.append({"inter_round_cooldown_sec": value} if value else {})
    for value in space.cooldown_every_n_rounds:
        singles.append({"cooldown_every_n_rounds": value} if value else {})
    for value in space.telemetry_hz:
        if value != 5:
            singles.append({"telemetry_hz": value})
    cooldown_points = [s for s in singles if s and ("inter_round_cooldown_sec" in s or "cooldown_every_n_rounds" in s)]
    pose_points = [
        p
        for p in (
            {"neutral_pose_between_blocks": True},
            {"rehome_between_blocks": True},
        )
        if True in space.neutral_pose_between_blocks or True in space.rehome_between_blocks
    ]
    combos = [dict(itertools.chain(c.items(), p.items())) for c in cooldown_points for p in pose_points]
    for changes in singles + combos:
        key = json.dumps(changes, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            raw_changes.append(changes)

    candidates: list[Candidate] = []
    for ordinal, changes in enumerate(raw_changes[:max_candidates]):
        errors = space.validate_candidate(changes)
        if errors:
            raise CandidateError(
                f"template/enumeration produced an out-of-space candidate "
                f"{changes}: {errors} (config/template bug — refuse to proceed)"
            )
        candidates.append(
            Candidate(
                candidate_id=_candidate_id(config.experiment_id, ordinal, changes),
                changes=changes,
                source_failure=source_failure,
                current_regime=current_regime,
                ordinal=ordinal,
            )
        )
    return candidates
