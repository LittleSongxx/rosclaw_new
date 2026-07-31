"""TwinTouch configuration — hard safety constants + evolution bounds.

Two disjoint key spaces (v4 §12):

* HARD limits — never candidate-modifiable, never tuned by HOW, loaded
  once and enforced by the gateway/supervisor: force ceilings,
  temperature gates, joint range, servo speeds during approach,
  one-pair rule, camera freshness, permit lifetime, and the
  unvalidated-pair ban.
* Evolution-eligible compensation bounds (§12.1) — the ONLY keys an
  AUTO candidate may touch, each with a hard bound.  A candidate
  proposing anything outside the whitelist, or beyond the bound, is
  rejected at schema time, not at runtime.

Numeric sources: coarse/fine step sizes and approach speeds come from
the proven single-hand disciplines (40-raw kernel steps; two-phase
40→10 OK approach; speed 150 cap).  Force thresholds are PROVISIONAL
upper bounds pending T1 calibration — T1 may lower them, never raise
them above these constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "rosclaw.twintouch_config.v1"

# §12.2 — hard limits, never evolution variables.
EVOLUTION_FORBIDDEN_KEYS = frozenset(
    {
        "hard_force_limit_raw",
        "temperature_start_max_c",
        "temperature_abort_c",
        "temperature_hardware_protection_c",
        "joint_min_raw",
        "joint_max_raw",
        "servo_max_speed_approach",
        "servo_max_speed_fine",
        "max_active_pairs",
        "camera_freshness_ms",
        "permit_lifetime_s",
        "authorization_scope",
    }
)

# §12.1 — the only keys a candidate may propose, with hard bounds.
EVOLUTION_BOUNDS: dict[str, tuple[float, float]] = {
    "finger_precontact_offset_raw": (-10.0, 10.0),
    "finger_contact_offset_raw": (-10.0, 10.0),
    "release_margin_raw": (0.0, 20.0),
    "settle_extension_ms": (0.0, 500.0),
    "left_right_start_skew_ms": (-200.0, 200.0),
    "fine_approach_step_raw": (5.0, 20.0),
    "peer_wait_timeout_ms": (500.0, 5000.0),
    "visual_near_threshold_m": (0.005, 0.05),
    "frame_voting_count": (1.0, 5.0),
    "contact_confirm_frames": (1.0, 5.0),
    "retreat_distance_raw": (50.0, 200.0),
    "retry_count": (0.0, 3.0),
    # Categorical §12.1 keys (no numeric bound): active_side_selection,
    # retry_side, rebaseline_force_before_retry — validated by name only.
}
EVOLUTION_CATEGORICAL_KEYS = frozenset(
    {"active_side_selection", "retry_side", "rebaseline_force_before_retry"}
)


@dataclass(frozen=True)
class TwinTouchConfig:
    """Runtime-enforced constants.  Defaults are the phase-1 safest
    values; the YAML may only make contact thresholds LOWER."""

    # Hard force / thermal / range limits (§12.2)
    hard_force_limit_raw: int = 300
    temperature_start_max_c: float = 46.0
    temperature_abort_c: float = 49.0
    temperature_hardware_protection_c: float = 52.0
    joint_min_raw: int = 50
    joint_max_raw: int = 1000
    servo_max_speed_approach: int = 150
    servo_max_speed_fine: int = 100
    max_active_pairs: int = 1
    camera_freshness_ms: float = 500.0
    permit_lifetime_s: float = 300.0

    # Approach discipline (proven single-hand values)
    coarse_step_raw: int = 40
    fine_step_raw: int = 10
    approach_force_set: int = 150
    dwell_force_set: int = 100
    coast_force_set: int = 50
    retreat_speed: int = 300
    retreat_force_set: int = 150

    # Contact thresholds — PROVISIONAL ceilings, T1-calibrated downward.
    contact_force_delta_raw: int = 60  # bilateral rise above session baseline
    non_target_force_abort_raw: int = 60  # any non-target finger rise => abort
    dwell_ms_default: float = 300.0

    # Coordination
    max_start_skew_ms: float = 250.0
    peer_wait_timeout_ms: float = 2000.0

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.approach_force_set > self.hard_force_limit_raw:
            violations.append("approach_force_set exceeds hard force limit")
        if self.dwell_force_set > self.hard_force_limit_raw:
            violations.append("dwell_force_set exceeds hard force limit")
        if self.retreat_force_set > self.hard_force_limit_raw:
            violations.append("retreat_force_set exceeds hard force limit")
        if self.contact_force_delta_raw > self.hard_force_limit_raw:
            violations.append("contact_force_delta_raw exceeds hard force limit")
        if self.temperature_start_max_c >= self.temperature_abort_c:
            violations.append("start temperature gate must be below abort gate")
        if self.temperature_abort_c >= self.temperature_hardware_protection_c:
            violations.append("abort gate must be below hardware protection")
        if self.fine_step_raw > self.coarse_step_raw:
            violations.append("fine step must not exceed coarse step")
        if self.servo_max_speed_fine > self.servo_max_speed_approach:
            violations.append("fine speed must not exceed approach speed")
        if self.max_active_pairs != 1:
            violations.append("one-pair-at-a-time rule is not configurable (v4 §3.2)")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **self.__dict__}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TwinTouchConfig:
        payload = {k: v for k, v in record.items() if k != "schema_version"}
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path | None = None) -> TwinTouchConfig:
        if path is None:
            path = Path(__file__).parent / "config" / "twintouch.default.yaml"
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_record(data)


def validate_candidate_changes(changes: dict[str, Any]) -> list[str]:
    """Gate every AUTO candidate's change set against §12: forbidden
    keys and out-of-bound values are schema-level rejections."""
    violations: list[str] = []
    for key, value in changes.items():
        if key in EVOLUTION_FORBIDDEN_KEYS:
            violations.append(f"{key} is a hard safety limit — never candidate-modifiable")
            continue
        if key in EVOLUTION_CATEGORICAL_KEYS:
            continue
        bounds = EVOLUTION_BOUNDS.get(key)
        if bounds is None:
            violations.append(f"{key} is not an evolution-eligible compensation (§12.1)")
            continue
        if not isinstance(value, (int, float)):
            violations.append(f"{key} candidate value must be numeric")
            continue
        low, high = bounds
        if not (low <= float(value) <= high):
            violations.append(f"{key}={value} outside hard bound [{low}, {high}]")
    return violations
