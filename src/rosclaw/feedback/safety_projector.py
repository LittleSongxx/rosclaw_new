"""Synchronous fail-closed projection for feedback residual commands."""

from __future__ import annotations

import math
from collections.abc import Mapping

from rosclaw.feedback.contracts import (
    FallbackMode,
    FeedbackLoopSpec,
    ResidualCommand,
    canonical_hash,
)


class FeedbackSafetyProjector:
    """Clamp residual and absolute action bounds before it reaches control."""

    def __init__(
        self,
        spec: FeedbackLoopSpec,
        *,
        absolute_action_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.spec = spec
        self.absolute_action_bounds = dict(absolute_action_bounds or {})
        for key, bounds in self.absolute_action_bounds.items():
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"invalid absolute action bounds for {key}")

    def project(
        self,
        *,
        sequence: int,
        timestamp_ns: int,
        base_action: Mapping[str, float],
        raw: Mapping[str, float],
        deadline_met: bool = True,
    ) -> ResidualCommand:
        base_hash = canonical_hash({"action": dict(sorted(base_action.items()))})
        invalid = sorted(
            key
            for key, value in raw.items()
            if key not in self.spec.output_limits or not math.isfinite(float(value))
        )
        if invalid:
            return self._fallback(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                base_action_hash=base_hash,
                raw=raw,
                reason="unsafe_projection:" + ",".join(invalid),
                mode=self.spec.fallback_unsafe_projection,
                deadline_met=deadline_met,
            )
        projected: dict[str, float] = {}
        reasons: list[str] = []
        saturation_count = 0
        for key, value in raw.items():
            value = float(value)
            limit = self.spec.output_limits[key]
            bounded = max(-limit, min(limit, value))
            if key in self.absolute_action_bounds:
                lower, upper = self.absolute_action_bounds[key]
                base = float(base_action.get(key, 0.0))
                bounded = max(lower - base, min(upper - base, bounded))
            if not math.isclose(bounded, value, rel_tol=0.0, abs_tol=1e-15):
                saturation_count += 1
                reasons.append(f"saturated:{key}")
            projected[key] = bounded
        return ResidualCommand(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            valid_until_ns=timestamp_ns + self.spec.period_ns,
            base_action_hash=base_hash,
            raw=raw,
            projected=projected,
            reasons=tuple(reasons),
            saturation_count=saturation_count,
            deadline_met=deadline_met,
        )

    def fallback(
        self,
        *,
        sequence: int,
        timestamp_ns: int,
        base_action: Mapping[str, float],
        reason: str,
        mode: FallbackMode,
        deadline_met: bool,
        raw: Mapping[str, float] | None = None,
        projected_override: Mapping[str, float] | None = None,
    ) -> ResidualCommand:
        return self._fallback(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            base_action_hash=canonical_hash({"action": dict(sorted(base_action.items()))}),
            raw=raw or {},
            reason=reason,
            mode=mode,
            deadline_met=deadline_met,
            projected_override=projected_override,
        )

    def _fallback(
        self,
        *,
        sequence: int,
        timestamp_ns: int,
        base_action_hash: str,
        raw: Mapping[str, float],
        reason: str,
        mode: FallbackMode,
        deadline_met: bool,
        projected_override: Mapping[str, float] | None = None,
    ) -> ResidualCommand:
        safe_raw: dict[str, float] = {}
        for key, value in raw.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                safe_raw[str(key)] = numeric
        projected = (
            dict(projected_override or {}) if mode is FallbackMode.FREEZE_AND_STABILIZE else {}
        )
        return ResidualCommand(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            valid_until_ns=timestamp_ns + self.spec.period_ns,
            base_action_hash=base_action_hash,
            raw=safe_raw,
            projected=projected,
            reasons=(reason,),
            saturation_count=0,
            deadline_met=deadline_met,
            fallback=mode,
        )
