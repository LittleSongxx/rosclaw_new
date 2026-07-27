"""Declarative safety envelope for the CMU ARE demo app.

The numbers that bound CMU ARE navigation (goal geofence, largest single
relative move, speed ceiling, LLM step budget) live in the ``cmu_are``
embodiment card under ``operational_limits`` / ``workspace_boundaries``. This
module resolves that card into a :class:`CmuSafetyLimits` value the bridge can
consult, and centralises the geofence check.

Resolution order, highest priority first:

1. explicit keyword overrides (CLI flags),
2. the embodiment card,
3. the module-level fallbacks in :mod:`rosclaw.apps.cmu_are_bridge`.

The card is optional: with no card and no overrides the resolved limits equal
the historical hardcoded defaults, so behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.connectors.ros.embodiment import (
    EmbodimentCard,
    EmbodimentCardError,
    load_embodiment_card,
)

# Robot id whose embodiment card describes the CMU ARE base.
CMU_ARE_ROBOT_ID = "cmu_are"

# Fallbacks used when neither an override nor a card supplies a value. These
# mirror the historical constants in rosclaw.apps.cmu_are_bridge.
FALLBACK_MAX_ABSOLUTE_COORDINATE_M = 100.0
FALLBACK_MAX_RELATIVE_MOVE_M = 20.0
FALLBACK_MAX_CIRCLE_RADIUS_M = 6.0
FALLBACK_MAX_SEQUENCE_STEPS = 8
FALLBACK_MAX_SPEED = 2.0
FALLBACK_TOLERANCE_M = 1.5


class CmuGeofenceError(Exception):
    """A navigation goal falls outside the permitted workspace."""


@dataclass(frozen=True)
class CmuSafetyLimits:
    """Resolved numeric safety envelope for CMU ARE navigation."""

    max_absolute_coordinate_m: float = FALLBACK_MAX_ABSOLUTE_COORDINATE_M
    max_relative_move_m: float = FALLBACK_MAX_RELATIVE_MOVE_M
    max_circle_radius_m: float = FALLBACK_MAX_CIRCLE_RADIUS_M
    max_sequence_steps: int = FALLBACK_MAX_SEQUENCE_STEPS
    max_speed: float = FALLBACK_MAX_SPEED
    default_tolerance_m: float = FALLBACK_TOLERANCE_M
    workspace_boundaries: dict[str, Any] | None = None
    robot_id: str = CMU_ARE_ROBOT_ID
    source: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_absolute_coordinate_m": self.max_absolute_coordinate_m,
            "max_relative_move_m": self.max_relative_move_m,
            "max_circle_radius_m": self.max_circle_radius_m,
            "max_sequence_steps": self.max_sequence_steps,
            "max_speed": self.max_speed,
            "default_tolerance_m": self.default_tolerance_m,
            "workspace_boundaries": self.workspace_boundaries,
            "robot_id": self.robot_id,
            "source": self.source,
        }

    def clamp_speed(self, speed: float) -> float:
        """Clamp a requested /speed value into the permitted range."""
        return max(0.0, min(float(speed), self.max_speed))

    def with_overrides(self, **overrides: Any) -> CmuSafetyLimits:
        """Return a copy with non-``None`` overrides applied."""
        applied = {k: v for k, v in overrides.items() if v is not None}
        if not applied:
            return self
        return replace(self, **applied)


def _positive_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if value > 0 else default


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = int(value)
    return value if value > 0 else default


def resolve_cmu_safety_limits(
    *,
    card: EmbodimentCard | None = None,
    robot_id: str = CMU_ARE_ROBOT_ID,
    specs_dir: Path | None = None,
    max_absolute_coordinate_m: float | None = None,
    max_relative_move_m: float | None = None,
    max_circle_radius_m: float | None = None,
    max_sequence_steps: int | None = None,
    max_speed: float | None = None,
    default_tolerance_m: float | None = None,
    workspace_boundaries: dict[str, Any] | None = None,
) -> CmuSafetyLimits:
    """Resolve the effective CMU ARE safety limits.

    Args:
        card: A pre-loaded embodiment card. When ``None``, the card for
            ``robot_id`` is loaded from disk if present.
        robot_id: Embodiment card id to look up.
        specs_dir: Override the specs directory (used by tests).
        max_absolute_coordinate_m: Override the goal geofence half-extent.
        max_relative_move_m: Override the largest single relative move.
        max_circle_radius_m: Override the largest circle-task radius.
        max_sequence_steps: Override the LLM waypoint budget.
        max_speed: Override the published ``/speed`` ceiling.
        default_tolerance_m: Override the goal-reached tolerance.
        workspace_boundaries: Override the hard geofence.

    A malformed card is treated as absent: the fallbacks apply rather than
    failing navigation closed on a config typo. The returned ``source`` records
    which path was taken.
    """
    resolved_card = card
    source = "fallback"

    if resolved_card is None:
        try:
            resolved_card = load_embodiment_card(robot_id, specs_dir=specs_dir)
        except EmbodimentCardError:
            resolved_card = None

    limits = CmuSafetyLimits(robot_id=robot_id)

    if resolved_card is not None:
        source = "embodiment_card"
        limits = CmuSafetyLimits(
            max_absolute_coordinate_m=_positive_float(
                resolved_card.limit("max_absolute_coordinate_m"),
                FALLBACK_MAX_ABSOLUTE_COORDINATE_M,
            ),
            max_relative_move_m=_positive_float(
                resolved_card.limit("max_relative_move_m"),
                FALLBACK_MAX_RELATIVE_MOVE_M,
            ),
            max_circle_radius_m=_positive_float(
                resolved_card.limit("max_circle_radius_m"),
                FALLBACK_MAX_CIRCLE_RADIUS_M,
            ),
            max_sequence_steps=_positive_int(
                resolved_card.operational_limits.get("max_sequence_steps"),
                FALLBACK_MAX_SEQUENCE_STEPS,
            ),
            max_speed=_positive_float(
                resolved_card.limit("max_speed"), FALLBACK_MAX_SPEED
            ),
            default_tolerance_m=_positive_float(
                resolved_card.limit("default_tolerance_m"), FALLBACK_TOLERANCE_M
            ),
            workspace_boundaries=dict(resolved_card.workspace_boundaries) or None,
            robot_id=resolved_card.robot_id,
            source=source,
        )

    limits = limits.with_overrides(
        max_absolute_coordinate_m=max_absolute_coordinate_m,
        max_relative_move_m=max_relative_move_m,
        max_circle_radius_m=max_circle_radius_m,
        max_sequence_steps=max_sequence_steps,
        max_speed=max_speed,
        default_tolerance_m=default_tolerance_m,
        workspace_boundaries=workspace_boundaries,
    )

    overridden = any(
        value is not None
        for value in (
            max_absolute_coordinate_m,
            max_relative_move_m,
            max_circle_radius_m,
            max_sequence_steps,
            max_speed,
            default_tolerance_m,
            workspace_boundaries,
        )
    )
    if overridden:
        limits = replace(limits, source=f"{source}+override" if source != "fallback" else "override")

    return limits


def resolve_workspace_ranges(
    workspace_boundaries: dict[str, Any] | None,
) -> tuple[list[float], list[float], list[float]] | None:
    """Normalise a geofence declaration into ``(x_range, y_range, z_range)``.

    Accepts either explicit ``{x: [lo, hi], ...}`` ranges or
    ``{type: bounding_box, center: [...], dimensions: [...]}``. Returns ``None``
    when no usable geofence is declared.
    """
    if not workspace_boundaries:
        return None

    if workspace_boundaries.get("type") == "bounding_box":
        center = workspace_boundaries.get("center", [0.0, 0.0, 0.0])
        dims = workspace_boundaries.get("dimensions", [0.0, 0.0, 0.0])
        if len(center) < 3 or len(dims) < 3:
            raise CmuGeofenceError(
                "bounding_box geofence requires 3-element 'center' and 'dimensions'"
            )
        return (
            [center[0] - dims[0] / 2, center[0] + dims[0] / 2],
            [center[1] - dims[1] / 2, center[1] + dims[1] / 2],
            [center[2] - dims[2] / 2, center[2] + dims[2] / 2],
        )

    infinite = [-float("inf"), float("inf")]
    x_range = list(workspace_boundaries.get("x", infinite))
    y_range = list(workspace_boundaries.get("y", infinite))
    z_range = list(workspace_boundaries.get("z", infinite))
    for name, rng in (("x", x_range), ("y", y_range), ("z", z_range)):
        if len(rng) < 2:
            raise CmuGeofenceError(f"geofence range '{name}' must have two elements")
    return x_range, y_range, z_range


def check_within_workspace(
    *,
    x: float,
    y: float,
    z: float,
    workspace_boundaries: dict[str, Any] | None,
) -> None:
    """Raise :class:`CmuGeofenceError` when a goal is outside the geofence.

    A missing or empty geofence permits everything; the absolute-coordinate cap
    in :class:`CmuSafetyLimits` is the always-on backstop.
    """
    ranges = resolve_workspace_ranges(workspace_boundaries)
    if ranges is None:
        return
    x_range, y_range, z_range = ranges

    violations: list[str] = []
    for axis, value, rng in (("x", x, x_range), ("y", y, y_range), ("z", z, z_range)):
        if value < rng[0] or value > rng[1]:
            violations.append(f"{axis}={value:.2f} not in [{rng[0]:.2f}, {rng[1]:.2f}]")

    if violations:
        raise CmuGeofenceError(
            f"Navigation goal violates workspace boundaries: {', '.join(violations)}"
        )


__all__ = [
    "CMU_ARE_ROBOT_ID",
    "FALLBACK_MAX_ABSOLUTE_COORDINATE_M",
    "FALLBACK_MAX_CIRCLE_RADIUS_M",
    "FALLBACK_MAX_RELATIVE_MOVE_M",
    "FALLBACK_MAX_SEQUENCE_STEPS",
    "FALLBACK_MAX_SPEED",
    "FALLBACK_TOLERANCE_M",
    "CmuGeofenceError",
    "CmuSafetyLimits",
    "check_within_workspace",
    "resolve_cmu_safety_limits",
    "resolve_workspace_ranges",
]
