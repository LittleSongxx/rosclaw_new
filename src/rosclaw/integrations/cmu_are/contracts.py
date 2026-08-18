"""Strict, ROS-free CMU ARE simulation contracts.

Only pure validation and configuration lives here.  ROS communication is
isolated in :mod:`adapter`, which talks to the container through rosbridge.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CMU_ARE_BODY_ID = "cmu_are_sim"
CMU_ARE_NAV_SCHEMA = "cmu_are.navigation.v1"
CMU_ARE_EXPLORE_SCHEMA = "cmu_are.exploration.v1"
CMU_ARE_STOP_SCHEMA = "cmu_are.stop.v1"
CMU_ARE_CARD_SCHEMA = "rosclaw.embodiment_card.v1"


class CmuAreContractError(ValueError):
    """Raised when a CMU ARE contract or action is malformed."""


def _strict_number(value: Any, field: str) -> float:
    """Return a finite YAML/JSON number without coercing strings or bools."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CmuAreContractError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise CmuAreContractError(f"{field} must be finite")
    return converted


def _strict_positive_number(value: Any, field: str) -> float:
    converted = _strict_number(value, field)
    if converted <= 0:
        raise CmuAreContractError(f"{field} must be positive")
    return converted


def _strict_positive_int(value: Any, field: str) -> int:
    # Do not accept 8.0 or "8" here.  A sequence budget is a structural
    # contract, not a numeric value that may be silently truncated.
    if type(value) is not int or value <= 0:  # noqa: E721 - exact bool-safe check
        raise CmuAreContractError(f"{field} must be a positive integer")
    return value


def _strict_boundary(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CmuAreContractError(f"{field} must be a two-value list")
    lower = _strict_number(value[0], f"{field}[0]")
    upper = _strict_number(value[1], f"{field}[1]")
    if lower >= upper:
        raise CmuAreContractError(f"{field} must be increasing")
    return [lower, upper]


@dataclass(frozen=True)
class CmuPlace:
    """One allow-listed map-frame navigation target."""

    name: str
    x: float
    y: float
    z: float = 0.0
    frame_id: str = "map"
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CmuAreSafetyContract:
    max_absolute_coordinate_m: float
    max_relative_move_m: float
    max_circle_radius_m: float
    max_sequence_steps: int
    max_speed_mps: float
    default_tolerance_m: float
    workspace_boundaries: dict[str, Any]
    source: str
    digest: str

    def validate_goal(self, *, x: float, y: float, z: float) -> None:
        values = {"x": x, "y": y, "z": z}
        for axis, value in values.items():
            converted = _strict_number(value, f"goal {axis}")
            if abs(converted) > self.max_absolute_coordinate_m:
                raise CmuAreContractError(
                    f"goal {axis} exceeds absolute limit ±{self.max_absolute_coordinate_m:g}m"
                )

        ranges = self.workspace_boundaries
        for axis, value in values.items():
            declared = ranges.get(axis)
            if declared is None:
                continue
            lower, upper = _strict_boundary(declared, f"workspace_boundaries.{axis}")
            converted = _strict_number(value, f"goal {axis}")
            if converted < lower or converted > upper:
                raise CmuAreContractError(
                    f"goal {axis}={converted:g} is outside [{lower:g}, {upper:g}]"
                )

    def validate_relative_move(self, *, dx: float, dy: float, dz: float = 0.0) -> None:
        """Validate a relative move without silently clipping it.

        The old imperative bridge clipped oversized requests.  The daemon
        contract rejects them so the caller has an auditable failure instead
        of an unannounced change in intent.
        """

        values = {"dx": dx, "dy": dy, "dz": dz}
        converted_values = {
            axis: _strict_number(value, f"relative move {axis}") for axis, value in values.items()
        }
        distance = math.sqrt(sum(value**2 for value in converted_values.values()))
        if distance > self.max_relative_move_m:
            raise CmuAreContractError(
                f"relative move magnitude {distance:g}m exceeds limit {self.max_relative_move_m:g}m"
            )

    def validate_sequence(self, sequence: list[Any]) -> None:
        """Reject waypoint sequences that exceed the declarative step budget."""

        if not isinstance(sequence, list):
            raise CmuAreContractError("waypoint sequence must be a list")
        if len(sequence) > self.max_sequence_steps:
            raise CmuAreContractError(
                f"waypoint sequence has {len(sequence)} steps; maximum is {self.max_sequence_steps}"
            )

    def clamp_speed(self, speed: float) -> float:
        converted = _strict_positive_number(speed, "speed_mps")
        if converted > self.max_speed_mps:
            raise CmuAreContractError(f"speed_mps exceeds limit {self.max_speed_mps:g}")
        return converted

    def validate_tolerance(self, tolerance: float) -> float:
        converted = _strict_number(tolerance, "tolerance_m")
        if not 0 < converted <= 20:
            raise CmuAreContractError("tolerance_m must be in (0, 20]")
        return converted

    def validate_timeout(self, timeout: float) -> float:
        converted = _strict_number(timeout, "timeout_sec")
        if not 0 < converted <= 600:
            raise CmuAreContractError("timeout_sec must be in (0, 600]")
        return converted


def _project_root() -> Path:
    # integrations/cmu_are/contracts.py -> repo root in a source checkout.
    return Path(__file__).resolve().parents[4]


def default_card_path() -> Path:
    return _project_root() / "src/rosclaw/connectors/ros/specs/cmu_are.yaml"


def default_places_path() -> Path:
    return _project_root() / "docker/ros1/places.campus.yaml"


def load_safety_contract(path: str | Path | None = None) -> CmuAreSafetyContract:
    """Load the CMU card fail-closed; there are no numeric fallbacks."""

    source = Path(path or default_card_path()).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != CMU_ARE_CARD_SCHEMA:
            raise CmuAreContractError("unsupported embodiment card schema")
        if str(raw.get("robot_id")) not in {"cmu_are", "cmu_are_sim"}:
            raise CmuAreContractError("card robot_id is not CMU ARE")
        limits = raw["operational_limits"]
        boundary = raw["workspace_boundaries"]
        if not isinstance(limits, dict) or not isinstance(boundary, dict):
            raise CmuAreContractError(
                "card must declare operational_limits and workspace_boundaries"
            )
        values = {
            "max_absolute_coordinate_m": _strict_positive_number(
                limits.get("max_absolute_coordinate_m"),
                "operational_limits.max_absolute_coordinate_m",
            ),
            "max_relative_move_m": _strict_positive_number(
                limits.get("max_relative_move_m"),
                "operational_limits.max_relative_move_m",
            ),
            "max_circle_radius_m": _strict_positive_number(
                limits.get("max_circle_radius_m"),
                "operational_limits.max_circle_radius_m",
            ),
            "max_sequence_steps": _strict_positive_int(
                limits.get("max_sequence_steps"),
                "operational_limits.max_sequence_steps",
            ),
            "max_speed_mps": _strict_positive_number(
                limits.get("max_speed"), "operational_limits.max_speed"
            ),
            "default_tolerance_m": _strict_positive_number(
                limits.get("default_tolerance_m"),
                "operational_limits.default_tolerance_m",
            ),
        }
        boundaries = {
            axis: _strict_boundary(boundary.get(axis), f"workspace_boundaries.{axis}")
            for axis in ("x", "y", "z")
        }
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return CmuAreSafetyContract(
            max_absolute_coordinate_m=values["max_absolute_coordinate_m"],
            max_relative_move_m=values["max_relative_move_m"],
            max_circle_radius_m=values["max_circle_radius_m"],
            max_sequence_steps=int(values["max_sequence_steps"]),
            max_speed_mps=values["max_speed_mps"],
            default_tolerance_m=values["default_tolerance_m"],
            workspace_boundaries=boundaries,
            source=str(source),
            digest=f"sha256:{digest}",
        )
    except CmuAreContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise CmuAreContractError(f"failed to load CMU ARE safety card: {exc}") from exc


def load_places(path: str | Path | None = None) -> dict[str, CmuPlace]:
    source = Path(path or default_places_path()).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CmuAreContractError("places file must contain a mapping")
        raw_places = raw.get("places", raw)
        if not isinstance(raw_places, dict):
            raise CmuAreContractError("places must be a mapping")
        places: dict[str, CmuPlace] = {}
        aliases: set[str] = set()
        for raw_name, value in raw_places.items():
            name = str(raw_name).strip()
            if not name or not isinstance(value, dict):
                raise CmuAreContractError("every place requires a name and mapping")
            unknown = set(value) - {"x", "y", "z", "frame_id", "aliases"}
            if unknown:
                raise CmuAreContractError(
                    f"place {name!r} contains unknown fields: {', '.join(sorted(unknown))}"
                )
            x = _strict_number(value.get("x"), f"place {name!r} x")
            y = _strict_number(value.get("y"), f"place {name!r} y")
            z = _strict_number(value.get("z", 0.0), f"place {name!r} z")
            frame_id = value.get("frame_id", "map")
            if frame_id != "map":
                raise CmuAreContractError(f"place {name!r} must use the map frame")
            raw_aliases = value.get("aliases", [])
            if not isinstance(raw_aliases, list) or any(
                not isinstance(alias, str) or not alias.strip() for alias in raw_aliases
            ):
                raise CmuAreContractError(f"place {name!r} aliases must be non-empty strings")
            normalized_aliases = tuple(alias.strip() for alias in raw_aliases)
            identities = {name.casefold(), *(alias.casefold() for alias in normalized_aliases)}
            if identities.intersection(aliases):
                raise CmuAreContractError(f"place {name!r} has a duplicate name or alias")
            aliases.update(identities)
            places[name] = CmuPlace(
                name=name,
                x=x,
                y=y,
                z=z,
                frame_id=frame_id,
                aliases=normalized_aliases,
            )
    except CmuAreContractError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise CmuAreContractError(f"failed to load CMU ARE places: {exc}") from exc
    if not places:
        raise CmuAreContractError("CMU ARE places file is empty")
    return places


def resolve_target(
    *,
    place: str | None,
    x: float | None,
    y: float | None,
    z: float = 0.0,
    frame_id: str = "map",
    places: dict[str, CmuPlace],
    safety: CmuAreSafetyContract,
) -> dict[str, Any]:
    if place and (x is not None or y is not None):
        raise CmuAreContractError("use either --place or --x/--y, not both")
    if place:
        lowered = place.casefold()
        target = places.get(place) or places.get(lowered)
        if target is None:
            for candidate in places.values():
                aliases = (candidate.name, *candidate.aliases)
                if any(lowered == alias.casefold() for alias in aliases):
                    target = candidate
                    break
        if target is None:
            raise CmuAreContractError(f"unknown CMU ARE place: {place}")
        x, y, z, frame_id = target.x, target.y, target.z, target.frame_id
    if x is None or y is None:
        raise CmuAreContractError("navigation requires --place or both --x and --y")
    if not isinstance(frame_id, str) or not frame_id or len(frame_id) > 128:
        raise CmuAreContractError("frame_id must be a non-empty bounded string")
    safety.validate_goal(x=x, y=y, z=z)
    coordinates = {
        axis: _strict_number(value, f"target {axis}")
        for axis, value in {"x": x, "y": y, "z": z}.items()
    }
    return {
        "frame_id": frame_id,
        **coordinates,
    }


def body_snapshot_hash(
    *,
    safety: CmuAreSafetyContract | None = None,
    asset_manifest: str | Path | None = None,
    pack_version: str = "0.1.0",
) -> str:
    """Derive a deterministic snapshot hash from immutable simulation contracts."""

    contract = safety or load_safety_contract()
    manifest_path = Path(asset_manifest or (_project_root() / "docs/assets/cmu-are-assets.yaml"))
    if not manifest_path.is_absolute():
        manifest_path = _project_root() / manifest_path
    payload = {
        "body_id": CMU_ARE_BODY_ID,
        "pack_version": pack_version,
        "safety_digest": contract.digest,
        "asset_manifest": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        ),
        "capabilities": [
            "cmu_are.navigate_to_waypoint",
            "cmu_are.exploration_control",
            "cmu_are.stop",
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "CMU_ARE_BODY_ID",
    "CMU_ARE_CARD_SCHEMA",
    "CMU_ARE_EXPLORE_SCHEMA",
    "CMU_ARE_NAV_SCHEMA",
    "CMU_ARE_STOP_SCHEMA",
    "CmuAreContractError",
    "CmuPlace",
    "CmuAreSafetyContract",
    "body_snapshot_hash",
    "default_card_path",
    "default_places_path",
    "load_places",
    "load_safety_contract",
    "resolve_target",
]
