"""ROS Connector - Embodiment card loading.

An embodiment card (``specs/<robot_id>.yaml``,
``schema_version: rosclaw.embodiment_card.v1``) is the declarative description
of one robot body: which ROS interfaces are preferred, which are discouraged,
and what safety envelope applies. This module is the single public entry point
for reading them, so callers do not each re-implement path resolution.

Deliberately free of rclpy/rospy imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SPECS_DIR = Path(__file__).parent / "specs"

EMBODIMENT_CARD_SCHEMA = "rosclaw.embodiment_card.v1"


class EmbodimentCardError(Exception):
    """Raised when an embodiment card is missing or malformed."""


@dataclass(frozen=True)
class EmbodimentCard:
    """A parsed embodiment card.

    Unknown top-level keys are preserved in :attr:`raw` so forward-compatible
    fields stay reachable without a schema bump.
    """

    robot_id: str
    schema_version: str = EMBODIMENT_CARD_SCHEMA
    aliases: tuple[str, ...] = ()
    body_type: str = ""
    ros_version: int | None = None
    ros_distro: str = ""
    preferred_interfaces: tuple[dict[str, Any], ...] = ()
    observation_interfaces: tuple[dict[str, Any], ...] = ()
    discouraged_interfaces: tuple[dict[str, Any], ...] = ()
    safety_defaults: dict[str, Any] = field(default_factory=dict)
    operational_limits: dict[str, Any] = field(default_factory=dict)
    workspace_boundaries: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    recovery_capabilities: tuple[str, ...] = ()
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def limit(self, name: str, default: float | None = None) -> float | None:
        """Read a numeric value from ``operational_limits``.

        Returns ``default`` when the key is absent or not a finite number.
        """
        value = self.operational_limits.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)

    def interface_for(self, capability_id: str) -> dict[str, Any] | None:
        """Look up a declared interface by capability id."""
        for group in (
            self.preferred_interfaces,
            self.observation_interfaces,
            self.discouraged_interfaces,
        ):
            for item in group:
                if item.get("capability_id") == capability_id:
                    return dict(item)
        return None

    def is_discouraged(self, ros_name: str) -> bool:
        """True when ``ros_name`` is listed under ``discouraged_interfaces``."""
        return any(item.get("ros_name") == ros_name for item in self.discouraged_interfaces)

    def to_dict(self) -> dict[str, Any]:
        """Return the card as the plain mapping the compilers expect."""
        return dict(self.raw)


def _as_tuple_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def parse_embodiment_card(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> EmbodimentCard:
    """Build an :class:`EmbodimentCard` from an already-loaded mapping."""
    if not isinstance(data, dict):
        raise EmbodimentCardError("Embodiment card must be a mapping")

    where = f" ({source_path})" if source_path else ""

    robot_id = str(data.get("robot_id", "")).strip()
    if not robot_id:
        raise EmbodimentCardError(f"Embodiment card is missing 'robot_id'{where}")

    # An absent schema_version is treated as v1 so hand-written cards stay easy,
    # but a declared mismatch is rejected rather than silently misread.
    declared_schema = str(data.get("schema_version", EMBODIMENT_CARD_SCHEMA)).strip()
    if declared_schema != EMBODIMENT_CARD_SCHEMA:
        raise EmbodimentCardError(
            f"Unsupported embodiment card schema_version {declared_schema!r}"
            f" (expected {EMBODIMENT_CARD_SCHEMA!r}){where}"
        )

    ros_version = data.get("ros_version")
    if isinstance(ros_version, bool) or not isinstance(ros_version, int):
        ros_version = None

    return EmbodimentCard(
        robot_id=robot_id,
        schema_version=declared_schema,
        aliases=_as_str_tuple(data.get("aliases")),
        body_type=str(data.get("body_type", "")),
        ros_version=ros_version,
        ros_distro=str(data.get("ros_distro", "")),
        preferred_interfaces=_as_tuple_of_dicts(data.get("preferred_interfaces")),
        observation_interfaces=_as_tuple_of_dicts(data.get("observation_interfaces")),
        discouraged_interfaces=_as_tuple_of_dicts(data.get("discouraged_interfaces")),
        safety_defaults=_as_dict(data.get("safety_defaults")),
        operational_limits=_as_dict(data.get("operational_limits")),
        workspace_boundaries=_as_dict(data.get("workspace_boundaries")),
        preconditions=_as_dict(data.get("preconditions")),
        recovery_capabilities=_as_str_tuple(data.get("recovery_capabilities")),
        source_path=source_path,
        raw=dict(data),
    )


def find_embodiment_card(robot_id: str, *, specs_dir: Path | None = None) -> Path | None:
    """Return the path to a robot's embodiment card, or ``None``.

    Filename matching comes first: the id verbatim and with ``_``/``-`` swapped,
    for both extensions. Only if that misses do we scan the directory and match
    against each card's declared ``robot_id``/``aliases``, since scanning has to
    parse every file.
    """
    base = specs_dir or SPECS_DIR
    candidates = {robot_id, robot_id.replace("_", "-"), robot_id.replace("-", "_")}
    for name in candidates:
        for ext in (".yaml", ".yml"):
            path = base / f"{name}{ext}"
            if path.is_file():
                return path

    if not base.is_dir():
        return None
    wanted = robot_id.strip().lower()
    if not wanted:
        return None
    for path in sorted(base.iterdir()):
        if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
            continue
        try:
            card = load_embodiment_card_file(path)
        except EmbodimentCardError:
            continue
        names = {card.robot_id.lower(), *(alias.lower() for alias in card.aliases)}
        if wanted in names:
            return path
    return None


def load_embodiment_card(
    robot_id: str,
    *,
    specs_dir: Path | None = None,
) -> EmbodimentCard | None:
    """Load a robot's embodiment card, or ``None`` when it does not exist.

    Raises:
        EmbodimentCardError: The file exists but is not a valid card.
    """
    path = find_embodiment_card(robot_id, specs_dir=specs_dir)
    if path is None:
        return None
    return load_embodiment_card_file(path)


def load_embodiment_card_file(path: str | Path) -> EmbodimentCard:
    """Load and parse an embodiment card from an explicit path."""
    card_path = Path(path)
    try:
        text = card_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EmbodimentCardError(f"Could not read embodiment card {card_path}: {exc}") from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise EmbodimentCardError(f"Invalid YAML in embodiment card {card_path}: {exc}") from exc

    return parse_embodiment_card(data, source_path=card_path)


def list_embodiment_cards(*, specs_dir: Path | None = None) -> list[str]:
    """Return the sorted robot ids that have an embodiment card on disk."""
    base = specs_dir or SPECS_DIR
    if not base.is_dir():
        return []
    ids: set[str] = set()
    for path in base.iterdir():
        if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
            continue
        try:
            card = load_embodiment_card_file(path)
        except EmbodimentCardError:
            continue
        ids.add(card.robot_id)
    return sorted(ids)


__all__ = [
    "EMBODIMENT_CARD_SCHEMA",
    "SPECS_DIR",
    "EmbodimentCard",
    "EmbodimentCardError",
    "find_embodiment_card",
    "list_embodiment_cards",
    "load_embodiment_card",
    "load_embodiment_card_file",
    "parse_embodiment_card",
]
