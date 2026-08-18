"""CMU ARE out-of-band asset manifest and verification helpers.

The ROS1 source tree is intentionally kept separate from large model, path,
and Gazebo mesh assets.  This module provides a read-only check that can be
used by the CLI and by Docker preflight code without importing ROS libraries.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

ASSET_MANIFEST_VERSION = "rosclaw.cmu_are.assets.v1"
DEFAULT_ASSET_MANIFEST = Path("docs/assets/cmu-are-assets.yaml")
_ASSET_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, field: str) -> Path:
    """Validate a manifest path before joining it to an asset root."""

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def load_asset_manifest(path: str | Path = DEFAULT_ASSET_MANIFEST) -> dict[str, Any]:
    """Load and minimally validate the CMU ARE asset manifest."""

    source = Path(path).expanduser()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != ASSET_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported CMU ARE asset manifest: {source} (expected {ASSET_MANIFEST_VERSION})"
        )
    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise ValueError("CMU ARE asset manifest requires an assets list")
    for item in assets:
        if not isinstance(item, dict):
            raise ValueError("Every CMU ARE asset entry must be a mapping")
        asset_id = str(item.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("Every CMU ARE asset entry requires an asset_id")
        required = {
            "relative_path",
            "source",
            "upstream_version_or_commit",
            "size_bytes",
            "sha256",
            "license",
            "mount_path",
            "required_for",
        }
        missing = sorted(key for key in required if key not in item)
        if missing:
            raise ValueError(f"Asset {asset_id!r} is missing fields: {', '.join(missing)}")
        relative_text = str(item.get("relative_path", "")).strip()
        if not relative_text:
            raise ValueError(f"Asset {asset_id!r} has no relative_path")
        _safe_relative_path(relative_text, field=f"Asset {asset_id!r} relative_path")
        external_relative = item.get("external_relative_path")
        if external_relative is not None:
            _safe_relative_path(
                external_relative,
                field=f"Asset {asset_id!r} external_relative_path",
            )
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ValueError(f"Asset {asset_id!r} has no source")
        if not isinstance(item.get("mount_path"), str) or not item["mount_path"].strip():
            raise ValueError(f"Asset {asset_id!r} has no mount_path")
        required_for = item.get("required_for")
        if not isinstance(required_for, list) or any(
            not isinstance(value, str) or not value.strip() for value in required_for
        ):
            raise ValueError(f"Asset {asset_id!r} required_for must be a list of strings")
        size = item.get("size_bytes")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(
                f"Asset {asset_id!r} size_bytes must be a non-negative integer or null"
            )
        digest = item.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or _ASSET_HASH_RE.fullmatch(digest) is None
        ):
            raise ValueError(f"Asset {asset_id!r} sha256 must be a 64-character hex digest or null")
    return raw


def verify_assets(
    *,
    project_root: str | Path,
    manifest_path: str | Path = DEFAULT_ASSET_MANIFEST,
    required_for: set[str] | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe verification report without changing the workspace."""

    root = Path(project_root).expanduser().resolve()
    external_root = Path(asset_root).expanduser().resolve() if asset_root is not None else None
    manifest = load_asset_manifest(root / manifest_path)
    selected = set(required_for or ())
    entries: list[dict[str, Any]] = []
    for raw in manifest["assets"]:
        requirements = {str(value) for value in raw.get("required_for", [])}
        if selected and not requirements.intersection(selected):
            continue
        relative_text = str(raw["relative_path"])
        relative = Path(relative_text)
        source_relative = relative
        if external_root is not None:
            # ``relative_path`` is the immutable in-repository/canonical path.
            # External workspaces are allowed to use a different layout (the
            # CMU ARE checkout has ``src/ARiADNE2-ROS-Planner/src`` and
            # ``src/autonomous_exploration_development_environment/src``).
            # Keep the mapping explicit in the manifest instead of guessing
            # from arbitrary host paths.
            external_text = raw.get("external_relative_path")
            if external_text is not None:
                source_relative = _safe_relative_path(
                    external_text,
                    field=f"Asset {raw['asset_id']!r} external_relative_path",
                )
            elif relative.parts[:1] == ("third_party",):
                source_relative = Path(*relative.parts[1:])

            candidate_root = external_root
            candidate = (external_root / source_relative).resolve()

            # It is common for an operator to point CMU_ARE_ASSET_ROOT at the
            # workspace's ``src`` directory rather than its parent.  Accept
            # that one unambiguous spelling while preserving the manifest's
            # explicit source mapping.
            if not candidate.exists() and source_relative.parts[:1] == ("src",):
                src_relative = Path(*source_relative.parts[1:])
                src_candidate = (external_root / src_relative).resolve()
                if src_candidate.exists():
                    candidate = src_candidate
                    source_relative = src_relative
        else:
            candidate_root = root
            candidate = (root / relative).resolve()
        try:
            candidate.relative_to(candidate_root)
        except ValueError:
            entries.append(
                {
                    "asset_id": raw["asset_id"],
                    "relative_path": str(relative),
                    "source_relative_path": str(source_relative),
                    "status": "invalid_path",
                    "message": "asset path escapes project root",
                }
            )
            continue

        # ``Path`` strips a trailing slash, so retain the manifest spelling to
        # distinguish directory mounts from regular files.
        expects_directory = relative_text.endswith("/")
        exists = candidate.is_dir() if expects_directory else candidate.is_file()
        item: dict[str, Any] = {
            "asset_id": raw["asset_id"],
            "relative_path": str(relative),
            "source_relative_path": str(source_relative),
            "resolved_path": str(candidate),
            "status": "ok" if exists else "missing",
            "required_for": sorted(requirements),
        }
        if exists and candidate.is_file():
            item["size_bytes"] = candidate.stat().st_size
            item["sha256"] = _sha256(candidate)
            expected_size = raw.get("size_bytes")
            expected_hash = raw.get("sha256")
            if expected_size is not None and item["size_bytes"] != expected_size:
                item["status"] = "size_mismatch"
            if expected_hash and item["sha256"] != expected_hash:
                item["status"] = "sha256_mismatch"
        entries.append(item)

    failed = [entry for entry in entries if entry["status"] != "ok"]
    return {
        "schema_version": ASSET_MANIFEST_VERSION,
        "manifest": str((root / manifest_path).resolve()),
        "project_root": str(root),
        "asset_root": str(external_root)
        if external_root is not None
        else str(root / "third_party"),
        "selected_requirements": sorted(selected),
        "ok": not failed,
        "assets": entries,
        "missing_or_invalid": failed,
    }


__all__ = [
    "ASSET_MANIFEST_VERSION",
    "DEFAULT_ASSET_MANIFEST",
    "load_asset_manifest",
    "verify_assets",
]
