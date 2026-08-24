"""Skill catalog sources (Skill Runtime 2.0, doc §9/§10).

Four unified sources feed the ``SkillCatalogService``:

- ``BuiltinCatalogSource``   — in-package skills (``rosclaw.skill.builtins``)
- ``InstalledCatalogSource`` — skills installed into ``$ROSCLAW_HOME/skills``
- ``OfficialCatalogSource``  — the official ``ros-claw/skills`` registry,
  fetched once and cached at ``$ROSCLAW_HOME/cache/skills/catalog.json``
  (offline falls back to the cache)
- ``WorkspaceCatalogSource`` — project-local ``./skills/*/skill.yaml``

A source that fails (e.g. network down with no cache) is skipped by the
service — one broken source must never kill the whole catalog.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import get_rosclaw_home

logger = logging.getLogger("rosclaw.skill.sources")

DEFAULT_OFFICIAL_REGISTRY_URL = (
    "https://raw.githubusercontent.com/ros-claw/skills/main/registry/skills.json"
)
REGISTRY_URL_ENV = "ROSCLAW_SKILLS_REGISTRY_URL"
_FETCH_TIMEOUT = 15.0


class CatalogSourceError(Exception):
    """A catalog source could not produce entries (and has no cache)."""


@dataclass
class CatalogHit:
    """One catalog entry, normalized across sources."""

    name: str  # namespaced ("ros-claw/ros_install") or bare builtin name
    version: str = ""
    description: str = ""
    display_name: str = ""
    source: str = "official"  # builtin | installed | official | workspace
    official: bool = False
    installable: bool = False
    installed: bool = False
    verification_status: str = ""
    tags: list[str] = field(default_factory=list)
    capability_id: str | None = None
    intents: dict[str, list[str]] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "display_name": self.display_name,
            "source": self.source,
            "official": self.official,
            "installable": self.installable,
            "installed": self.installed,
            "verification_status": self.verification_status,
            "tags": list(self.tags),
            "capability_id": self.capability_id,
            "score": round(self.score, 2),
        }


class BuiltinCatalogSource:
    """In-package builtin skills (always available, T0 trust)."""

    name = "builtin"

    def entries(self) -> list[CatalogHit]:
        from rosclaw.skill.builtins import list_builtin_skills

        hits = []
        for info in list_builtin_skills():
            hits.append(
                CatalogHit(
                    name=info["name"],
                    version=str(info.get("version", "")),
                    description=str(info.get("description", "")),
                    display_name=str(info.get("display_name", "")),
                    source="builtin",
                    official=True,
                    installable=False,
                    installed=True,  # builtins ship with the package
                    verification_status="builtin",
                    raw=info,
                )
            )
        return hits


class InstalledCatalogSource:
    """Skills installed into ``$ROSCLAW_HOME/skills`` (lockfile + legacy hub)."""

    name = "installed"

    def __init__(self, home: Path | None = None) -> None:
        self._home = home

    def _resolve_home(self) -> Path:
        return self._home or get_rosclaw_home()

    def entries(self) -> list[CatalogHit]:
        home = self._resolve_home()
        hits: list[CatalogHit] = []
        lockfile = home / "skills" / "installed.lock.json"
        if lockfile.exists():
            try:
                data = json.loads(lockfile.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("installed.lock.json unreadable: %s", exc)
                data = {}
            for name, entry in data.items():
                version = str(entry.get("version", ""))
                manifest = self._read_manifest(home, name, version)
                hits.append(
                    CatalogHit(
                        name=name,
                        version=version,
                        description=manifest.get("description", ""),
                        source="installed",
                        official=bool(entry.get("trust", "").startswith("official")),
                        installable=False,
                        installed=True,
                        verification_status=str(entry.get("verification_status", "")),
                        tags=manifest.get("tags", []),
                        capability_id=manifest.get("capability_id"),
                        intents=manifest.get("intents", {}),
                        compatibility=manifest.get("compatibility", {}),
                        raw=entry,
                    )
                )
        return hits

    @staticmethod
    def _read_manifest(home: Path, name: str, version: str) -> dict[str, Any]:
        """Best-effort manifest read for an installed skill (tolerates v1/v2)."""
        manifest_path = home / "skills" / name / version / "skill.yaml"
        if not manifest_path.exists():
            return {}
        try:
            import yaml

            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 — catalog must stay robust
            logger.warning("installed manifest unreadable %s: %s", manifest_path, exc)
            return {}
        metadata = data.get("metadata", {}) or {}
        capability = data.get("capability", {}) or {}
        return {
            "description": str(metadata.get("description", "")),
            "tags": list(metadata.get("tags", []) or []),
            "capability_id": capability.get("id"),
            "intents": capability.get("intents", {}) or {},
            "compatibility": data.get("compatibility", {}) or {},
        }


class OfficialCatalogSource:
    """The official ``ros-claw/skills`` registry with local cache (doc §10).

    First use fetches the registry into ``$ROSCLAW_HOME/cache/skills/
    catalog.json``; when the network is unavailable the cache is used.
    Registry signature verification lands with the installer (PR-3, doc §13).
    """

    name = "official"

    def __init__(
        self,
        home: Path | None = None,
        registry_url: str | None = None,
        fetch_timeout: float = _FETCH_TIMEOUT,
    ) -> None:
        self._home = home
        self._registry_url = registry_url
        self._fetch_timeout = fetch_timeout

    def _resolve_home(self) -> Path:
        return self._home or get_rosclaw_home()

    def _resolve_url(self) -> str:
        return (
            self._registry_url
            or os.environ.get(REGISTRY_URL_ENV)
            or DEFAULT_OFFICIAL_REGISTRY_URL
        )

    @property
    def cache_file(self) -> Path:
        return self._resolve_home() / "cache" / "skills" / "catalog.json"

    def entries(self) -> list[CatalogHit]:
        payload = self._load_payload()
        hits = []
        for raw in payload.get("skills", []):
            capability = raw.get("capability", {}) or {}
            hits.append(
                CatalogHit(
                    name=str(raw.get("name", "")),
                    version=str(raw.get("version", "")),
                    description=str(raw.get("description", "")).strip(),
                    display_name=str(raw.get("display_name", "")),
                    source="official",
                    official=bool(raw.get("official", False)),
                    installable=bool(raw.get("installable", False)),
                    installed=False,  # overlay happens in the service merge
                    verification_status=str(raw.get("verification_status", "")),
                    tags=list(raw.get("tags", []) or []),
                    capability_id=capability.get("id"),
                    intents=capability.get("intents", {}) or {},
                    compatibility=raw.get("compatibility", {}) or {},
                    raw=raw,
                )
            )
        return [h for h in hits if h.name]

    def _load_payload(self) -> dict[str, Any]:
        url = self._resolve_url()
        try:
            text = self._fetch(url)
        except CatalogSourceError:
            cache = self.cache_file
            if cache.exists():
                logger.info("official registry unreachable; using cache %s", cache)
                return json.loads(cache.read_text(encoding="utf-8"))
            raise
        payload = json.loads(text)
        self._write_cache(text)
        return payload

    def _fetch(self, url: str) -> str:
        try:
            with urllib.request.urlopen(url, timeout=self._fetch_timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise CatalogSourceError(f"official registry fetch failed: {exc}") from exc

    def _write_cache(self, text: str) -> None:
        cache = self.cache_file
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: no half-written cache may be served later.
        fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=".catalog-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, cache)
        except OSError:
            with suppress(OSError):
                os.unlink(tmp)
            raise


class WorkspaceCatalogSource:
    """Project-local skills under ``./skills/*/skill.yaml`` (T4 local_dev)."""

    name = "workspace"

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._dir = workspace_dir

    def entries(self) -> list[CatalogHit]:
        base = self._dir or (Path.cwd() / "skills")
        if not base.is_dir():
            return []
        hits = []
        for manifest_path in sorted(base.glob("*/skill.yaml")):
            try:
                import yaml

                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001 — skip broken manifests
                logger.warning("workspace manifest unreadable %s: %s", manifest_path, exc)
                continue
            metadata = data.get("metadata", {}) or {}
            capability = data.get("capability", {}) or {}
            name = metadata.get("name", manifest_path.parent.name)
            namespace = metadata.get("namespace")
            hits.append(
                CatalogHit(
                    name=f"{namespace}/{name}" if namespace else str(name),
                    version=str(metadata.get("version", "")),
                    description=str(metadata.get("description", "")),
                    source="workspace",
                    official=False,
                    installable=False,
                    installed=True,
                    verification_status="local_dev",
                    tags=list(metadata.get("tags", []) or []),
                    capability_id=capability.get("id"),
                    intents=capability.get("intents", {}) or {},
                    compatibility=data.get("compatibility", {}) or {},
                    raw={"path": str(manifest_path.parent)},
                )
            )
        return hits
