"""Skill activation: bridge installed packages into the runtime (doc §14).

The two registries keep their distinct responsibilities:

- ``SkillLocalRegistry`` / ``installed.lock.json`` = **InstalledSkillIndex**
  (persistent, on disk)
- ``SkillManager.SkillRegistry`` = **ActiveRuntimeRegistry**
  (in-memory, executable)

``SkillLoader`` is the missing link: at startup it converts installed
skill packages into runtime entries so a freshly installed official skill
is visible to the executor — and survives runtime restarts because the
source of truth is on disk, not in memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

logger = logging.getLogger("rosclaw.skill.loader")


class SkillLoader:
    """Loads installed skills from ``$ROSCLAW_HOME/skills`` into a runtime registry."""

    def __init__(self, home: Path | None, runtime_registry: SkillRegistry) -> None:
        self._home = Path(home) if home is not None else get_rosclaw_home()
        self._registry = runtime_registry

    def load_installed(self) -> int:
        """Register every installed skill not already active. Returns count."""
        loaded = 0
        for name, lock_entry in self._read_lockfile().items():
            if self._registry.get_by_name(name) is not None:
                continue  # idempotent: already active in this runtime
            version = str(lock_entry.get("version", ""))
            manifest = self._read_manifest(name, version)
            metadata = manifest.get("metadata", {}) or {}
            execution = manifest.get("execution", {}) or {}
            capability = manifest.get("capability", {}) or {}
            entry = SkillEntry(
                name=name,
                description=str(metadata.get("description", "")),
                skill_type="programmed",
                version=version or "1.0.0",
                metadata={
                    "source": "installed",
                    "trust": lock_entry.get("trust", ""),
                    "package_digest": lock_entry.get("package_digest", ""),
                    "install_dir": str(self._home / "skills" / name / version),
                    "domain": execution.get("domain", ""),
                    "planner": execution.get("planner", {}) or {},
                    "verifier": execution.get("verifier", {}) or {},
                    "capability_id": capability.get("id"),
                    "permissions": manifest.get("permissions", []) or [],
                },
            )
            self._registry.register(entry)
            loaded += 1
            logger.info("activated installed skill %s@%s", name, version)
        return loaded

    def _read_lockfile(self) -> dict:
        lockfile = self._home / "skills" / "installed.lock.json"
        if not lockfile.exists():
            return {}
        try:
            return json.loads(lockfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("installed.lock.json unreadable: %s", exc)
            return {}

    def _read_manifest(self, name: str, version: str) -> dict:
        manifest_path = self._home / "skills" / name / version / "skill.yaml"
        if not manifest_path.exists():
            logger.warning("installed skill %s@%s manifest missing", name, version)
            return {}
        try:
            return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            logger.warning("installed skill %s manifest invalid: %s", name, exc)
            return {}
