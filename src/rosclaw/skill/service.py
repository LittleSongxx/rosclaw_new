"""Unified skill facade (Skill Runtime 2.0, doc §15).

CLI, MCP, the native agent and external harnesses all go through
``SkillService`` instead of touching ``SkillLocalRegistry`` /
``SkillManager.SkillRegistry`` / hub clients / builtins directly.

PR-4 scope: discovery + acquisition + activation (search / resolve /
inspect / ensure_installed / list_active). Execution (plan / invoke /
verify / rollback) lands with the HostOps plane (PR-5+).
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.skill.catalog_service import SkillCatalogService
from rosclaw.skill.installer import InstallReceipt, SkillInstaller
from rosclaw.skill.loader import SkillLoader
from rosclaw.skill.resolver import CapabilityResolution, CapabilityResolver
from rosclaw.skill.sources import CatalogHit
from rosclaw.skill_manager.registry import SkillRegistry


class SkillService:
    """One facade over catalog, resolver, installer and runtime registry."""

    def __init__(self, home: Path | None = None, runtime_registry: SkillRegistry | None = None) -> None:
        self._home = Path(home) if home is not None else get_rosclaw_home()
        self._catalog = SkillCatalogService.default(self._home)
        self._resolver = CapabilityResolver(self._catalog)
        self._installer = SkillInstaller(self._home, self._catalog)
        self._runtime_registry = runtime_registry or SkillRegistry()
        SkillLoader(self._home, self._runtime_registry).load_installed()

    # Discovery plane -------------------------------------------------

    def search(self, query: str, context: dict | None = None) -> list[CatalogHit]:
        return self._catalog.search(query, context=context)

    def resolve(self, intent: str, context: dict | None = None) -> CapabilityResolution:
        return self._resolver.resolve(intent, context=context)

    def inspect(self, ref: str) -> CatalogHit | None:
        return self._catalog.get(ref)

    # Acquisition plane -------------------------------------------------

    def ensure_installed(self, ref: str) -> InstallReceipt:
        """doc §28: official (T0/T1) skills may auto-acquire; the rest need
        an explicit install call from the operator."""
        if self._installer.lockfile.exists():
            data = json.loads(self._installer.lockfile.read_text(encoding="utf-8"))
            if ref in data:
                entry = data[ref]
                return InstallReceipt(
                    name=ref,
                    version=str(entry.get("version", "")),
                    package_digest=str(entry.get("package_digest", "")),
                    install_dir=self._home / "skills" / ref / str(entry.get("version", "")),
                    trust=str(entry.get("trust", "")),
                )
        receipt = self._installer.install(ref)
        SkillLoader(self._home, self._runtime_registry).load_installed()
        return receipt

    # Runtime plane -------------------------------------------------

    def list_active(self) -> list[dict]:
        """Skills active in the runtime registry (what MCP should expose)."""
        return [
            {
                "name": e.name,
                "version": e.version,
                "description": e.description,
                "domain": (e.metadata or {}).get("domain", ""),
                "trust": (e.metadata or {}).get("trust", ""),
            }
            for e in self._runtime_registry.list_skills(return_entries=True)
        ]

    @property
    def runtime_registry(self) -> SkillRegistry:
        return self._runtime_registry
