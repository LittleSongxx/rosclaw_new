"""Unified skill facade (Skill Runtime 2.0, doc §15).

CLI, MCP, the native agent and external harnesses all go through
``SkillService`` instead of touching ``SkillLocalRegistry`` /
``SkillManager.SkillRegistry`` / hub clients / builtins directly.

PR-4 scope: discovery + acquisition + activation (search / resolve /
inspect / ensure_installed / list_active). Execution (plan / invoke /
verify / rollback) lands with the HostOps plane (PR-5+).
"""

from __future__ import annotations

import importlib.util
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

    # Execution plane -------------------------------------------------

    def find_by_capability(self, capability_id: str) -> CatalogHit | None:
        """Locate the best skill implementing a capability (installed first)."""
        installed_hit: CatalogHit | None = None
        for hit in self._catalog.search(capability_id):
            if hit.capability_id != capability_id:
                continue
            if hit.installed:
                installed_hit = installed_hit or hit
            else:
                return installed_hit or hit
        return installed_hit

    def plan(self, ref: str, args: dict | None = None) -> dict:
        """Build the skill's ExecutionPlan from the detected host state.

        The plan is validated against HostOps policy and bound to a plan
        hash by the caller (CLI ``skill run`` / MCP ``invoke_capability``).
        """
        return build_skill_plan(self._home, ref, args or {})

    @property
    def runtime_registry(self) -> SkillRegistry:
        return self._runtime_registry


class SkillPlanError(Exception):
    """The skill's plan could not be built or is malformed."""


def build_skill_plan(home: Path, ref: str, skill_args: dict) -> dict:
    """Load an installed skill's planner entrypoint and produce the plan.

    Trust basis: the package was digest-pinned at install time; its
    *output* is policy-checked by the HostOps gate before anything runs.
    """
    import yaml

    from rosclaw.hostops.models import make_plan
    from rosclaw.skill.resolver import detect_host_context

    lockfile = home / "skills" / "installed.lock.json"
    installed: dict = {}
    if lockfile.exists():
        try:
            installed = json.loads(lockfile.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            installed = {}
    if ref not in installed:
        raise SkillPlanError(
            f"skill {ref} is not installed; run `rosclaw skill install {ref}` first"
        )
    version = str(installed[ref].get("version", ""))
    install_dir = home / "skills" / ref / version
    manifest_path = install_dir / "skill.yaml"
    if not manifest_path.exists():
        raise SkillPlanError(f"installed skill {ref}@{version} has no skill.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    execution = manifest.get("execution", {}) or {}
    planner_spec = (execution.get("planner", {}) or {}).get("entrypoint")
    if not planner_spec:
        raise SkillPlanError(f"skill {ref} declares no execution.planner entrypoint")
    module_name, _, func_name = planner_spec.partition(":")
    if not module_name or not func_name:
        raise SkillPlanError(f"invalid planner entrypoint {planner_spec!r}")
    entrypoint_path = install_dir / module_name
    if not entrypoint_path.exists():
        raise SkillPlanError(f"planner module {module_name} missing in {install_dir}")

    spec = importlib.util.spec_from_file_location(f"rosclaw_skill_{ref}", entrypoint_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    planner_fn = getattr(module, func_name, None)
    if not callable(planner_fn):
        raise SkillPlanError(f"planner {planner_spec} is not callable")

    context = detect_host_context()
    raw_plan = planner_fn(context, skill_args)
    if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("operations"), list):
        raise SkillPlanError("planner must return a dict with an operations list")

    host_target = {
        "os": context.get("os", ""),
        "version": context.get("os_version", ""),
        "arch": context.get("arch", ""),
    }
    return make_plan(
        skill=raw_plan.get("skill") or f"{ref}@{version}",
        domain=raw_plan.get("domain") or execution.get("domain", "host"),
        target=raw_plan.get("target") or host_target,
        operations=raw_plan["operations"],
    )
