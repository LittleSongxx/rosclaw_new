"""Deterministic capability resolver (Skill Runtime 2.0, doc §4/§5/§6).

The agent expresses an intent ("帮我安装 ROS2"); the resolver maps it to a
capability and a concrete skill implementation — without the agent ever
guessing a skill name, and without requiring an LLM. Ambiguous rerank by
a model may be layered on top later; the base path stays deterministic.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass, field
from pathlib import Path

from rosclaw.skill.catalog_service import SkillCatalogService
from rosclaw.skill.sources import CatalogHit

_INTENT_MATCH_SCORE = 100.0


@dataclass
class CapabilityResolution:
    """Result of resolving a natural-language intent (doc §5)."""

    capability: str | None
    selected_skill: str | None
    confidence: float
    source: str | None = None
    installed: bool = False
    compatible: bool = False
    reasons: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "selected_skill": self.selected_skill,
            "confidence": round(self.confidence, 2),
            "source": self.source,
            "installed": self.installed,
            "compatible": self.compatible,
            "reasons": list(self.reasons),
            "candidates": self.candidates,
        }


def detect_host_context() -> dict[str, str]:
    """Best-effort host facts for compatibility checks (doc §34 light)."""
    os_id, os_version = "", ""
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("ID="):
                os_id = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("VERSION_ID="):
                os_version = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)
    return {"os": os_id, "os_version": os_version, "arch": arch}


class CapabilityResolver:
    """intent → capability → skill implementation (deterministic)."""

    def __init__(self, catalog: SkillCatalogService | None = None) -> None:
        self._catalog = catalog or SkillCatalogService.default()

    def resolve(self, intent: str, context: dict | None = None) -> CapabilityResolution:
        host = context or detect_host_context()
        hits = self._catalog.search(intent)
        candidates = [
            {"name": h.name, "score": round(h.score, 2), "source": h.source}
            for h in hits[:5]
        ]
        for hit in hits:
            compatible, compat_reasons = self._check_compatibility(hit, host)
            if hit.score < _INTENT_MATCH_SCORE:
                # Below intent match: never auto-select (doc §6 determinism).
                break
            if not compatible:
                continue
            reasons = ["intent_match", *compat_reasons]
            if hit.official:
                reasons.append("official")
            if hit.verification_status:
                reasons.append(hit.verification_status)
            if hit.installed:
                reasons.append("installed")
            return CapabilityResolution(
                capability=hit.capability_id,
                selected_skill=hit.name,
                confidence=0.98,
                source=hit.source,
                installed=hit.installed,
                compatible=True,
                reasons=reasons,
                candidates=candidates,
            )
        # Nothing selectable: report the best-effort confidence honestly.
        best = hits[0].score if hits else 0.0
        return CapabilityResolution(
            capability=None,
            selected_skill=None,
            confidence=min(0.89, best / 200.0),
            compatible=False,
            reasons=["no_intent_match"] if not hits else ["below_selection_threshold"],
            candidates=candidates,
        )

    @staticmethod
    def _check_compatibility(hit: CatalogHit, host: dict[str, str]) -> tuple[bool, list[str]]:
        compat = hit.compatibility or {}
        reasons: list[str] = []
        os_list = [str(o).lower() for o in compat.get("os", []) or []]
        if os_list:
            if host.get("os", "").lower() not in os_list:
                return False, [f"os_{host.get('os', 'unknown')}_unsupported"]
            reasons.append(f"{host['os']}_{host.get('os_version', '')}_supported")
        arch_list = [str(a).lower() for a in compat.get("architectures", []) or []]
        if arch_list:
            host_arch = host.get("arch", "").lower()
            aliases = {host_arch, re.sub(r"^(x86_64)$", "amd64", host_arch)}
            if not aliases & set(arch_list):
                return False, [f"arch_{host_arch}_unsupported"]
            reasons.append(f"{host_arch}_supported")
        return True, reasons
