"""Unified skill catalog service (Skill Runtime 2.0, doc §9).

One entry point over Builtin / Installed / Official / Workspace sources.
Search is fully deterministic (doc §6): intent-phrase match, tag match,
description token overlap, then trust / evidence / installed bonuses —
no LLM required to find a skill.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rosclaw.skill.sources import (
    BuiltinCatalogSource,
    CatalogHit,
    InstalledCatalogSource,
    OfficialCatalogSource,
    WorkspaceCatalogSource,
)

logger = logging.getLogger("rosclaw.skill.catalog_service")

_INTENT_MATCH_SCORE = 100.0
_VERIFIED_STATUSES = {
    "official_verified",
    "host_matrix_verified",
    "sandbox_validated",
    "hardware_verified",
    "field_verified",
}


class SkillCatalogService:
    """Unified, deterministic search across all skill sources."""

    def __init__(self, sources: list) -> None:
        self._sources = list(sources)

    @classmethod
    def default(cls, home: Path | None = None) -> SkillCatalogService:
        """The standard four-source catalog (doc §9)."""
        return cls(
            [
                InstalledCatalogSource(home),
                BuiltinCatalogSource(),
                OfficialCatalogSource(home),
                WorkspaceCatalogSource(),
            ]
        )

    def search(self, query: str, context: dict | None = None, limit: int = 20) -> list[CatalogHit]:
        """Rank all sources' entries against ``query`` (deterministic, doc §6)."""
        hits = self._collect()
        for hit in hits:
            hit.score = self._score(query, hit)
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: (-h.score, not h.official, h.name))
        return hits[:limit]

    def get(self, name: str) -> CatalogHit | None:
        """Exact-name lookup across all sources (installed overlay applied)."""
        for hit in self._collect():
            if hit.name == name:
                return hit
        return None

    def _collect(self) -> list[CatalogHit]:
        """Gather entries from every source; a broken source is skipped.

        Entries with the same name are merged: ``installed``/``official``
        flags overlay so an installed official skill reads as both.
        """
        merged: dict[str, CatalogHit] = {}
        for source in self._sources:
            try:
                entries = source.entries()
            except Exception as exc:  # noqa: BLE001 — source isolation
                logger.info("catalog source %s unavailable: %s", source.name, exc)
                continue
            for hit in entries:
                existing = merged.get(hit.name)
                if existing is None:
                    merged[hit.name] = hit
                    continue
                existing.installed = existing.installed or hit.installed
                existing.official = existing.official or hit.official
                if hit.source == "installed":
                    # Local availability wins as the representative entry,
                    # but keep official trust/evidence metadata.
                    hit.official = existing.official
                    hit.verification_status = hit.verification_status or existing.verification_status
                    merged[hit.name] = hit
        return list(merged.values())

    @staticmethod
    def _score(query: str, hit: CatalogHit) -> float:
        q = query.lower().strip()
        score = 0.0
        # Intent phrase match (multilingual): phrase contained in the query.
        for phrases in hit.intents.values():
            if any(p.lower() and p.lower() in q for p in phrases):
                score += _INTENT_MATCH_SCORE
                break
        # Name match (e.g. the user typed the skill name).
        name_tokens = set(re.findall(r"[a-z0-9]+", hit.name.lower()))
        q_tokens = set(re.findall(r"[a-z0-9]+", q))
        if name_tokens and name_tokens <= q_tokens:
            score += 50.0
        # Tag match: token hit or CJK substring.
        matched_tags = [
            t for t in hit.tags if t.lower() in q_tokens or (t and t.lower() in q)
        ]
        score += 10.0 * len(matched_tags)
        # Description token overlap.
        desc_tokens = set(re.findall(r"[a-z0-9]+", hit.description.lower()))
        score += float(len(q_tokens & desc_tokens))
        # Trust / evidence / local availability bonuses.
        if hit.official:
            score += 15.0
        if hit.verification_status in _VERIFIED_STATUSES:
            score += 10.0
        if hit.installed:
            score += 5.0
        return score
