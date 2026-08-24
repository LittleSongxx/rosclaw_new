"""PR-2 GREEN — the official catalog is a real runtime source (doc §9/§10).

(Was PR-1 RED: ``skill search`` had no query and never consulted the
official ``ros-claw/skills`` registry. PR-2 wires the unified catalog.)
"""

from __future__ import annotations

import argparse
import json

import pytest

from rosclaw.skill.cli import add_skill_hub_parsers, cmd_skill_search


def _search_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    parser = argparse.ArgumentParser()
    add_skill_hub_parsers(parser.add_subparsers())
    args = parser.parse_args(list(argv))
    rc = cmd_skill_search(args)
    return rc, capsys.readouterr().out


class TestCliSkillSearch:
    def test_search_accepts_query_and_finds_official_ros_install(
        self, official_registry, rosclaw_home, capsys
    ):
        """doc §10: `rosclaw skill search "install ros2"` must hit the official catalog."""
        rc, out = _search_cli(capsys, "search", "install ros2")
        assert rc == 0
        assert "ros-claw/ros_install" in out

    def test_search_output_shows_trust_and_install_state(
        self, official_registry, rosclaw_home, capsys
    ):
        rc, out = _search_cli(capsys, "search", "install ros2", "--json")
        assert rc == 0
        hits = json.loads(out)["results"]
        hit = next(h for h in hits if h["name"] == "ros-claw/ros_install")
        assert hit["source"] == "official"
        assert hit["official"] is True
        assert hit["version"] == "0.2.0"
        assert hit["installed"] is False


class TestSkillCatalogService:
    """doc §9: one service unifying Builtin/Installed/Official/Workspace."""

    def test_search_unifies_sources_and_ranks_official_hit(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.catalog_service import SkillCatalogService

        service = SkillCatalogService.default()
        hits = service.search("install ros2")
        assert hits, "catalog search returned nothing for an official skill intent"
        top = hits[0]
        assert top.name == "ros-claw/ros_install"
        assert top.source == "official"
        assert top.official is True
        assert top.installable is True

    def test_catalog_hit_carries_verification_status(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.catalog_service import SkillCatalogService

        hits = SkillCatalogService.default().search("install ros2")
        hit = next(h for h in hits if h.name == "ros-claw/ros_install")
        # doc §40: evidence level comes from the registry, not self-declared YAML.
        assert hit.verification_status == "host_matrix_verified"

    def test_official_catalog_is_cached_and_works_offline(
        self, official_registry, rosclaw_home, monkeypatch, capsys
    ):
        """doc §10: first search pulls the registry into
        ``$ROSCLAW_HOME/cache/skills/catalog.json``; offline falls back to cache."""
        from rosclaw.skill.catalog_service import SkillCatalogService

        service = SkillCatalogService.default()
        assert service.search("install ros2"), "first (online) search must populate cache"

        cache_file = rosclaw_home / "cache" / "skills" / "catalog.json"
        assert cache_file.exists(), "registry was not cached under ROSCLAW_HOME"

        # Break the network: any further fetch must fail, cache must serve.
        monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", "http://127.0.0.1:1/nope.json")
        offline_hits = SkillCatalogService.default().search("install ros2")
        assert any(h.name == "ros-claw/ros_install" for h in offline_hits)
