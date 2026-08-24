"""PR-2 GREEN — deterministic CapabilityResolver (doc §4/§5/§6).

The agent never guesses a skill name, and resolution works without an
LLM: intent → capability → skill implementation, ranked by deterministic
signals (intent match, compatibility, trust, evidence).

(Was PR-1 RED; PR-2 implements ``rosclaw.skill.resolver``.)
"""

from __future__ import annotations


def _resolver():
    from rosclaw.skill.catalog_service import SkillCatalogService
    from rosclaw.skill.resolver import CapabilityResolver

    return CapabilityResolver(catalog=SkillCatalogService.default())


class TestCapabilityResolver:
    def test_resolve_chinese_intent_install_ros2(self, official_registry, rosclaw_home):
        """doc §27/§50: the golden sentence must resolve without naming the skill."""
        r = _resolver().resolve("帮我安装 ROS2")
        assert r.capability == "environment.install.ros"
        assert r.selected_skill == "ros-claw/ros_install"
        assert r.source == "official"
        assert r.compatible is True
        assert r.confidence >= 0.9
        assert "intent_match" in r.reasons

    def test_resolve_english_intent_install_ros(self, official_registry, rosclaw_home):
        r = _resolver().resolve("install ROS 2")
        assert r.capability == "environment.install.ros"
        assert r.selected_skill == "ros-claw/ros_install"

    def test_resolution_reports_install_state(self, official_registry, rosclaw_home):
        """doc §28: resolver output drives auto-acquisition of official skills."""
        r = _resolver().resolve("帮我安装 ROS2")
        assert r.installed is False

    def test_unrelated_intent_does_not_crash_and_selects_nothing(
        self, official_registry, rosclaw_home
    ):
        r = _resolver().resolve("把杯子抓起来")
        assert r.selected_skill is None
        assert r.confidence < 0.9
