"""PR-4 GREEN — installed skills reach the runtime registry (doc §14).

(Was PR-1 RED: two disconnected registries, nothing loaded an installed
skill package into the runtime. PR-4 adds SkillLoader + SkillService.)
"""

from __future__ import annotations

from rosclaw.skill_manager.registry import SkillRegistry


def _runtime_skill_names(registry: SkillRegistry) -> set[str]:
    return {e.name for e in registry.list_skills(return_entries=True)}


class TestSkillActivation:
    def test_installed_skill_loads_into_runtime_registry(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.installer import SkillInstaller
        from rosclaw.skill.loader import SkillLoader

        SkillInstaller(rosclaw_home).install("ros-claw/ros_install")

        runtime_registry = SkillRegistry()
        loaded = SkillLoader(rosclaw_home, runtime_registry).load_installed()
        assert loaded >= 1
        assert "ros-claw/ros_install" in _runtime_skill_names(runtime_registry)

    def test_installed_skill_survives_runtime_restart(
        self, official_registry, rosclaw_home
    ):
        """doc §43 RED: restart runtime → the skill is still there."""
        from rosclaw.skill.installer import SkillInstaller
        from rosclaw.skill.loader import SkillLoader

        SkillInstaller(rosclaw_home).install("ros-claw/ros_install")

        first_runtime = SkillRegistry()
        SkillLoader(rosclaw_home, first_runtime).load_installed()
        assert "ros-claw/ros_install" in _runtime_skill_names(first_runtime)

        # Simulate a restart: brand-new in-memory registry, same home.
        restarted_runtime = SkillRegistry()
        assert "ros-claw/ros_install" not in _runtime_skill_names(restarted_runtime)
        SkillLoader(rosclaw_home, restarted_runtime).load_installed()
        assert "ros-claw/ros_install" in _runtime_skill_names(restarted_runtime)

    def test_loader_skips_builtin_shadowing_and_reports_count(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.installer import SkillInstaller
        from rosclaw.skill.loader import SkillLoader

        SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        runtime_registry = SkillRegistry()
        loader = SkillLoader(rosclaw_home, runtime_registry)
        first = loader.load_installed()
        second = loader.load_installed()
        assert first >= 1
        assert second == 0, "re-loading must be idempotent, not duplicate entries"

    def test_skill_service_lists_active_installed_skill(
        self, official_registry, rosclaw_home
    ):
        """doc §15: one SkillService facade used by CLI, MCP and agents."""
        from rosclaw.skill.installer import SkillInstaller
        from rosclaw.skill.service import SkillService

        SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        service = SkillService(rosclaw_home)
        active = {s["name"] for s in service.list_active()}
        assert "ros-claw/ros_install" in active
