"""PR-1 RED — MCP must expose capability resolution, not just list_skills (doc §26).

Today the MCP surface only has ``list_skills``, and it reads the runtime
registry — so even an installed official skill is invisible, and an agent
must *guess* skill names. These tests pin the PR-6 contract:
``resolve_capability`` / ``invoke_capability`` / ``get_skill_job`` /
``cancel_skill_job``.

Imports of the not-yet-existing tools live inside test bodies so each
test fails RED (xfail strict) until PR-6.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="RED (skill-runtime-2.0 PR-1): MCP capability tools missing; unmark in PR-6",
)


class TestMcpCapabilityTools:
    def test_resolve_capability_zh_intent(self, official_registry, rosclaw_home):
        """doc §27: the agent never learns the name `ros_install`."""
        from rosclaw.mcp.tools import resolve_capability

        result = asyncio.run(resolve_capability(intent="帮我安装 ROS2"))
        assert result["capability"] == "environment.install.ros"
        assert result["selected_skill"] == "ros-claw/ros_install"
        assert result["source"] == "official"
        assert result["compatible"] is True

    def test_capability_tool_surface_is_exposed(self, official_registry, rosclaw_home):
        from rosclaw.mcp import tools

        for name in (
            "resolve_capability",
            "invoke_capability",
            "get_skill_job",
            "cancel_skill_job",
        ):
            assert callable(getattr(tools, name, None)), f"MCP tool {name} missing"

    def test_invoke_capability_requires_approval_for_host_domain(
        self, official_registry, rosclaw_home
    ):
        """doc §22/§43: host-domain execution without approval must fail."""
        from rosclaw.mcp.tools import invoke_capability

        result = asyncio.run(
            invoke_capability(capability_id="environment.install.ros")
        )
        assert result.get("status") in {"AWAITING_APPROVAL", "AUTHENTICATION_REQUIRED"}
        assert result.get("executed") is not True
