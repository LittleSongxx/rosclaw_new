"""PR-6 GREEN — MCP exposes capability resolution, not just list_skills (doc §26).

(Was PR-1 RED: MCP only saw the runtime registry and agents had to guess
skill names. PR-6 adds resolve_capability / invoke_capability /
get_skill_job / cancel_skill_job, envelope-wrapped like all P0 tools.)
"""

from __future__ import annotations

import asyncio
import json

# Reuse the hermetic skill fixtures (local file:// registry + tmp home).
from tests.skill.conftest import official_registry, rosclaw_home  # noqa: F401


def _call(tool, **kwargs) -> dict:
    """Invoke an envelope-wrapped MCP tool and return the data payload."""
    envelope = json.loads(asyncio.run(tool(**kwargs)))
    assert envelope["ok"] is True, envelope
    return envelope["data"]


class TestMcpCapabilityTools:
    def test_resolve_capability_zh_intent(self, official_registry, rosclaw_home):  # noqa: F811
        """doc §27: the agent never learns the name `ros_install`."""
        from rosclaw.mcp.tools import resolve_capability

        data = _call(resolve_capability, intent="帮我安装 ROS2")
        assert data["capability"] == "environment.install.ros"
        assert data["selected_skill"] == "ros-claw/ros_install"
        assert data["source"] == "official"
        assert data["compatible"] is True

    def test_capability_tool_surface_is_exposed(self, official_registry, rosclaw_home):  # noqa: F811
        from rosclaw.mcp import tools

        for name in (
            "resolve_capability",
            "invoke_capability",
            "get_skill_job",
            "cancel_skill_job",
        ):
            assert callable(getattr(tools, name, None)), f"MCP tool {name} missing"

    def test_invoke_capability_requires_approval_for_host_domain(
        self, official_registry, rosclaw_home  # noqa: F811
    ):
        """doc §22/§43: host-domain execution without approval must fail."""
        from rosclaw.mcp.tools import invoke_capability

        data = _call(invoke_capability, capability_id="environment.install.ros")
        assert data["status"] in {"AWAITING_APPROVAL", "AUTHENTICATION_REQUIRED"}
        assert data["executed"] is not True

    def test_skill_job_roundtrip_via_mcp(self, official_registry, rosclaw_home):  # noqa: F811
        """doc §24: invoke creates a job; get/cancel operate on it (doc §26)."""
        from rosclaw.mcp.tools import cancel_skill_job, get_skill_job, invoke_capability

        created = _call(invoke_capability, capability_id="environment.install.ros")
        job_id = created["job_id"]

        fetched = _call(get_skill_job, job_id=job_id)
        assert fetched["job_id"] == job_id
        assert fetched["status"] == "AWAITING_APPROVAL"

        cancelled = _call(cancel_skill_job, job_id=job_id)
        assert cancelled["status"] == "CANCELLED"

        # Terminal jobs are stable: cancelling again is a no-op.
        again = _call(cancel_skill_job, job_id=job_id)
        assert again["status"] == "CANCELLED"
