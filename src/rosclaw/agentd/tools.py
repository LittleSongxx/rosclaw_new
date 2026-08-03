"""Built-in agentd tools (P0).

SIM-only, honest tools: they never claim physical truth. ``sim_state``
reports a deterministic simulated body state explicitly marked as
simulation; real daemon-backed tools (get_robot_state via rosclawd) are
wired in a later PR through the daemon client — the registry is the seam.
"""

from __future__ import annotations

import json
from typing import Any

from rosclaw.agentd.models.gateway import StrictTool
from rosclaw.contracts.common import ValidationError

SIM_STATE_TOOL = "sim_get_state"
SIM_BODY_TOOL = "sim_body_profile"

_TOOL_SCHEMAS: dict[str, StrictTool] = {
    SIM_STATE_TOOL: StrictTool(
        name=SIM_STATE_TOOL,
        description=(
            "Read the SIMULATED body state (joints, health). "
            "Evidence class: simulated — never usable as REAL proof."
        ),
        parameters={
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
            "required": ["verbose"],
            "additionalProperties": False,
        },
    ),
    SIM_BODY_TOOL: StrictTool(
        name=SIM_BODY_TOOL,
        description="Read the bound body's static profile summary.",
        parameters={
            "type": "object",
            "properties": {"detail": {"type": "boolean"}},
            "required": ["detail"],
            "additionalProperties": False,
        },
    ),
}


class BuiltinToolRegistry:
    """Allowlisted executor for the agentd's own P0 tools."""

    def __init__(self, *, body_id: str, body_summary: str) -> None:
        self._body_id = body_id
        self._body_summary = body_summary

    def strict_tools(self, names: list[str]) -> list[StrictTool]:
        return [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in _TOOL_SCHEMAS:
            raise ValidationError(f"tool {name!r} not allowlisted")
        if name == SIM_STATE_TOOL:
            return json.dumps(
                {
                    "evidence_class": "simulated",
                    "mode": "SIMULATION",
                    "body_id": self._body_id,
                    "joints_rad": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                    "health": "OK",
                    "fresh": True,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "evidence_class": "configured",
                "body_id": self._body_id,
                "summary": self._body_summary,
            },
            ensure_ascii=False,
        )
