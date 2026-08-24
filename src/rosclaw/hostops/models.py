"""HostOps plan models (doc §18/§20).

Plans are JSON-native dicts so they can cross the MCP / CLI / receipt
boundary without custom codecs. ``ALLOWED_OPERATION_TYPES`` is the
allowlist the policy enforces — anything outside it fails closed.
"""

from __future__ import annotations

# Typed host operations (doc §18) plus the artifact/deb operations used by
# the official ros2-apt-source flow (doc §20/§33).
ALLOWED_OPERATION_TYPES: frozenset[str] = frozenset(
    {
        "package.install",
        "package.remove",
        "package.install_deb",
        "repository.add",
        "repository.remove",
        "repository.enable",
        "repository.install_package",
        "keyring.install",
        "systemd.enable",
        "systemd.start",
        "systemd.restart",
        "udev.install_rule",
        "udev.reload",
        "user.group_add",
        "file.managed_write",
        "network.fetch_verified",
        "artifact.fetch",
    }
)

# Fields that would smuggle arbitrary execution into an otherwise typed op.
FORBIDDEN_OPERATION_FIELDS: frozenset[str] = frozenset(
    {"command", "shell", "script", "bash", "sh", "eval", "exec"}
)

PLAN_DOMAINS: frozenset[str] = frozenset(
    {"host", "physical", "simulation", "compute", "data", "workflow"}
)


def make_plan(
    *,
    skill: str,
    domain: str,
    target: dict,
    operations: list[dict],
) -> dict:
    """Assemble a normalized ExecutionPlan dict (doc §20)."""
    return {
        "skill": skill,
        "domain": domain,
        "target": dict(target),
        "operations": [dict(op) for op in operations],
    }
