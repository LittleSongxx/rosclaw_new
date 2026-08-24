"""HostOps policy gate (doc §19/§21/§47).

Default fail closed: unknown operation types are rejected, and no plan may
contain fields that smuggle arbitrary execution (``command``, ``shell``,
``bash``, …) — not even on an otherwise-allowed typed op. Argument values
are validated against injection (no ``--flag`` smuggled as a package name).
"""

from __future__ import annotations

import re

from rosclaw.hostops.models import (
    ALLOWED_OPERATION_TYPES,
    FORBIDDEN_OPERATION_FIELDS,
)
from rosclaw.hostops.planner import plan_hash

# Apt package / repository names: no leading dash, no whitespace, no shell
# metacharacters — the executor must never be an argv-injection vector.
_SAFE_TOKEN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9+_.:~/-]*$")


class HostOpsPolicyError(Exception):
    """The plan violates HostOps policy and must not execute."""


class ApprovalMismatchError(HostOpsPolicyError):
    """The approval does not match this plan's hash (re-approval needed)."""


class HostOpsPolicy:
    """Validates plans and binds approvals to plan hashes."""

    # ------------------------------------------------------------------
    # Plan validation (fail closed)
    # ------------------------------------------------------------------

    def validate_plan(self, plan: dict) -> None:
        if not isinstance(plan, dict):
            raise HostOpsPolicyError("plan must be a JSON object")
        operations = plan.get("operations")
        if not isinstance(operations, list) or not operations:
            raise HostOpsPolicyError("plan.operations must be a non-empty list")
        for index, op in enumerate(operations):
            self._validate_operation(index, op)

    def _validate_operation(self, index: int, op: object) -> None:
        where = f"operations[{index}]"
        if not isinstance(op, dict):
            raise HostOpsPolicyError(f"{where} must be an object")
        op_type = op.get("type")
        if not isinstance(op_type, str) or not op_type:
            raise HostOpsPolicyError(f"{where} lacks a string 'type'")
        if op_type not in ALLOWED_OPERATION_TYPES:
            raise HostOpsPolicyError(
                f"{where} type {op_type!r} is not in the HostOps allowlist; "
                f"arbitrary shell execution is never allowed (doc §19)"
            )
        smuggled = FORBIDDEN_OPERATION_FIELDS & {k for k, v in op.items() if v}
        if smuggled:
            raise HostOpsPolicyError(
                f"{where} contains forbidden shell field(s) "
                f"{sorted(smuggled)}; typed operations only (doc §19)"
            )
        for key, value in op.items():
            if key == "type":
                continue
            if op_type == "file.managed_write" and key == "content":
                # File content is data written verbatim to the managed file;
                # it is never executed, so it is not token-validated.
                if not isinstance(value, str):
                    raise HostOpsPolicyError(
                        f"{where}.content must be a string"
                    )
                continue
            if op_type == "file.managed_write" and key == "path":
                # Absolute managed path: no traversal, no whitespace (the
                # executor additionally confines it to managed roots).
                if (
                    not isinstance(value, str)
                    or not value.startswith("/")
                    or ".." in value
                    or any(c.isspace() for c in value)
                ):
                    raise HostOpsPolicyError(
                        f"{where}.path must be an absolute path without "
                        f"traversal or whitespace, got {value!r}"
                    )
                continue
            self._validate_value(f"{where}.{key}", value)

    def _validate_value(self, where: str, value: object) -> None:
        if isinstance(value, str):
            if not _SAFE_TOKEN.match(value) or ".." in value:
                raise HostOpsPolicyError(
                    f"{where} value {value!r} is not a safe token "
                    f"(no flags, whitespace, path traversal or shell metacharacters)"
                )
        elif isinstance(value, list):
            if not value:
                raise HostOpsPolicyError(f"{where} must not be an empty list")
            for item in value:
                self._validate_value(where, item)
        elif isinstance(value, (int, float, bool)) or value is None:
            return
        else:
            raise HostOpsPolicyError(
                f"{where} has unsupported value type {type(value).__name__}"
            )

    # ------------------------------------------------------------------
    # Approval binding (doc §21)
    # ------------------------------------------------------------------

    def approve(self, plan_hash_value: str) -> dict:
        """Record an operator approval for one exact plan hash."""
        if not plan_hash_value:
            raise HostOpsPolicyError("cannot approve an empty plan hash")
        return {"plan_hash": plan_hash_value, "approved": True}

    def require_approval(self, plan: dict, approval: dict | None) -> None:
        """Raise unless ``approval`` matches *this* plan (doc §21)."""
        if not approval or not approval.get("approved"):
            raise ApprovalMismatchError(
                "plan has no approval; approve plan_hash "
                f"{plan_hash(plan)} first"
            )
        actual = plan_hash(plan)
        if approval.get("plan_hash") != actual:
            raise ApprovalMismatchError(
                "plan changed after approval "
                f"(approved {approval.get('plan_hash')}, current {actual}); "
                f"the changed plan needs a new approval (doc §21)"
            )
