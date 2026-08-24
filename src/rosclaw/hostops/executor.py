"""HostOps executor (doc §18/§19/§47).

Accepts validated, approved typed plans and converts each operation into
safe argv *itself* — the skill never supplies argv. ``shell=True``,
``bash -c``, ``sudo sh`` and ``curl | bash`` are structurally impossible
here: there is no shell to inject into.

Real execution requires ``dry_run=False`` *and* an approval bound to the
plan hash; the default stays a preview so nothing runs by accident.
"""

from __future__ import annotations

import subprocess  # noqa: S404 — argv-only, shell=False, policy-gated

from rosclaw.hostops.policy import HostOpsPolicy, HostOpsPolicyError


class HostOpsExecutor:
    """Executes typed HostOps plans (preview by default)."""

    def __init__(self, *, dry_run: bool = True, policy: HostOpsPolicy | None = None) -> None:
        self._dry_run = dry_run
        self._policy = policy or HostOpsPolicy()

    def execute(self, plan: dict, approval: dict | None = None) -> dict:
        self._policy.validate_plan(plan)
        if not self._dry_run:
            # Mutating the host always requires an approval bound to the
            # exact plan hash (doc §21); previews are read-only.
            self._policy.require_approval(plan, approval)
        results = []
        for op in plan["operations"]:
            if self._dry_run:
                try:
                    argv: list[str] | None = self._to_argv(op)
                except HostOpsPolicyError:
                    argv = None
                results.append(
                    {
                        "type": op["type"],
                        "argv": argv,
                        "status": "PREVIEW" if argv else "UNMAPPED",
                    }
                )
                continue
            argv = self._to_argv(op)  # unmapped ops fail closed here
            completed = subprocess.run(  # noqa: S603 — shell=False by construction
                argv, check=False, capture_output=True, text=True, timeout=1800
            )
            results.append(
                {
                    "type": op["type"],
                    "argv": argv,
                    "status": "OK" if completed.returncode == 0 else "FAILED",
                    "returncode": completed.returncode,
                }
            )
            if completed.returncode != 0:
                break  # fail fast; recovery belongs to the skill (doc §37)
        return {
            "dry_run": self._dry_run,
            "results": results,
            "status": "PREVIEW" if self._dry_run else results[-1]["status"],
        }

    @staticmethod
    def _to_argv(op: dict) -> list[str]:
        """Map a typed operation to safe argv owned by the broker."""
        op_type = op["type"]
        if op_type == "package.install":
            return ["apt-get", "install", "-y", *op["packages"]]
        if op_type == "package.remove":
            return ["apt-get", "remove", "-y", *op["packages"]]
        if op_type == "package.install_deb":
            return ["dpkg", "-i", str(op["artifact"])]
        if op_type == "repository.enable":
            return ["add-apt-repository", "-y", str(op["repository"])]
        if op_type == "systemd.enable":
            return ["systemctl", "enable", str(op["unit"])]
        if op_type == "systemd.start":
            return ["systemctl", "start", str(op["unit"])]
        if op_type == "systemd.restart":
            return ["systemctl", "restart", str(op["unit"])]
        if op_type == "udev.reload":
            return ["udevadm", "control", "--reload-rules"]
        if op_type == "user.group_add":
            return ["usermod", "-aG", str(op["group"]), str(op["user"])]
        # Operations without a local argv mapping yet (artifact.fetch,
        # keyring.install, udev.install_rule, file.managed_write,
        # network.fetch_verified, repository.add/remove/install_package)
        # fail closed rather than improvising a shell.
        raise HostOpsPolicyError(
            f"operation {op_type!r} has no broker argv mapping yet; "
            f"refusing to improvise one (fail closed, doc §47)"
        )
