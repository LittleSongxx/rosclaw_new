"""HostOps executor (doc §18/§19/§47).

Accepts validated, approved typed plans and converts each operation into
safe argv or a broker-owned Python implementation — the skill never
supplies argv. ``shell=True``, ``bash -c``, ``sudo sh`` and
``curl | bash`` are structurally impossible here: there is no shell to
inject into.

Two operation families:

- **argv ops** (``package.install`` …): the broker builds the argv.
- **python ops** (``artifact.fetch``, ``file.managed_write``): the broker
  performs the operation itself — digest-verified fetch into a managed
  artifacts dir, atomic writes under allowlisted roots only.

Real execution requires ``dry_run=False`` *and* an approval bound to the
plan hash; the default stays a preview so nothing runs by accident.
"""

from __future__ import annotations

import hashlib
import os
import subprocess  # noqa: S404 — argv-only, shell=False, policy-gated
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.hostops.policy import HostOpsPolicy, HostOpsPolicyError

_FETCH_TIMEOUT = 120.0

# Roots the broker may write managed files into (doc §18 file.managed_write).
DEFAULT_MANAGED_ROOTS: tuple[str, ...] = (
    "/etc/profile.d",
    "/etc/udev/rules.d",
    "/etc/apt/sources.list.d",
    "/etc/apt/keyrings",
    "/etc/ros",
)


class HostOpsExecutor:
    """Executes typed HostOps plans (preview by default)."""

    def __init__(
        self,
        *,
        dry_run: bool = True,
        policy: HostOpsPolicy | None = None,
        artifacts_dir: Path | None = None,
        managed_roots: tuple[str, ...] | None = None,
    ) -> None:
        self._dry_run = dry_run
        self._policy = policy or HostOpsPolicy()
        self._artifacts_dir = artifacts_dir
        self._managed_roots = managed_roots or DEFAULT_MANAGED_ROOTS

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def execute(self, plan: dict, approval: dict | None = None) -> dict:
        self._policy.validate_plan(plan)
        if not self._dry_run:
            # Mutating the host always requires an approval bound to the
            # exact plan hash (doc §21); previews are read-only.
            self._policy.require_approval(plan, approval)
        results = []
        for op in plan["operations"]:
            if self._dry_run:
                results.append(self._preview(op))
                continue
            result = self._execute_op(op)
            results.append(result)
            if result["status"] == "FAILED":
                break  # fail fast; recovery belongs to the skill (doc §37)
        return {
            "dry_run": self._dry_run,
            "results": results,
            "status": "PREVIEW" if self._dry_run else results[-1]["status"],
        }

    def _preview(self, op: dict) -> dict:
        try:
            if op["type"] in self._PYTHON_OPS:
                return {"type": op["type"], "argv": None, "status": "PREVIEW"}
            argv: list[str] | None = self._to_argv(op)
        except HostOpsPolicyError:
            argv = None
        return {"type": op["type"], "argv": argv, "status": "PREVIEW" if argv else "UNMAPPED"}

    def _execute_op(self, op: dict) -> dict:
        handler = self._PYTHON_OPS.get(op["type"])
        try:
            if handler is not None:
                return handler(self, op)
            argv = self._to_argv(op)  # unmapped ops fail closed here
            completed = subprocess.run(  # noqa: S603 — shell=False by construction
                argv, check=False, capture_output=True, text=True, timeout=1800
            )
            return {
                "type": op["type"],
                "argv": argv,
                "status": "OK" if completed.returncode == 0 else "FAILED",
                "returncode": completed.returncode,
            }
        except HostOpsPolicyError as exc:
            return {"type": op["type"], "status": "FAILED", "error": str(exc)}

    # ------------------------------------------------------------------
    # argv operations
    # ------------------------------------------------------------------

    def _to_argv(self, op: dict) -> list[str]:
        """Map a typed operation to safe argv owned by the broker."""
        op_type = op["type"]
        if op_type == "package.install":
            return ["apt-get", "install", "-y", *op["packages"]]
        if op_type == "package.remove":
            return ["apt-get", "remove", "-y", *op["packages"]]
        if op_type == "package.update":
            return ["apt-get", "update"]
        if op_type == "package.install_deb":
            # Only broker-fetched artifacts may be installed — never a
            # skill-supplied path.
            return ["dpkg", "-i", str(self._resolve_artifact(op["artifact"]))]
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
        # Operations without a broker mapping yet (keyring.install,
        # udev.install_rule, repository.add/remove/install_package,
        # network.fetch_verified) fail closed rather than improvising.
        raise HostOpsPolicyError(
            f"operation {op_type!r} has no broker argv mapping yet; "
            f"refusing to improvise one (fail closed, doc §47)"
        )

    # ------------------------------------------------------------------
    # python operations
    # ------------------------------------------------------------------

    @property
    def artifacts_dir(self) -> Path:
        if self._artifacts_dir is not None:
            return Path(self._artifacts_dir)
        return get_rosclaw_home() / "hostops" / "artifacts"

    def _resolve_artifact(self, name: str) -> Path:
        if "/" in name or ".." in name or not name:
            raise HostOpsPolicyError(f"artifact name {name!r} is not a safe identifier")
        path = self.artifacts_dir / name
        if not path.exists():
            raise HostOpsPolicyError(
                f"artifact {name!r} was not fetched by the broker; "
                f"package.install_deb only accepts broker-managed artifacts"
            )
        return path

    def _op_artifact_fetch(self, op: dict) -> dict:
        """Digest-verified fetch into the managed artifacts dir (doc §19)."""
        name = str(op.get("name") or op.get("source") or "")
        url = str(op.get("url", ""))
        expected = str(op.get("sha256", ""))
        if not name or "/" in name or ".." in name:
            raise HostOpsPolicyError(f"artifact.fetch name {name!r} is not safe")
        if not url:
            raise HostOpsPolicyError("artifact.fetch requires a url")
        if not expected:
            raise HostOpsPolicyError(
                "artifact.fetch requires sha256; unverified fetches are never "
                "performed (doc §19)"
            )
        try:
            with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
                blob = resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"type": op["type"], "status": "FAILED", "error": f"fetch failed: {exc}"}
        actual = "sha256:" + hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise HostOpsPolicyError(
                f"artifact {name!r} digest mismatch: expected {expected}, got {actual}"
            )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.artifacts_dir, prefix=".fetch-", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, self.artifacts_dir / name)
        return {
            "type": op["type"],
            "status": "OK",
            "artifact": name,
            "path": str(self.artifacts_dir / name),
            "digest": actual,
        }

    def _op_file_managed_write(self, op: dict) -> dict:
        """Atomic write under an allowlisted root only (doc §18)."""
        raw_path = str(op.get("path", ""))
        content = str(op.get("content", ""))
        mode = int(str(op.get("mode", "0644")), 8)
        target = Path(raw_path)
        if not target.is_absolute():
            raise HostOpsPolicyError(
                f"file.managed_write path {raw_path!r} must be absolute"
            )
        resolved = Path(os.path.realpath(target))
        roots = [Path(os.path.realpath(r)) for r in self._managed_roots]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise HostOpsPolicyError(
                f"file.managed_write path {raw_path!r} is outside the managed "
                f"roots {sorted(self._managed_roots)}"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=resolved.parent, prefix=".managed-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(tmp, mode)
        os.replace(tmp, resolved)
        return {"type": op["type"], "status": "OK", "path": str(resolved), "mode": oct(mode)}

    _PYTHON_OPS = {
        "artifact.fetch": _op_artifact_fetch,
        "file.managed_write": _op_file_managed_write,
    }
