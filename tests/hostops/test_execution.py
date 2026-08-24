"""GREEN — HostOps execution flow (doc §18/§19/§21/§23/§24/§25).

Covers the PR-7 core execution plane: broker python ops (artifact.fetch,
file.managed_write), new argv mappings (package.update, install_deb via
broker artifacts only), policy hardening (no path traversal), and the
authorize → execute → verify → receipt job flow — all hermetic (file://
artifacts, tmp managed roots, injectable sudo runner, stub skill
verifier). No real apt, no real sudo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rosclaw.hostops.executor import HostOpsExecutor
from rosclaw.hostops.planner import plan_hash
from rosclaw.hostops.policy import HostOpsPolicy, HostOpsPolicyError
from rosclaw.hostops.runner import (
    AuthorizationError,
    authorize_job,
    execute_authorized_job,
)
from rosclaw.skill.jobs import SkillJobStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FLOW_MANIFEST = """\
schema_version: rosclaw.skill.v2
metadata:
  name: flow_skill
  namespace: test
  version: 0.1.0
execution:
  domain: host
  planner:
    type: python
    entrypoint: entrypoint.py:plan
  verifier:
    type: python
    entrypoint: entrypoint.py:verify
  recover:
    type: python
    entrypoint: entrypoint.py:recover
"""

FLOW_ENTRYPOINT = '''\
def plan(context, args):
    return {"domain": "host", "operations": [{"type": "package.update"}]}


def verify(context, receipt):
    return {"flow_check": "PASS", "result": "VERIFIED"}


def recover(context, failure):
    return {"action": "retry_after_dpkg_fix"}
'''


@pytest.fixture
def deb_file(tmp_path: Path) -> tuple[Path, str]:
    blob = b"fake deb payload"
    path = tmp_path / "fake.deb"
    path.write_bytes(blob)
    return path, "sha256:" + hashlib.sha256(blob).hexdigest()


@pytest.fixture
def managed_etc(tmp_path: Path) -> Path:
    etc = tmp_path / "etc" / "profile.d"
    etc.mkdir(parents=True)
    return tmp_path / "etc"


def _plan(operations: list[dict]) -> dict:
    return {
        "skill": "test/flow_skill@0.1.0",
        "domain": "host",
        "target": {"os": "ubuntu", "version": "24.04", "arch": "arm64"},
        "operations": operations,
    }


# ---------------------------------------------------------------------------
# Broker python ops
# ---------------------------------------------------------------------------


class TestArtifactFetch:
    def test_fetch_verifies_digest_and_stores(self, rosclaw_home, deb_file, tmp_path):
        path, digest = deb_file
        executor = HostOpsExecutor(dry_run=False)
        plan = _plan(
            [{"type": "artifact.fetch", "name": "fake-deb", "url": path.as_uri(), "sha256": digest}]
        )
        approval = HostOpsPolicy().approve(plan_hash(plan))
        result = executor.execute(plan, approval)
        assert result["status"] == "OK"
        stored = rosclaw_home / "hostops" / "artifacts" / "fake-deb"
        assert stored.read_bytes() == path.read_bytes()

    def test_fetch_digest_mismatch_fails(self, rosclaw_home, deb_file):
        path, _digest = deb_file
        executor = HostOpsExecutor(dry_run=False)
        plan = _plan(
            [
                {
                    "type": "artifact.fetch",
                    "name": "fake-deb",
                    "url": path.as_uri(),
                    "sha256": "sha256:" + "0" * 64,
                }
            ]
        )
        approval = HostOpsPolicy().approve(plan_hash(plan))
        result = executor.execute(plan, approval)
        assert result["status"] == "FAILED"
        assert "digest mismatch" in result["results"][0]["error"]
        assert not (rosclaw_home / "hostops" / "artifacts" / "fake-deb").exists()

    def test_fetch_requires_sha256(self, rosclaw_home, deb_file):
        path, _ = deb_file
        executor = HostOpsExecutor(dry_run=False)
        plan = _plan(
            [{"type": "artifact.fetch", "name": "fake-deb", "url": path.as_uri(), "sha256": "x"}]
        )
        # Policy allows the shape; the broker refuses unverified fetches.
        plan["operations"][0].pop("sha256")
        approval = HostOpsPolicy().approve(plan_hash(plan))
        result = executor.execute(plan, approval)
        assert result["status"] == "FAILED"
        assert "requires sha256" in result["results"][0]["error"]


class TestManagedWrite:
    def test_write_under_allowlisted_root(self, rosclaw_home, managed_etc):
        executor = HostOpsExecutor(dry_run=False, managed_roots=(str(managed_etc),))
        target = managed_etc / "profile.d" / "rosclaw-ros.sh"
        plan = _plan(
            [
                {
                    "type": "file.managed_write",
                    "path": str(target),
                    "content": "source /opt/ros/jazzy/setup.sh\n",
                }
            ]
        )
        approval = HostOpsPolicy().approve(plan_hash(plan))
        assert executor.execute(plan, approval)["status"] == "OK"
        assert target.read_text() == "source /opt/ros/jazzy/setup.sh\n"

    def test_write_outside_roots_is_refused(self, rosclaw_home, managed_etc, tmp_path):
        executor = HostOpsExecutor(dry_run=False, managed_roots=(str(managed_etc),))
        plan = _plan(
            [
                {
                    "type": "file.managed_write",
                    "path": str(tmp_path / "elsewhere" / "evil.sh"),
                    "content": "x",
                }
            ]
        )
        approval = HostOpsPolicy().approve(plan_hash(plan))
        result = executor.execute(plan, approval)
        assert result["status"] == "FAILED"
        assert "outside the managed roots" in result["results"][0]["error"]


class TestArgvMappings:
    def test_package_update_preview(self):
        preview = HostOpsExecutor(dry_run=True).execute(_plan([{"type": "package.update"}]))
        assert preview["results"][0]["argv"] == ["apt-get", "update"]

    def test_install_deb_only_accepts_broker_artifacts(self, rosclaw_home):
        executor = HostOpsExecutor(dry_run=False)
        plan = _plan([{"type": "package.install_deb", "artifact": "never-fetched"}])
        approval = HostOpsPolicy().approve(plan_hash(plan))
        result = executor.execute(plan, approval)
        assert result["status"] == "FAILED"
        assert "not fetched by the broker" in result["results"][0]["error"]

    def test_policy_rejects_path_traversal_values(self):
        with pytest.raises(HostOpsPolicyError):
            HostOpsPolicy().validate_plan(
                _plan(
                    [
                        {
                            "type": "artifact.fetch",
                            "name": "..evil",
                            "url": "file:///tmp/x",
                            "sha256": "sha256:" + "0" * 64,
                        }
                    ]
                )
            )


# ---------------------------------------------------------------------------
# authorize → execute → verify → receipt flow
# ---------------------------------------------------------------------------


def _install_flow_skill(home: Path) -> None:
    pkg = home / "skills" / "test" / "flow_skill" / "0.1.0"
    pkg.mkdir(parents=True)
    (pkg / "skill.yaml").write_text(FLOW_MANIFEST, encoding="utf-8")
    (pkg / "entrypoint.py").write_text(FLOW_ENTRYPOINT, encoding="utf-8")
    lockfile = home / "skills" / "installed.lock.json"
    lockfile.write_text(
        json.dumps({"test/flow_skill": {"version": "0.1.0", "trust": "official"}}),
        encoding="utf-8",
    )


class TestAuthorizeFlow:
    def test_authorize_runs_sudo_then_root_phase(self, rosclaw_home):
        store = SkillJobStore(rosclaw_home)
        job = store.create(
            skill="test/flow_skill",
            capability=None,
            status="AWAITING_APPROVAL",
            plan_hash="abc",
            plan=_plan([{"type": "package.update"}]),
        )
        calls: list[list[str]] = []

        def fake_sudo(argv: list[str], home: Path) -> int:
            calls.append(argv)
            return 0

        updated = authorize_job(job["job_id"], home=rosclaw_home, sudo_runner=fake_sudo)
        assert calls[0] == ["sudo", "-v"]
        assert calls[1][:3] == ["sudo", "-n", "env"]
        assert f"ROSCLAW_HOME={rosclaw_home}" in calls[1]
        assert updated["status"] == "AUTHORIZED"

    def test_authorize_refuses_wrong_state(self, rosclaw_home):
        store = SkillJobStore(rosclaw_home)
        job = store.create(
            skill="test/flow_skill", capability=None, status="SUCCEEDED"
        )
        with pytest.raises(AuthorizationError, match="SUCCEEDED"):
            authorize_job(job["job_id"], home=rosclaw_home, sudo_runner=lambda a, h: 0)

    def test_sudo_failure_marks_job_failed(self, rosclaw_home):
        store = SkillJobStore(rosclaw_home)
        job = store.create(
            skill="test/flow_skill",
            capability=None,
            status="AUTHENTICATION_REQUIRED",
            plan_hash="abc",
            plan=_plan([{"type": "package.update"}]),
        )
        with pytest.raises(AuthorizationError, match="sudo"):
            authorize_job(job["job_id"], home=rosclaw_home, sudo_runner=lambda a, h: 1)
        assert store.get(job["job_id"])["status"] == "FAILED"


class TestExecuteAuthorizedJob:
    def _authorized_job(self, home: Path, plan: dict) -> dict:
        store = SkillJobStore(home)
        return store.create(
            skill="test/flow_skill",
            capability="test.flow",
            status="AUTHORIZED",
            plan_hash=plan_hash(plan),
            plan=plan,
        )

    def test_full_flow_succeeds_with_receipt(self, rosclaw_home, managed_etc, deb_file):
        _install_flow_skill(rosclaw_home)
        path, digest = deb_file
        target = managed_etc / "profile.d" / "ros.sh"
        plan = _plan(
            [
                {
                    "type": "artifact.fetch",
                    "name": "fake-deb",
                    "url": path.as_uri(),
                    "sha256": digest,
                },
                {
                    "type": "file.managed_write",
                    "path": str(target),
                    "content": "source /opt/ros/jazzy/setup.sh\n",
                },
            ]
        )
        job = self._authorized_job(rosclaw_home, plan)
        executor = HostOpsExecutor(dry_run=False, managed_roots=(str(managed_etc),))
        receipt = execute_authorized_job(job["job_id"], home=rosclaw_home, executor=executor)

        assert receipt["result"] == "VERIFIED"
        assert receipt["verification"]["result"] == "VERIFIED"
        assert receipt["plan_hash"] == plan_hash(plan)
        assert (rosclaw_home / "hostops" / "artifacts" / "fake-deb").exists()
        assert target.exists()

        updated = SkillJobStore(rosclaw_home).get(job["job_id"])
        assert updated["status"] == "SUCCEEDED"
        receipt_file = Path(updated["receipt_path"])
        assert json.loads(receipt_file.read_text())["result"] == "VERIFIED"

    def test_execute_refuses_unauthorized_job(self, rosclaw_home):
        store = SkillJobStore(rosclaw_home)
        job = store.create(
            skill="test/flow_skill",
            capability=None,
            status="AWAITING_APPROVAL",
            plan_hash="abc",
            plan=_plan([{"type": "package.update"}]),
        )
        with pytest.raises(AuthorizationError, match="refusing to execute"):
            execute_authorized_job(job["job_id"], home=rosclaw_home)

    def test_failed_op_yields_failed_job_with_recovery(self, rosclaw_home, deb_file):
        _install_flow_skill(rosclaw_home)
        path, _digest = deb_file
        plan = _plan(
            [
                {
                    "type": "artifact.fetch",
                    "name": "fake-deb",
                    "url": path.as_uri(),
                    "sha256": "sha256:" + "1" * 64,
                }
            ]
        )
        job = self._authorized_job(rosclaw_home, plan)
        receipt = execute_authorized_job(job["job_id"], home=rosclaw_home)
        assert receipt["result"] == "FAILED"
        assert receipt["recovery"] == {"action": "retry_after_dpkg_fix"}
        assert SkillJobStore(rosclaw_home).get(job["job_id"])["status"] == "FAILED"
