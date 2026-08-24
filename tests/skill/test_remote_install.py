"""PR-1 RED — installing remote official skills must actually work (doc §11/§12).

Today ``rosclaw skill install ros-claw/ros_install`` fails with
"Builtin skill not found" because the installer only knows in-package
builtins. These tests pin the PR-3 contract: resolve → fetch → verify
digest → atomic extract → validate → lockfile, with
``~/.rosclaw/skills/<namespace>/<name>/<version>/`` layout.

Imports of the not-yet-existing modules live inside test bodies so each
test fails RED (xfail strict) until PR-3.
"""

from __future__ import annotations

import argparse
import json

import pytest

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="RED (skill-runtime-2.0 PR-1): remote installer missing; unmark in PR-3",
)

from rosclaw.skill.cli import add_skill_hub_parsers, cmd_skill_install  # noqa: E402


class TestSkillInstaller:
    def test_install_namespaced_ref_from_official_registry(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.installer import SkillInstaller

        receipt = SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        assert receipt.version == "0.2.0"

        pkg_dir = rosclaw_home / "skills" / "ros-claw" / "ros_install" / "0.2.0"
        assert (pkg_dir / "skill.yaml").exists()
        assert (pkg_dir / "entrypoint.py").exists()

    def test_install_writes_lockfile_with_pinned_digest(
        self, official_registry, rosclaw_home
    ):
        """doc §11: installed.lock.json pins version + package digest forever."""
        from rosclaw.skill.installer import SkillInstaller

        receipt = SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        lockfile = rosclaw_home / "skills" / "installed.lock.json"
        entry = json.loads(lockfile.read_text(encoding="utf-8"))["ros-claw/ros_install"]
        assert entry["version"] == "0.2.0"
        assert entry["package_digest"] == receipt.package_digest
        assert entry["package_digest"].startswith("sha256:")
        assert entry["trust"] in {"official_signed", "official"}

    def test_install_rejects_package_digest_mismatch(
        self, official_registry, rosclaw_home, monkeypatch, tmp_path
    ):
        """doc §13: a tampered package must never be extracted."""
        from rosclaw.skill.installer import SkillInstaller, SkillInstallError

        payload = json.loads(official_registry.read_text(encoding="utf-8"))
        payload["skills"][0]["checksums"]["package_sha256"] = "sha256:" + "0" * 64
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", tampered.as_uri())

        with pytest.raises(SkillInstallError):
            SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        assert not (rosclaw_home / "skills" / "ros-claw").exists()

    def test_install_is_atomic_when_fetch_fails(self, rosclaw_home, monkeypatch):
        from rosclaw.skill.installer import SkillInstaller, SkillInstallError

        monkeypatch.setenv(
            "ROSCLAW_SKILLS_REGISTRY_URL", "http://127.0.0.1:1/unreachable.json"
        )
        with pytest.raises(SkillInstallError):
            SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        # No half-extracted skill tree may survive a failed install.
        skills_dir = rosclaw_home / "skills" / "ros-claw"
        assert not skills_dir.exists() or not any(skills_dir.rglob("skill.yaml"))

    def test_unknown_namespaced_ref_fails_with_clear_error(
        self, official_registry, rosclaw_home
    ):
        from rosclaw.skill.installer import SkillInstaller, SkillInstallError

        with pytest.raises(SkillInstallError, match="not found"):
            SkillInstaller(rosclaw_home).install("ros-claw/does_not_exist")


class TestCliSkillInstall:
    def test_cli_install_namespaced_ref_succeeds(
        self, official_registry, rosclaw_home, capsys
    ):
        """doc §16: `rosclaw skill install ros-claw/ros_install` must work."""
        parser = argparse.ArgumentParser()
        add_skill_hub_parsers(parser.add_subparsers())
        args = parser.parse_args(["install", "ros-claw/ros_install"])
        assert cmd_skill_install(args) == 0
        assert "ros-claw/ros_install" in capsys.readouterr().out
