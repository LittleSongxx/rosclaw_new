"""Shared hermetic fixtures for the Skill Runtime 2.0 red-test suite (PR-1).

No network and no real ``~/.rosclaw`` writes: the "official registry" is a
local ``file://`` URL and the skill package tarball is built inside the
test's tmp dir. The fixtures pin the *contract* the implementation PRs
(PR-2 catalog/resolver, PR-3 installer, PR-5 hostops) must satisfy — see
rosclaw_skill优化.md §43.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

# Minimal v2 manifest for the fixture ros_install package (doc §7).
ROS_INSTALL_V2_MANIFEST = """\
schema_version: rosclaw.skill.v2
kind: Skill

metadata:
  name: ros_install
  namespace: ros-claw
  version: 0.2.0
  description: Install, verify or repair a ROS / ROS 2 environment.
  tags: [ros, ros2, install, environment]

capability:
  id: environment.install.ros
  intents:
    en: [install ROS, install ROS 2, setup ROS2]
    zh: [安装 ROS, 安装 ROS2, 配置 ROS2, 修复 ROS 环境]

execution:
  domain: host
  planner:
    type: python
    entrypoint: entrypoint.py:plan
  verifier:
    type: python
    entrypoint: entrypoint.py:verify

permissions:
  - host.package.install
  - host.repository.configure

compatibility:
  os: [ubuntu]
  architectures: [amd64, arm64]

safety:
  approval_scope: transaction
  privilege: admin
  arbitrary_root_shell: false
"""

# The fixture planner emits typed HostOps only — never raw shell (doc §19).
ROS_INSTALL_ENTRYPOINT = '''\
"""Fixture ros_install entrypoint: typed plan, verify and recover hooks."""


def plan(context, args):
    distro = {"24.04": "jazzy", "22.04": "humble"}[context["os_version"]]
    return {
        "skill": "ros-claw/ros_install@0.2.0",
        "domain": "host",
        "target": {
            "os": "ubuntu",
            "version": context["os_version"],
            "arch": context["arch"],
        },
        "operations": [
            {"type": "package.install", "packages": ["software-properties-common"]},
            {"type": "repository.enable", "repository": "universe"},
            {"type": "artifact.fetch", "source": "ros2-apt-source"},
            {"type": "package.install_deb", "artifact": "ros2-apt-source"},
            {"type": "package.install", "packages": [f"ros-{distro}-desktop"]},
        ],
    }


def verify(context, receipt):
    return {"ros2_cli": "PASS", "rosdep": "PASS", "pub_sub": "PASS", "result": "VERIFIED"}


def recover(context, failure):
    return {"action": "dpkg_recovery"}
'''


def build_package_tarball(root: Path, *, name: str = "ros_install",
                          manifest: str = ROS_INSTALL_V2_MANIFEST,
                          entrypoint: str = ROS_INSTALL_ENTRYPOINT) -> tuple[Path, str]:
    """Build a skill package tarball; return (tarball_path, sha256 digest)."""
    pkg_dir = root / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "skill.yaml").write_text(manifest, encoding="utf-8")
    (pkg_dir / "entrypoint.py").write_text(entrypoint, encoding="utf-8")
    tarball = root / f"{name}-0.2.0.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(pkg_dir, arcname=name)
    digest = "sha256:" + hashlib.sha256(tarball.read_bytes()).hexdigest()
    return tarball, digest


def registry_payload(tarball: Path, digest: str) -> dict:
    """Official-registry-shaped payload pinning the fixture package."""
    return {
        "schema_version": "rosclaw.skills_registry.v1",
        "source_repo": "https://github.com/ros-claw/skills",
        "skills": [
            {
                "name": "ros-claw/ros_install",
                "display_name": "ROS Install",
                "version": "0.2.0",
                "description": "Install, verify or repair a ROS / ROS 2 environment.",
                "category": "environment",
                "tags": ["ros", "ros2", "install", "environment"],
                "official": True,
                "installable": True,
                "verification_status": "host_matrix_verified",
                "capability": {
                    "id": "environment.install.ros",
                    "intents": {
                        "en": ["install ROS", "install ROS 2", "setup ROS2"],
                        "zh": ["安装 ROS", "安装 ROS2", "配置 ROS2", "修复 ROS 环境"],
                    },
                },
                "compatibility": {"os": ["ubuntu"], "architectures": ["amd64", "arm64"]},
                "source": {"type": "tarball", "url": tarball.as_uri()},
                "checksums": {"package_sha256": digest},
            }
        ],
    }


@pytest.fixture
def official_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic official catalog pointed at via ROSCLAW_SKILLS_REGISTRY_URL."""
    tarball, digest = build_package_tarball(tmp_path)
    registry_path = tmp_path / "skills.json"
    registry_path.write_text(
        json.dumps(registry_payload(tarball, digest)), encoding="utf-8"
    )
    monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", registry_path.as_uri())
    return registry_path


@pytest.fixture
def rosclaw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all ROSClaw persistent state into a tmp home."""
    home = tmp_path / "rosclaw-home"
    home.mkdir()
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    return home
