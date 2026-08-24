"""GREEN — github_subdir sources: the official registry's native form (doc §11/§12).

The real ``ros-claw/skills`` registry points at repo subdirectories, not
tarballs. The installer pins the ref to a commit, fetches the repo
archive, extracts only the skill subdir and verifies the directory digest
with the same recipe the official registry builder uses.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from rosclaw.skill.installer import SkillInstaller, SkillInstallError, sha256_dir
from tests.skill.conftest import ROS_INSTALL_ENTRYPOINT, ROS_INSTALL_V2_MANIFEST

_FAKE_COMMIT = "a" * 40


def _build_repo_archive(root: Path) -> tuple[Path, str]:
    """Build a fake GitHub repo archive; return (archive, subdir digest)."""
    files = {
        "skills/ros_install/skill.yaml": ROS_INSTALL_V2_MANIFEST.encode(),
        "skills/ros_install/entrypoint.py": ROS_INSTALL_ENTRYPOINT.encode(),
        "skills/other_skill/skill.yaml": b"metadata: {name: other}\n",
        "README.md": b"# repo\n",
    }
    pkg_root = root / "pkg-src" / "skills" / "ros_install"
    pkg_root.mkdir(parents=True)
    (pkg_root / "skill.yaml").write_text(ROS_INSTALL_V2_MANIFEST, encoding="utf-8")
    (pkg_root / "entrypoint.py").write_text(ROS_INSTALL_ENTRYPOINT, encoding="utf-8")
    digest = sha256_dir(pkg_root)

    archive = root / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for relname, content in files.items():
            info = tarfile.TarInfo(f"skills-{_FAKE_COMMIT[:7]}/{relname}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return archive, digest


def _subdir_registry(root: Path, archive: Path, digest: str) -> Path:
    payload = {
        "schema_version": "rosclaw.skills_registry.v1",
        "skills": [
            {
                "name": "ros-claw/ros_install",
                "version": "0.2.0",
                "description": "Install, verify or repair ROS / ROS 2.",
                "tags": ["ros", "ros2", "install"],
                "official": True,
                "installable": True,
                "verification_status": "host_matrix_verified",
                "capability": {"id": "environment.install.ros", "intents": {"zh": ["安装 ROS2"]}},
                "source": {
                    "type": "github_subdir",
                    "repo": "https://github.com/ros-claw/skills",
                    "ref": _FAKE_COMMIT,
                    "subdir": "skills/ros_install",
                    # Hermetic override: production derives the codeload URL
                    # from repo+commit; tests serve a local archive instead.
                    "archive_url": archive.as_uri(),
                },
                "checksums": {"package_sha256": digest},
            }
        ],
    }
    registry = root / "skills.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    return registry


class TestGithubSubdirInstall:
    def test_install_from_github_subdir_source(self, rosclaw_home, monkeypatch, tmp_path):
        archive, digest = _build_repo_archive(tmp_path)
        registry = _subdir_registry(tmp_path, archive, digest)
        monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", registry.as_uri())

        receipt = SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        assert receipt.version == "0.2.0"
        assert receipt.source_commit == _FAKE_COMMIT
        assert receipt.package_digest == digest

        pkg_dir = rosclaw_home / "skills" / "ros-claw" / "ros_install" / "0.2.0"
        assert (pkg_dir / "skill.yaml").exists()
        assert (pkg_dir / "entrypoint.py").exists()
        # Only the requested subdir was installed — not the whole repo.
        assert not (pkg_dir / "README.md").exists()

        lock = json.loads(
            (rosclaw_home / "skills" / "installed.lock.json").read_text(encoding="utf-8")
        )
        assert lock["ros-claw/ros_install"]["source_commit"] == _FAKE_COMMIT

    def test_subdir_digest_mismatch_is_refused_without_litter(
        self, rosclaw_home, monkeypatch, tmp_path
    ):
        archive, _digest = _build_repo_archive(tmp_path)
        registry = _subdir_registry(tmp_path, archive, "sha256:" + "0" * 64)
        monkeypatch.setenv("ROSCLAW_SKILLS_REGISTRY_URL", registry.as_uri())

        with pytest.raises(SkillInstallError, match="digest mismatch"):
            SkillInstaller(rosclaw_home).install("ros-claw/ros_install")
        skills_dir = rosclaw_home / "skills"
        leftovers = [p for p in skills_dir.rglob("*") if p.name != "skills"] if skills_dir.exists() else []
        assert leftovers == [], f"failed install left litter: {leftovers}"

    def test_sha256_dir_matches_official_recipe_vector(self, tmp_path):
        """Guard the digest recipe against drift from the official builder."""
        (tmp_path / "dir").mkdir()
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "dir" / "b.txt").write_bytes(b"world\n")
        assert sha256_dir(tmp_path) == (
            "sha256:e1b742f81543c933ed70ce7aeb1c2d58f94d6f11c2c4121b3eb470d88a46dd27"
        )
