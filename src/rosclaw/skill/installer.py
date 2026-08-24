"""Remote skill installer (Skill Runtime 2.0, doc §11/§12/§13).

``rosclaw skill install ros-claw/ros_install`` finally installs for real:

    resolve → fetch registry entry → pin version → fetch immutable source
    → verify digest → extract atomically → validate package → lockfile

Layout (doc §11)::

    $ROSCLAW_HOME/skills/<namespace>/<name>/<version>/skill.yaml ...
    $ROSCLAW_HOME/skills/installed.lock.json

Digest pinning (doc §12) is the hard guarantee today: the registry entry
carries ``checksums.package_sha256`` and a mismatch refuses extraction
before anything touches the final tree. Ed25519 registry signature
verification (doc §13) activates once the official registry starts
publishing ``skills.json.sig`` — until then trust is recorded honestly
as ``official`` rather than ``official_signed``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.skill.catalog_service import SkillCatalogService
from rosclaw.skill.sources import CatalogHit

logger = logging.getLogger("rosclaw.skill.installer")

_FETCH_TIMEOUT = 60.0


class SkillInstallError(Exception):
    """Install failed; nothing was left half-installed."""


def sha256_dir(path: Path) -> str:
    """Directory digest matching the official registry builder's recipe:
    sha256 over sorted ``relpath_bytes + file_bytes`` (ros-claw/skills
    ``scripts/build_registry.py``)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


@dataclass
class InstallReceipt:
    name: str
    version: str
    package_digest: str
    install_dir: Path
    trust: str  # official | official_signed | third_party
    source_commit: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "package_digest": self.package_digest,
            "install_dir": str(self.install_dir),
            "trust": self.trust,
            "source_commit": self.source_commit,
        }


class SkillInstaller:
    """Installs catalog skills into ``$ROSCLAW_HOME/skills``."""

    def __init__(self, home: Path | None = None, catalog: SkillCatalogService | None = None) -> None:
        self._home = Path(home) if home is not None else get_rosclaw_home()
        self._catalog = catalog or SkillCatalogService.default(self._home)

    @property
    def skills_dir(self) -> Path:
        return self._home / "skills"

    @property
    def lockfile(self) -> Path:
        return self.skills_dir / "installed.lock.json"

    def install(self, ref: str) -> InstallReceipt:
        if "/" not in ref:
            raise SkillInstallError(
                f"skill ref {ref!r} is not namespaced; "
                f"remote install expects <namespace>/<name> (e.g. ros-claw/ros_install)"
            )
        hit = self._catalog.get(ref)
        if hit is None:
            raise SkillInstallError(f"skill {ref!r} not found in any catalog")
        if not hit.installable and hit.source == "official":
            raise SkillInstallError(f"skill {ref!r} is marked not installable")

        source = (hit.raw or {}).get("source", {}) or {}
        source_type = str(source.get("type", ""))
        expected_digest = str((hit.raw or {}).get("checksums", {}).get("package_sha256", ""))
        if not expected_digest:
            raise SkillInstallError(
                f"skill {ref!r} registry entry has no package_sha256; refusing to "
                f"install an unpinned executable package (doc §12)"
            )

        version = hit.version or "0.0.0"
        final_dir = self.skills_dir / ref / version
        resolved_commit = ""
        if source_type == "github_subdir":
            staging, resolved_commit = self._stage_github_subdir(source, expected_digest, ref)
            self._move_atomically(staging, final_dir)
            actual_digest = expected_digest
            source_ref = f"{source.get('repo', '')}@{resolved_commit}"
        else:
            url = str(source.get("url", ""))
            if not url:
                raise SkillInstallError(f"skill {ref!r} has no fetchable source URL")
            blob = self._fetch(url)
            actual_digest = "sha256:" + hashlib.sha256(blob).hexdigest()
            if actual_digest != expected_digest:
                raise SkillInstallError(
                    f"package digest mismatch for {ref!r}: registry pins "
                    f"{expected_digest}, fetched {actual_digest}; refusing to extract"
                )
            self._extract_atomically(blob, final_dir, ref)
            source_ref = url
        self._validate_package(final_dir, ref)

        trust = "official" if hit.official else "third_party"
        if (hit.raw or {}).get("signature_verified"):
            trust = "official_signed"
        receipt = InstallReceipt(
            name=ref,
            version=version,
            package_digest=actual_digest,
            install_dir=final_dir,
            trust=trust,
            source_commit=resolved_commit,
        )
        self._write_lockfile(hit, receipt, source_ref)
        logger.info("installed %s@%s (%s)", ref, version, actual_digest[:19])
        return receipt

    # ------------------------------------------------------------------
    # github_subdir sources (the official registry's native form)
    # ------------------------------------------------------------------

    def _stage_github_subdir(
        self, source: dict, expected_digest: str, ref: str
    ) -> tuple[Path, str]:
        """Fetch a repo archive, extract only the subdir, verify the digest.

        The digest recipe mirrors the official registry builder
        (``scripts/build_registry.py`` in ros-claw/skills): sha256 over
        sorted ``relpath_bytes + file_bytes``. ``ref`` is pinned to a commit
        whenever possible (doc §12: never trust a moving branch name).
        """
        repo = str(source.get("repo", ""))
        git_ref = str(source.get("ref", "main"))
        subdir = str(source.get("subdir", "")).strip("/")
        if not repo or not subdir:
            raise SkillInstallError(
                f"skill {ref!r} github_subdir source lacks repo/subdir"
            )
        commit = self._resolve_commit(repo, git_ref)
        archive_url = source.get("archive_url") or self._archive_url(repo, commit)
        blob = self._fetch(str(archive_url))

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=self.skills_dir, prefix=".install-"))
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_tar:
                tmp_tar.write(blob)
                tmp_tar_path = tmp_tar.name
            try:
                with tarfile.open(tmp_tar_path, "r:gz") as tf:
                    self._safe_extract_members(tf, tf.getmembers(), work)
            finally:
                os.unlink(tmp_tar_path)
            roots = [p for p in work.iterdir() if p.is_dir()]
            src_dir = roots[0] / subdir if len(roots) == 1 else None
            if src_dir is None or not src_dir.is_dir():
                raise SkillInstallError(
                    f"subdir {subdir!r} not found in archive of {repo}"
                )
            actual = sha256_dir(src_dir)
            if actual != expected_digest:
                raise SkillInstallError(
                    f"package digest mismatch for {ref!r}: registry pins "
                    f"{expected_digest}, fetched {actual}; refusing to install"
                )
            staging = work / "pkg"
            shutil.move(str(src_dir), str(staging))
            for item in work.iterdir():
                if item != staging:
                    shutil.rmtree(item, ignore_errors=True)
            return staging, commit
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise

    @staticmethod
    def _resolve_commit(repo: str, git_ref: str) -> str:
        """Pin a branch/tag to a commit SHA; 40-hex refs pass through."""
        if len(git_ref) == 40 and all(c in "0123456789abcdef" for c in git_ref.lower()):
            return git_ref
        match = repo.removesuffix(".git").rstrip("/").removeprefix("https://github.com/")
        api = f"https://api.github.com/repos/{match}/commits/{git_ref}"
        try:
            with urllib.request.urlopen(api, timeout=_FETCH_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["sha"])
        except Exception as exc:  # noqa: BLE001 — pin failure must not be silent
            raise SkillInstallError(
                f"could not pin {repo} ref {git_ref!r} to a commit: {exc} "
                f"(doc §12: a moving ref is not a trust basis)"
            ) from exc

    @staticmethod
    def _archive_url(repo: str, commit: str) -> str:
        match = repo.removesuffix(".git").rstrip("/").removeprefix("https://github.com/")
        return f"https://codeload.github.com/{match}/tar.gz/{commit}"

    def _move_atomically(self, staging: Path, final_dir: Path) -> None:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging, final_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        # Consume the temp workspace the staging dir lived in.
        parent = staging.parent
        if parent.name.startswith(".install-"):
            shutil.rmtree(parent, ignore_errors=True)

    @staticmethod
    def _safe_extract_members(
        tf: tarfile.TarFile, members: list[tarfile.TarInfo], dest: Path
    ) -> None:
        """Reject absolute paths and ``..`` escapes before extracting."""
        dest_resolved = dest.resolve()
        for member in members:
            target = (dest_resolved / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep):
                raise SkillInstallError(
                    f"unsafe path in package archive: {member.name!r}"
                )
        tf.extractall(dest_resolved, members=members)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch(url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise SkillInstallError(f"fetch failed for {url}: {exc}") from exc

    def _extract_atomically(self, payload: bytes, final_dir: Path, ref: str) -> None:
        """Extract to a temp dir first, then rename into place (doc §11)."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=self.skills_dir, prefix=".install-"))
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_tar:
                tmp_tar.write(payload)
                tmp_tar_path = tmp_tar.name
            try:
                with tarfile.open(tmp_tar_path, "r:gz") as tf:
                    self._safe_extract_members(tf, tf.getmembers(), tmp_dir)
            finally:
                os.unlink(tmp_tar_path)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    @staticmethod
    def _validate_package(pkg_dir: Path, ref: str) -> None:
        manifest_path = pkg_dir / "skill.yaml"
        # Packages may nest one directory level (e.g. tarball root folder).
        if not manifest_path.exists():
            candidates = list(pkg_dir.glob("*/skill.yaml"))
            if len(candidates) == 1:
                nested = candidates[0].parent
                for item in nested.iterdir():
                    shutil.move(str(item), str(pkg_dir / item.name))
                nested.rmdir()
            else:
                raise SkillInstallError(
                    f"package {ref!r} has no skill.yaml at the expected location"
                )
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise SkillInstallError(f"package {ref!r} manifest is not valid YAML: {exc}") from exc
        metadata = data.get("metadata", {}) or {}
        if not metadata.get("name"):
            raise SkillInstallError(f"package {ref!r} manifest lacks metadata.name")

    def _write_lockfile(self, hit: CatalogHit, receipt: InstallReceipt, url: str) -> None:
        data: dict = {}
        if self.lockfile.exists():
            try:
                data = json.loads(self.lockfile.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("installed.lock.json unreadable; rewriting")
                data = {}
        data[receipt.name] = {
            "version": receipt.version,
            "source_url": url,
            "source_commit": receipt.source_commit,
            "package_digest": receipt.package_digest,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trust": receipt.trust,
            "verification_status": hit.verification_status,
        }
        fd, tmp = tempfile.mkstemp(dir=self.skills_dir, prefix=".lock-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.lockfile)
