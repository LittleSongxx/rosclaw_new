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


@dataclass
class InstallReceipt:
    name: str
    version: str
    package_digest: str
    install_dir: Path
    trust: str  # official | official_signed | third_party

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "package_digest": self.package_digest,
            "install_dir": str(self.install_dir),
            "trust": self.trust,
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
        url = str(source.get("url", ""))
        if not url:
            raise SkillInstallError(f"skill {ref!r} has no fetchable source URL")
        expected_digest = str((hit.raw or {}).get("checksums", {}).get("package_sha256", ""))
        if not expected_digest:
            raise SkillInstallError(
                f"skill {ref!r} registry entry has no package_sha256; refusing to "
                f"install an unpinned executable package (doc §12)"
            )

        payload = self._fetch(url)
        actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise SkillInstallError(
                f"package digest mismatch for {ref!r}: registry pins "
                f"{expected_digest}, fetched {actual_digest}; refusing to extract"
            )

        version = hit.version or "0.0.0"
        final_dir = self.skills_dir / ref / version
        self._extract_atomically(payload, final_dir, ref)
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
        )
        self._write_lockfile(hit, receipt, url)
        logger.info("installed %s@%s (%s)", ref, version, actual_digest[:19])
        return receipt

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
                    self._safe_extract(tf, tmp_dir)
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
    def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
        """Reject absolute paths and ``..`` escapes before extracting."""
        dest_resolved = dest.resolve()
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep):
                raise SkillInstallError(
                    f"unsafe path in package archive: {member.name!r}"
                )
        tf.extractall(dest_resolved)

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
            "package_digest": receipt.package_digest,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trust": receipt.trust,
            "verification_status": hit.verification_status,
        }
        fd, tmp = tempfile.mkstemp(dir=self.skills_dir, prefix=".lock-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.lockfile)
