"""Sanity checks for the shared skill fixtures themselves (not xfail-marked).

These guard the hermetic fixture registry so a broken fixture is not
mistaken for a RED signal.
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse
from urllib.request import url2pathname


def test_fixture_registry_digest_matches_tarball(official_registry):
    payload = json.loads(official_registry.read_text(encoding="utf-8"))
    entry = payload["skills"][0]
    url = entry["source"]["url"]
    assert url.startswith("file://")
    path = url2pathname(urlparse(url).path)
    with open(path, "rb") as fh:
        digest = "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    assert digest == entry["checksums"]["package_sha256"]


def test_fixture_manifest_is_valid_yaml(official_registry):
    import yaml

    from tests.skill.conftest import ROS_INSTALL_V2_MANIFEST

    manifest = yaml.safe_load(ROS_INSTALL_V2_MANIFEST)
    assert manifest["schema_version"] == "rosclaw.skill.v2"
    assert manifest["capability"]["id"] == "environment.install.ros"
    assert manifest["execution"]["domain"] == "host"
    assert manifest["safety"]["arbitrary_root_shell"] is False
