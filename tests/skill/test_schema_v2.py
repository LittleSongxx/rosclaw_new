"""GREEN — core accepts rosclaw.skill.v2 manifests (doc §7).

v2 adds capability/permissions/safety sections and a v2 schema marker
without breaking the v1 format; the golden ros_install 0.2.0 package
carries both v1 and v2 sections.
"""

from __future__ import annotations

import yaml

from rosclaw.skill.models import SkillYaml
from tests.skill.conftest import ROS_INSTALL_V2_MANIFEST

GOLDEN_LIKE = """\
schema_version: rosclaw.skill.v2
kind: Skill
metadata:
  name: ros_install
  namespace: ros-claw
  version: 0.2.0
identity:
  skill_id: ros-claw/ros_install
  git_ref: main
task:
  intent: Install ROS 2.
execution:
  domain: host
  planner:
    type: python
    entrypoint: entrypoint.py:plan
  verifier:
    type: python
    entrypoint: entrypoint.py:verify
  entrypoint:
    type: behavior_tree
    file: behavior_tree.xml
compatibility:
  eurdf: e-urdf-compat.yaml
  os: [ubuntu]
  architectures: [amd64, arm64]
capability:
  id: environment.install.ros
  intents:
    zh: [安装 ROS2]
permissions:
  - host.package.install
safety:
  approval_scope: transaction
  privilege: admin
  arbitrary_root_shell: false
status:
  official: true
  installable: true
  verification_status: host_matrix_verified
"""


def test_v2_minimal_manifest_validates():
    model = SkillYaml.model_validate(yaml.safe_load(ROS_INSTALL_V2_MANIFEST))
    assert model.schema_version == "rosclaw.skill.v2"
    assert model.metadata.name == "ros_install"
    assert model.capability is not None
    assert model.capability.id == "environment.install.ros"
    assert model.capability.intents["zh"]
    assert model.safety is not None
    assert model.safety.arbitrary_root_shell is False
    assert "host.package.install" in model.permissions


def test_v2_golden_like_manifest_with_v1_sections_validates():
    raw = yaml.safe_load(GOLDEN_LIKE)
    model = SkillYaml.model_validate(raw)
    assert model.schema_version == "rosclaw.skill.v2"
    assert model.identity.skill_id == "ros-claw/ros_install"
    assert model.compatibility.eurdf == "e-urdf-compat.yaml"
    # status.official/installable/verification_status ride the raw manifest
    # through to the registry builder; they are intentionally not pydantic
    # fields yet.
    assert raw["status"]["verification_status"] == "host_matrix_verified"


def test_v1_manifest_still_validates():
    model = SkillYaml.model_validate(
        {"schema_version": "rosclaw.skill.v1", "metadata": {"name": "legacy", "version": "1.0.0"}}
    )
    assert model.schema_version == "rosclaw.skill.v1"
    assert model.capability is None
