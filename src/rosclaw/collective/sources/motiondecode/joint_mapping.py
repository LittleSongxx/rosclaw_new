"""Exact MotionDecode CSV to ROSClaw Unitree HG joint mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

MOTIONDECODE_ROOT_COLUMNS = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
)
MOTIONDECODE_TIME_COLUMNS = ("time(s)", "time_sec")
_JOINT_COLUMN = re.compile(r"^dof_([a-z0-9_]+_joint)\(rad\)$")


@dataclass(frozen=True)
class MotionDecodeJointMapping:
    source_joint_names: tuple[str, ...]
    target_joint_names: tuple[str, ...]
    source_indices_by_target: tuple[int, ...]
    exact_order: bool
    source_unit: str = "rad"
    target_contract: str = "unitree_hg_dds_29"
    schema_version: str = "rosclaw.collective.motiondecode_joint_mapping.v1"

    def __post_init__(self) -> None:
        source = tuple(self.source_joint_names)
        target = tuple(self.target_joint_names)
        indices = tuple(self.source_indices_by_target)
        if not source or len(source) != len(set(source)):
            raise ValueError("source_joint_names must be non-empty and unique")
        if not target or len(target) != len(set(target)):
            raise ValueError("target_joint_names must be non-empty and unique")
        if set(source) != set(target):
            missing = sorted(set(target) - set(source))
            extra = sorted(set(source) - set(target))
            raise ValueError(f"joint mapping is incomplete; missing={missing}, extra={extra}")
        if len(indices) != len(target) or set(indices) != set(range(len(source))):
            raise ValueError("source_indices_by_target must be a complete permutation")
        if self.exact_order != (source == target):
            raise ValueError("exact_order does not match the declared joint sequences")
        object.__setattr__(self, "source_joint_names", source)
        object.__setattr__(self, "target_joint_names", target)
        object.__setattr__(self, "source_indices_by_target", indices)

    @property
    def mapping_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_joint_names": list(self.source_joint_names),
            "target_joint_names": list(self.target_joint_names),
            "source_indices_by_target": list(self.source_indices_by_target),
            "exact_order": self.exact_order,
            "source_unit": self.source_unit,
            "target_contract": self.target_contract,
        }


def mapping_from_header(
    header: tuple[str, ...],
    *,
    target_joint_names: tuple[str, ...] = G1_DDS_JOINT_NAMES,
) -> tuple[MotionDecodeJointMapping, str | None]:
    """Validate the complete CSV header and return an explicit permutation."""

    if not header or len(header) != len(set(header)):
        raise ValueError("MotionDecode CSV header must be non-empty and unique")
    offset = 0
    time_column: str | None = None
    if header[0] in MOTIONDECODE_TIME_COLUMNS:
        time_column = header[0]
        offset = 1
    root = header[offset : offset + len(MOTIONDECODE_ROOT_COLUMNS)]
    if root != MOTIONDECODE_ROOT_COLUMNS:
        raise ValueError(
            "MotionDecode root schema must declare xyz in metres and quaternion in wxyz order"
        )
    joint_columns = header[offset + len(MOTIONDECODE_ROOT_COLUMNS) :]
    source_names: list[str] = []
    for column in joint_columns:
        match = _JOINT_COLUMN.fullmatch(column)
        if match is None:
            raise ValueError(f"unexpected MotionDecode CSV column: {column}")
        source_names.append(match.group(1))
    source = tuple(source_names)
    target = tuple(target_joint_names)
    source_index = {name: index for index, name in enumerate(source)}
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(
            f"MotionDecode G1 joint contract mismatch; missing={missing}, extra={extra}"
        )
    mapping = MotionDecodeJointMapping(
        source_joint_names=source,
        target_joint_names=target,
        source_indices_by_target=tuple(source_index[name] for name in target),
        exact_order=source == target,
    )
    return mapping, time_column


__all__ = [
    "MOTIONDECODE_ROOT_COLUMNS",
    "MOTIONDECODE_TIME_COLUMNS",
    "MotionDecodeJointMapping",
    "mapping_from_header",
]
