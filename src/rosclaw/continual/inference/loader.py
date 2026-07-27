"""Fail-closed, read-only loader for residual SAC actor artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from rosclaw.continual.contracts import PolicyVersion

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_SCHEMA_V1 = "rosclaw.continual.residual_sac_artifact.v1"


@dataclass(frozen=True)
class ResidualCandidateArtifact:
    """Verified inference-only view of one content-addressed actor."""

    policy: PolicyVersion
    parent: PolicyVersion
    artifact_hash: str
    schema_version: str
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_limits: tuple[float, ...]
    hidden_dims: tuple[int, int]
    update_index: int
    tensors: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


def load_residual_candidate(
    artifact_path: Path,
    *,
    policy: PolicyVersion,
    parent: PolicyVersion,
    expected_body_hash: str | None = None,
    expected_observation_names: tuple[str, ...] | None = None,
    expected_action_names: tuple[str, ...] | None = None,
) -> ResidualCandidateArtifact:
    """Load and verify a candidate without touching Registry, DDS, or an active slot."""

    path = artifact_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact is missing: {path}")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_ARTIFACT_BYTES:
        raise ValueError("candidate artifact size is outside the read-only loader limit")
    payload = path.read_bytes()
    artifact_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if artifact_hash != policy.artifact_hash:
        raise ValueError("candidate artifact hash does not match PolicyVersion")
    _verify_lineage(policy=policy, parent=parent, expected_body_hash=expected_body_hash)

    metadata_end = _json_object_end(payload)
    try:
        metadata = json.loads(payload[:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate artifact metadata is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_V1:
        raise ValueError("unsupported residual candidate artifact schema")
    observations = _unique_names(metadata, "observation_names")
    actions = _unique_names(metadata, "action_names")
    action_limits = _positive_floats(metadata, "action_limits", count=len(actions))
    hidden_values = _positive_ints(metadata, "hidden_dims", count=2)
    hidden = (hidden_values[0], hidden_values[1])
    update_index = metadata.get("update_index")
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 0:
        raise ValueError("candidate update_index must be a non-negative integer")
    if observations != policy.observation_names:
        raise ValueError("candidate observation contract does not match PolicyVersion")
    if actions != policy.residual_action_names:
        raise ValueError("candidate residual action contract does not match PolicyVersion")
    if expected_observation_names is not None and observations != expected_observation_names:
        raise ValueError("candidate observation contract does not match the runtime contract")
    if expected_action_names is not None and actions != expected_action_names:
        raise ValueError("candidate action contract does not match the runtime contract")

    tensors = _parse_v1_tensors(payload, metadata_end)
    _verify_actor_tensors(
        tensors,
        observation_count=len(observations),
        action_count=len(actions),
        hidden_dims=hidden,
        action_limits=action_limits,
    )
    return ResidualCandidateArtifact(
        policy=policy,
        parent=parent,
        artifact_hash=artifact_hash,
        schema_version=str(metadata["schema_version"]),
        observation_names=observations,
        action_names=actions,
        action_limits=action_limits,
        hidden_dims=hidden,
        update_index=update_index,
        tensors=tensors,
    )


def _verify_lineage(
    *,
    policy: PolicyVersion,
    parent: PolicyVersion,
    expected_body_hash: str | None,
) -> None:
    if policy.version != parent.version + 1:
        raise ValueError("candidate version is not the direct successor of the parent")
    if policy.parent_version_hash != parent.version_hash:
        raise ValueError("candidate parent version hash does not match the supplied parent")
    for label in ("body_hash", "safety_kernel_hash", "controller_snapshot_hash"):
        if getattr(policy, label) != getattr(parent, label):
            raise ValueError(f"candidate {label} does not match the parent")
    if policy.observation_names != parent.observation_names:
        raise ValueError("candidate observation contract does not match the parent")
    if policy.residual_action_names != parent.residual_action_names:
        raise ValueError("candidate residual action contract does not match the parent")
    if expected_body_hash is not None and policy.body_hash != expected_body_hash:
        raise ValueError("candidate body hash does not match qualified simulation assets")


def _json_object_end(payload: bytes) -> int:
    if not payload or payload[0] != ord("{"):
        raise ValueError("candidate artifact must begin with JSON metadata")
    depth = 0
    in_string = False
    escaped = False
    for index, byte in enumerate(payload):
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            depth += 1
        elif byte == ord("}"):
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                break
    raise ValueError("candidate artifact metadata JSON is incomplete")


def _parse_v1_tensors(payload: bytes, offset: int) -> Mapping[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    cursor = offset
    while cursor < len(payload):
        name_size, cursor = _read_u32(payload, cursor, "tensor name length")
        name_bytes, cursor = _read_bytes(payload, cursor, name_size, "tensor name")
        dtype_size, cursor = _read_u32(payload, cursor, "tensor dtype length")
        dtype_bytes, cursor = _read_bytes(payload, cursor, dtype_size, "tensor dtype")
        ndim, cursor = _read_u32(payload, cursor, "tensor rank")
        if ndim > 4:
            raise ValueError("candidate tensor rank exceeds the inference limit")
        shape: tuple[int, ...] = ()
        if ndim:
            shape_size = 8 * ndim
            shape_bytes, cursor = _read_bytes(payload, cursor, shape_size, "tensor shape")
            shape = tuple(int(value) for value in struct.unpack("!" + "Q" * ndim, shape_bytes))
        try:
            name = name_bytes.decode("utf-8")
            dtype_name = dtype_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("candidate tensor descriptor is not valid text") from exc
        if not name or name in tensors:
            raise ValueError("candidate tensor names must be non-empty and unique")
        if dtype_name != "float32":
            raise ValueError("candidate inference permits only float32 actor tensors")
        if any(dimension <= 0 or dimension > 1_000_000 for dimension in shape):
            raise ValueError("candidate tensor shape is invalid")
        count = math.prod(shape)
        raw, cursor = _read_bytes(payload, cursor, count * 4, f"tensor {name}")
        value = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if not np.all(np.isfinite(value)):
            raise ValueError(f"candidate tensor {name} contains non-finite values")
        value.flags.writeable = False
        tensors[name] = value
    if cursor != len(payload) or not tensors:
        raise ValueError("candidate artifact tensor payload is empty or truncated")
    return MappingProxyType(tensors)


def _verify_actor_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    observation_count: int,
    action_count: int,
    hidden_dims: tuple[int, int],
    action_limits: tuple[float, ...],
) -> None:
    h0, h1 = hidden_dims
    expected = {
        "action_limits": (action_count,),
        "backbone.hidden0.bias": (h0,),
        "backbone.hidden0.weight": (h0, observation_count),
        "backbone.hidden1.bias": (h1,),
        "backbone.hidden1.weight": (h1, h0),
        "backbone.output.bias": (2 * action_count,),
        "backbone.output.weight": (2 * action_count, h1),
    }
    if set(tensors) != set(expected):
        raise ValueError(
            "candidate artifact does not contain the exact deterministic actor tensors"
        )
    for name, shape in expected.items():
        if tensors[name].shape != shape:
            raise ValueError(
                f"candidate tensor {name} has shape {tensors[name].shape}, expected {shape}"
            )
    if not np.allclose(tensors["action_limits"], action_limits, rtol=0.0, atol=1e-8):
        raise ValueError("candidate tensor action limits do not match metadata")


def _read_u32(payload: bytes, cursor: int, label: str) -> tuple[int, int]:
    raw, next_cursor = _read_bytes(payload, cursor, 4, label)
    return struct.unpack("!I", raw)[0], next_cursor


def _read_bytes(payload: bytes, cursor: int, count: int, label: str) -> tuple[bytes, int]:
    if count < 0 or cursor < 0 or cursor + count > len(payload):
        raise ValueError(f"candidate artifact is truncated at {label}")
    return payload[cursor : cursor + count], cursor + count


def _unique_names(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"candidate {key} must be a non-empty list")
    names = tuple(item for item in value if isinstance(item, str) and item.strip())
    if len(names) != len(value) or len(set(names)) != len(names):
        raise ValueError(f"candidate {key} must contain unique non-empty strings")
    return names


def _positive_floats(metadata: Mapping[str, Any], key: str, *, count: int) -> tuple[float, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"candidate {key} has the wrong length")
    normalized = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in normalized):
        raise ValueError(f"candidate {key} must contain positive finite values")
    return normalized


def _positive_ints(metadata: Mapping[str, Any], key: str, *, count: int) -> tuple[int, ...]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"candidate {key} must contain {count} positive integers")
    return tuple(value)


__all__ = ["ResidualCandidateArtifact", "load_residual_candidate"]
