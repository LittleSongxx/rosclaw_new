"""Post-hoc persistent-subnetwork discovery on shared reference states.

This module implements the non-causal analysis portion of Jhunjhunwala,
Goldfeder, and Lipson (2026): dead-unit filtering, co-activation graphs,
permutation-aware neuron-family alignment, and persistence scoring.  Its output
is intentionally named a *candidate*.  ROSClaw may call it SelfCore only after
multi-seed controls and matched freeze/lesion interventions pass the separate
stability--plasticity gate.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class PersistentSubnetworkCandidate:
    source_policy_hash: str
    target_policy_hash: str
    shared_reference_hash: str
    layer_name: str
    threshold: float
    alive_unit_count: int
    dead_unit_count: int
    source_for_target: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    persistent_candidate_units: tuple[int, ...]
    task_candidate_units: tuple[int, ...]
    persistence_scores: tuple[float, ...]
    persistent_mean: float
    task_mean: float
    persistence_gap: float
    source_activation_hash: str
    target_activation_hash: str
    causal_validated: bool = False
    schema_version: str = "rosclaw.self.persistent_subnetwork_candidate.v1"

    def __post_init__(self) -> None:
        if self.causal_validated:
            raise ValueError("discovery analysis cannot self-assert causal validation")
        for label, value in (
            ("source_policy_hash", self.source_policy_hash),
            ("target_policy_hash", self.target_policy_hash),
            ("shared_reference_hash", self.shared_reference_hash),
            ("source_activation_hash", self.source_activation_hash),
            ("target_activation_hash", self.target_activation_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")
        if not self.layer_name.strip():
            raise ValueError("layer_name must not be empty")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("co-activation threshold must be in (0, 1]")
        if not self.persistent_candidate_units or not self.task_candidate_units:
            raise ValueError("persistent/task decomposition must be non-degenerate")
        if self.alive_unit_count <= 0 or self.dead_unit_count < 0:
            raise ValueError("alive/dead unit counts are invalid")
        unit_count = self.alive_unit_count + self.dead_unit_count
        if len(self.source_for_target) != unit_count or len(self.persistence_scores) != unit_count:
            raise ValueError("alignment and persistence scores must cover every hidden unit")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.persistence_scores
        ):
            raise ValueError("persistence scores must be finite values in [0, 1]")
        if any(
            not math.isfinite(value)
            for value in (self.persistent_mean, self.task_mean, self.persistence_gap)
        ):
            raise ValueError("persistence summary values must be finite")

    @property
    def candidate_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_policy_hash": self.source_policy_hash,
            "target_policy_hash": self.target_policy_hash,
            "shared_reference_hash": self.shared_reference_hash,
            "layer_name": self.layer_name,
            "threshold": self.threshold,
            "alive_unit_count": self.alive_unit_count,
            "dead_unit_count": self.dead_unit_count,
            "source_for_target": list(self.source_for_target),
            "components": [list(component) for component in self.components],
            "persistent_candidate_units": list(self.persistent_candidate_units),
            "task_candidate_units": list(self.task_candidate_units),
            "persistence_scores": list(self.persistence_scores),
            "persistent_mean": self.persistent_mean,
            "task_mean": self.task_mean,
            "persistence_gap": self.persistence_gap,
            "source_activation_hash": self.source_activation_hash,
            "target_activation_hash": self.target_activation_hash,
            "causal_validated": self.causal_validated,
        }


@dataclass(frozen=True)
class ThresholdSensitivity:
    candidates: tuple[PersistentSubnetworkCandidate, ...]
    positive_gap_fraction: float
    median_gap: float
    stable: bool
    schema_version: str = "rosclaw.self.threshold_sensitivity.v1"

    def __post_init__(self) -> None:
        if len(self.candidates) < 3:
            raise ValueError("threshold sensitivity requires at least three thresholds")
        if len({candidate.shared_reference_hash for candidate in self.candidates}) != 1:
            raise ValueError("threshold candidates must share one reference-state bank")

    @property
    def sweep_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "candidate_hashes": [candidate.candidate_hash for candidate in self.candidates],
                "positive_gap_fraction": self.positive_gap_fraction,
                "median_gap": self.median_gap,
                "stable": self.stable,
            }
        )


def discover_persistent_subnetwork(
    *,
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    source_policy_hash: str,
    target_policy_hash: str,
    shared_reference_hash: str,
    layer_name: str,
    threshold: float = 0.70,
    dead_unit_epsilon: float = 1e-8,
) -> PersistentSubnetworkCandidate:
    """Discover a candidate persistent block without making a causal claim."""

    source = _activation_matrix(source_activations, "source_activations")
    target = _activation_matrix(target_activations, "target_activations")
    if source.shape != target.shape:
        raise ValueError("source and target activation matrices must have the same shape")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if dead_unit_epsilon <= 0.0:
        raise ValueError("dead_unit_epsilon must be positive")

    source_normalized, source_alive = _normalize_units(source, dead_unit_epsilon)
    target_normalized, target_alive = _normalize_units(target, dead_unit_epsilon)
    jointly_alive = source_alive & target_alive
    alive_indices = np.flatnonzero(jointly_alive)
    if alive_indices.size < 2:
        raise ValueError("at least two jointly alive units are required")

    family_similarity = _cosine_columns(source_normalized, target_normalized)
    source_for_target = _source_for_target(family_similarity)
    aligned_source = source_normalized[:, source_for_target]
    aligned_alive = source_alive[source_for_target] & target_alive
    target_coactivation = np.abs(_cosine_columns(target_normalized, target_normalized))
    source_coactivation = np.abs(_cosine_columns(aligned_source, aligned_source))
    components = _components(target_coactivation, aligned_alive, threshold)
    if len(components) < 2:
        raise ValueError("co-activation decomposition is degenerate at this threshold")
    persistent = max(components, key=lambda value: (len(value), tuple(-item for item in value)))
    task = tuple(sorted(set(map(int, np.flatnonzero(aligned_alive))).difference(persistent)))
    if not task:
        raise ValueError("co-activation decomposition has no task-like units")

    activation_similarity = np.asarray(
        [
            max(0.0, _cosine_vector(aligned_source[:, unit], target_normalized[:, unit]))
            if aligned_alive[unit]
            else 0.0
            for unit in range(target.shape[1])
        ],
        dtype=np.float64,
    )
    connectivity_similarity = np.asarray(
        [
            max(0.0, _cosine_vector(source_coactivation[unit], target_coactivation[unit]))
            if aligned_alive[unit]
            else 0.0
            for unit in range(target.shape[1])
        ],
        dtype=np.float64,
    )
    persistence = np.clip(
        0.5 * (activation_similarity + connectivity_similarity),
        0.0,
        1.0,
    )
    persistent_mean = float(np.mean(persistence[list(persistent)]))
    task_mean = float(np.mean(persistence[list(task)]))
    return PersistentSubnetworkCandidate(
        source_policy_hash=source_policy_hash,
        target_policy_hash=target_policy_hash,
        shared_reference_hash=shared_reference_hash,
        layer_name=layer_name,
        threshold=threshold,
        alive_unit_count=int(np.count_nonzero(aligned_alive)),
        dead_unit_count=int(target.shape[1] - np.count_nonzero(aligned_alive)),
        source_for_target=tuple(map(int, source_for_target)),
        components=components,
        persistent_candidate_units=persistent,
        task_candidate_units=task,
        persistence_scores=tuple(map(float, persistence)),
        persistent_mean=persistent_mean,
        task_mean=task_mean,
        persistence_gap=persistent_mean - task_mean,
        source_activation_hash=_array_hash(source),
        target_activation_hash=_array_hash(target),
    )


def sweep_thresholds(
    *,
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    source_policy_hash: str,
    target_policy_hash: str,
    shared_reference_hash: str,
    layer_name: str,
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
) -> ThresholdSensitivity:
    if len(thresholds) < 3 or len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold sweep must contain at least three unique values")
    candidates = tuple(
        discover_persistent_subnetwork(
            source_activations=source_activations,
            target_activations=target_activations,
            source_policy_hash=source_policy_hash,
            target_policy_hash=target_policy_hash,
            shared_reference_hash=shared_reference_hash,
            layer_name=layer_name,
            threshold=threshold,
        )
        for threshold in thresholds
    )
    gaps = np.asarray([candidate.persistence_gap for candidate in candidates], dtype=np.float64)
    fraction = float(np.mean(gaps > 0.0))
    median = float(np.median(gaps))
    return ThresholdSensitivity(
        candidates=candidates,
        positive_gap_fraction=fraction,
        median_gap=median,
        stable=bool(fraction >= 0.80 and median >= 0.05),
    )


def _activation_matrix(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"{label} must have shape [reference_states, hidden_units]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(array)


def _normalize_units(value: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    centered = value - np.mean(value, axis=0, keepdims=True)
    scale = np.std(centered, axis=0)
    alive = scale > epsilon
    normalized = np.zeros_like(centered)
    normalized[:, alive] = centered[:, alive] / scale[alive]
    return normalized, alive


def _cosine_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = left.T @ right
    left_norm = np.linalg.norm(left, axis=0)
    right_norm = np.linalg.norm(right, axis=0)
    denominator = np.outer(left_norm, right_norm)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _cosine_vector(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _source_for_target(similarity: np.ndarray) -> np.ndarray:
    """Maximum-weight one-to-one neuron alignment via Hungarian assignment."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("neuron-family similarity must be square")
    size = similarity.shape[0]
    cost = float(np.max(similarity)) - similarity
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        matched_row[0] = row
        column0 = 0
        minimum = np.full(size + 1, math.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = int(matched_row[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = float(minimum[column])
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    source_for_target = np.empty(size, dtype=np.int64)
    for target in range(1, size + 1):
        source_for_target[target - 1] = matched_row[target] - 1
    return source_for_target


def _components(
    affinity: np.ndarray,
    alive: np.ndarray,
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    remaining = set(map(int, np.flatnonzero(alive)))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            unit = stack.pop()
            if unit in component:
                continue
            component.add(unit)
            neighbors = {
                int(index) for index in np.flatnonzero(affinity[unit] >= threshold) if alive[index]
            }
            stack.extend(sorted(neighbors.difference(component), reverse=True))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda value: (-len(value), value)))


def _array_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return "sha256:" + digest.hexdigest()


__all__ = [
    "PersistentSubnetworkCandidate",
    "ThresholdSensitivity",
    "discover_persistent_subnetwork",
    "sweep_thresholds",
]
