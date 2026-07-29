from __future__ import annotations

import numpy as np
import pytest

from rosclaw.self_model import discover_persistent_subnetwork, sweep_thresholds
from tests.continual.helpers import digest


def _activations() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    states = 256
    body = rng.normal(size=states)
    source_task = rng.normal(size=(states, 3))
    target_task = rng.normal(size=(states, 3))
    source = np.column_stack(
        (
            body,
            body + rng.normal(scale=0.01, size=states),
            -body + rng.normal(scale=0.01, size=states),
            source_task,
        )
    )
    target = np.column_stack(
        (
            body + rng.normal(scale=0.01, size=states),
            body + rng.normal(scale=0.01, size=states),
            -body,
            target_task,
        )
    )
    return source, target


def test_persistent_subnetwork_discovery_is_permutation_aware_but_not_causal() -> None:
    source, target = _activations()
    permutation = np.asarray((3, 0, 4, 1, 5, 2))
    candidate = discover_persistent_subnetwork(
        source_activations=source,
        target_activations=target[:, permutation],
        source_policy_hash=digest("walk"),
        target_policy_hash=digest("kick"),
        shared_reference_hash=digest("shared-reference"),
        layer_name="actor.hidden.0",
        threshold=0.90,
    )

    persistent_original_indices = {
        int(permutation[index]) for index in candidate.persistent_candidate_units
    }
    assert persistent_original_indices == {0, 1, 2}
    assert candidate.persistence_gap > 0.25
    assert not candidate.causal_validated
    assert len(candidate.candidate_hash) == 71


def test_threshold_sweep_requires_a_stable_midrange_decomposition() -> None:
    source, target = _activations()
    result = sweep_thresholds(
        source_activations=source,
        target_activations=target,
        source_policy_hash=digest("walk"),
        target_policy_hash=digest("kick"),
        shared_reference_hash=digest("shared-reference"),
        layer_name="actor.hidden.0",
        thresholds=(0.80, 0.85, 0.90, 0.95),
    )

    assert result.stable
    assert result.positive_gap_fraction == 1.0
    assert result.median_gap > 0.25


def test_discovery_rejects_degenerate_all_connected_network() -> None:
    source = np.column_stack([np.arange(20, dtype=float)] * 4)
    with pytest.raises(ValueError, match="degenerate"):
        discover_persistent_subnetwork(
            source_activations=source,
            target_activations=source,
            source_policy_hash=digest("a"),
            target_policy_hash=digest("b"),
            shared_reference_hash=digest("states"),
            layer_name="hidden",
            threshold=0.70,
        )
