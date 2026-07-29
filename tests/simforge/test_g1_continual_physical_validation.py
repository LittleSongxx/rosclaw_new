from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_continual_physical_validation import (
    G1ContinualPhysicalFoundation,
    run_g1_continual_physical_foundation,
)


def test_physical_continual_foundation_rejects_evidence_inside_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_continual_physical_foundation(
            asset_root=tmp_path / "missing",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


def test_failed_worker_path_still_emits_a_safe_nonpassing_summary() -> None:
    result = G1ContinualPhysicalFoundation(
        body_hash="sha256:" + "1" * 64,
        kick_prior_hash="sha256:" + "2" * 64,
        backend_commit="abc",
        rollouts=(),
        shards=(),
        failures=("gpu0:failed",),
        gate_report={"decision": "NEED_MORE_EVIDENCE", "activation_allowed": False},
        stage_receipt={},
        activation_receipt={},
        active_policy_unchanged=False,
    )

    summary = result.to_dict()

    assert not result.passed
    assert not summary["safe_activation_refusal"]
    assert summary["failures"] == ["gpu0:failed"]
