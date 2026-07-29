from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_continual_gpu_smoke import (
    ContinualGPUShard,
    G1ContinualFourGPUSmoke,
    run_g1_continual_four_gpu_smoke,
)


def _shard(gpu: int) -> ContinualGPUShard:
    return ContinualGPUShard(
        physical_gpu=gpu,
        gpu_uuid=f"GPU-{gpu}",
        candidate_policy_hash="sha256:" + str(gpu) * 64,
        updates_finite=True,
        stale_actor_transition_count=16,
        action_bounded=True,
        hidden_activations_finite=True,
        max_memory_allocated_bytes=1024,
        evidence_path=f"/evidence/gpu-{gpu}.json",
    )


def test_four_gpu_smoke_requires_four_distinct_finite_cuda_shards() -> None:
    result = G1ContinualFourGPUSmoke(
        shards=tuple(_shard(gpu) for gpu in range(4)),
        failures=(),
    )

    assert result.passed
    assert result.to_dict()["claims"]["g1_motion_effect_proven"] is False
    assert result.to_dict()["promotion_assessment"]["eligible"] is False


def test_four_gpu_smoke_rejects_evidence_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_continual_four_gpu_smoke(
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )
