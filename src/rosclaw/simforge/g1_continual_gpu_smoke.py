"""Four-GPU systems smoke for versioned continual residual SAC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContinualGPUShard:
    physical_gpu: int
    gpu_uuid: str
    candidate_policy_hash: str
    updates_finite: bool
    stale_actor_transition_count: int
    action_bounded: bool
    hidden_activations_finite: bool
    max_memory_allocated_bytes: int
    evidence_path: str


@dataclass(frozen=True)
class G1ContinualFourGPUSmoke:
    shards: tuple[ContinualGPUShard, ...]
    failures: tuple[str, ...]
    schema_version: str = "rosclaw.continual.four_gpu_smoke.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.shards) == 4
            and not self.failures
            and len({shard.gpu_uuid for shard in self.shards}) == 4
            and all(
                shard.updates_finite
                and shard.stale_actor_transition_count > 0
                and shard.action_bounded
                and shard.hidden_activations_finite
                and shard.max_memory_allocated_bytes > 0
                for shard in self.shards
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shards": [shard.__dict__ for shard in self.shards],
            "failures": list(self.failures),
            "unique_gpu_uuids": len({shard.gpu_uuid for shard in self.shards}),
            "passed": self.passed,
            "promotion_assessment": {
                "status": "NEED_MORE_EVIDENCE",
                "eligible": False,
                "blockers": [
                    "worker transitions are synthetic contract fixtures, not G1 physics rollouts",
                    "multi-seed retention, plasticity, and SelfCore causal gates are incomplete",
                    "no candidate is staged or activated by this smoke test",
                ],
            },
            "claims": {
                "evidence_domain": "CUDA_SCREENING",
                "four_physical_gpus_exercised": len(self.shards) == 4,
                "g1_motion_effect_proven": False,
                "hardware_authorized": False,
            },
        }


def run_g1_continual_four_gpu_smoke(
    *,
    output_dir: Path,
    source_checkout: Path,
    updates_per_gpu: int = 3,
) -> G1ContinualFourGPUSmoke:
    """Run one isolated learner process per physical GPU and aggregate evidence."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("continual four-GPU evidence must be outside source checkout")
    if not 1 <= updates_per_gpu <= 20:
        raise ValueError("updates_per_gpu must be in [1, 20]")
    root.mkdir(parents=True, exist_ok=False)
    worker = checkout / "scripts/simforge/g1_continual_gpu_worker.py"
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    for physical_gpu in range(4):
        output = root / f"gpu-{physical_gpu}-learner.json"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        process = subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--physical-gpu",
                str(physical_gpu),
                "--updates",
                str(updates_per_gpu),
                "--output",
                str(output),
            ],
            cwd=checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((physical_gpu, output, process))
    shards: list[ContinualGPUShard] = []
    failures: list[str] = []
    for physical_gpu, output, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=120.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            failures.append(f"gpu{physical_gpu}:timeout:{stdout[-200:]}:{stderr[-400:]}")
            continue
        if process.returncode != 0 or not output.is_file():
            failures.append(
                f"gpu{physical_gpu}:exit={process.returncode}:{stdout[-200:]}:{stderr[-600:]}"
            )
            continue
        value = json.loads(output.read_text(encoding="utf-8"))
        shards.append(
            ContinualGPUShard(
                physical_gpu=physical_gpu,
                gpu_uuid=str(value["gpu_uuid"]),
                candidate_policy_hash=str(value["candidate_policy_hash"]),
                updates_finite=bool(value["updates_finite"]),
                stale_actor_transition_count=int(value["stale_actor_transition_count"]),
                action_bounded=bool(value["action_bounded"]),
                hidden_activations_finite=bool(value["hidden_activations_finite"]),
                max_memory_allocated_bytes=int(value["max_memory_allocated_bytes"]),
                evidence_path=str(output),
            )
        )
    result = G1ContinualFourGPUSmoke(
        shards=tuple(sorted(shards, key=lambda item: item.physical_gpu)),
        failures=tuple(failures),
    )
    _atomic_json(root / "four-gpu-continual-summary.json", result.to_dict())
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "ContinualGPUShard",
    "G1ContinualFourGPUSmoke",
    "run_g1_continual_four_gpu_smoke",
]
