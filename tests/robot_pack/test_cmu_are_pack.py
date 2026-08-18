from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rosclaw.robot_pack.runtime_loader import load_cmu_are_simulation_pack
from rosclaw.robot_pack.verifier import verify_robot_pack


def test_cmu_are_pack_is_signed_and_shadow_only() -> None:
    pack = Path(__file__).parents[2] / "src/rosclaw/robot_pack/packs/cmu-are-sim"
    verification = verify_robot_pack(pack)
    assert verification.ok is True
    assert verification.trusted is True
    assert all(
        "REAL" not in capability.execution_modes
        for capability in verification.manifest.capabilities
    )


def test_builtin_loader_registers_only_shadow() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls = []

        def register_executor(self, capability, mode, executor) -> None:
            self.calls.append((capability, mode.value, executor))

    runtime = SimpleNamespace(action_gateway=Gateway())
    status = load_cmu_are_simulation_pack(runtime)
    assert status["instance_id"] == "cmu_are_sim"
    assert all(mode == "SHADOW" for _, mode, _ in runtime.action_gateway.calls)
