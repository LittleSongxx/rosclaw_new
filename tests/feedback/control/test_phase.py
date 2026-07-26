from __future__ import annotations

from rosclaw.feedback.controllers.phase import LatchedPhaseGate


def test_phase_gate_rejects_late_trigger_and_latches_eligible_trigger() -> None:
    gate = LatchedPhaseGate(
        active_start=0.2,
        trigger_end=0.5,
        active_end=0.8,
        fade_fraction=0.1,
    )

    assert gate.update(phase=0.3, trigger=False) == 0.0
    assert gate.update(phase=0.4, trigger=True) == 1.0
    assert gate.update(phase=0.6, trigger=False) == 1.0
    assert 0.0 < gate.update(phase=0.75, trigger=False) < 1.0
    gate.reset()
    assert gate.update(phase=0.6, trigger=True) == 0.0
