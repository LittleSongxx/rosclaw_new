from __future__ import annotations

from rosclaw.continual.contracts import SkillPhase
from rosclaw.continual.stability import StabilityPlasticityGate
from rosclaw.continual.weight_update import ResidualWeightSlot, WeightSlotState
from tests.continual.helpers import policy
from tests.continual.test_stability import _passing_evidence


def test_weight_update_is_atomic_only_at_safe_motion_boundary() -> None:
    v0, artifact0 = policy(0)
    v1, artifact1 = policy(1, parent=v0)
    slot = ResidualWeightSlot(v0, active_artifact=artifact0)
    staged = slot.stage(v1, artifact=artifact1)
    evidence = _passing_evidence()
    evidence = type(evidence)(
        **{
            **evidence.__dict__,
            "parent_policy_hash": v0.artifact_hash,
            "candidate_policy_hash": v1.artifact_hash,
        }
    )
    report = StabilityPlasticityGate().evaluate(evidence)

    activated = slot.activate(phase=SkillPhase.PREPARE, gate_report=report)

    assert staged.state is WeightSlotState.CANDIDATE_STAGED
    assert activated.state is WeightSlotState.ACTIVE
    assert slot.active == v1
    assert not activated.hardware_authorized


def test_mid_swing_weight_swap_freezes_instead_of_switching() -> None:
    v0, artifact0 = policy(0)
    v1, artifact1 = policy(1, parent=v0)
    slot = ResidualWeightSlot(v0, active_artifact=artifact0)
    slot.stage(v1, artifact=artifact1)
    evidence = _passing_evidence()
    evidence = type(evidence)(
        **{
            **evidence.__dict__,
            "parent_policy_hash": v0.artifact_hash,
            "candidate_policy_hash": v1.artifact_hash,
        }
    )
    report = StabilityPlasticityGate().evaluate(evidence)

    receipt = slot.activate(phase=SkillPhase.SWING, gate_report=report)

    assert receipt.state is WeightSlotState.FROZEN
    assert slot.active == v0
    assert slot.candidate == v1
