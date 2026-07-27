"""Versioned continual-learning substrate for ROSClaw simulation."""

from rosclaw.continual.contracts import (
    ControlSegment,
    CostVector,
    ExperiencePartition,
    ExperienceUse,
    PolicyVersion,
    RewardVector,
    SkillPhase,
    VersionedTrajectory,
)
from rosclaw.continual.experience import (
    ContinualExperienceStore,
    ExperienceBatch,
    ExperienceBufferConfig,
    ExperienceRecord,
    ReplayMix,
)
from rosclaw.continual.stability import (
    ContinualCandidateEvidence,
    ContinualDecision,
    ContinualGateReport,
    PlasticityEvidence,
    SelfCoreEvidence,
    StabilityPlasticityGate,
    TaskRetention,
)
from rosclaw.continual.weight_update import ResidualWeightSlot, WeightSlotReceipt, WeightSlotState

__all__ = [
    "ControlSegment",
    "CostVector",
    "ExperiencePartition",
    "ExperienceUse",
    "PolicyVersion",
    "RewardVector",
    "SkillPhase",
    "VersionedTrajectory",
    "ContinualCandidateEvidence",
    "ContinualDecision",
    "ContinualExperienceStore",
    "ContinualGateReport",
    "ExperienceBatch",
    "ExperienceBufferConfig",
    "ExperienceRecord",
    "PlasticityEvidence",
    "ReplayMix",
    "ResidualWeightSlot",
    "SelfCoreEvidence",
    "StabilityPlasticityGate",
    "TaskRetention",
    "WeightSlotReceipt",
    "WeightSlotState",
]
