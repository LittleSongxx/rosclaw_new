"""Bounded dream-campaign contracts and offline control plane.

The replay consolidation service is imported lazily so lightweight
consumers (contracts, control) do not require the optional training
stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rosclaw.dream.contracts import (
    DreamBudget,
    DreamCampaign,
    DreamEpisode,
    DreamType,
)
from rosclaw.dream.control import (
    DreamBudgetExceededError,
    DreamBudgetUsage,
    DreamCampaignState,
    DreamCampaignStatus,
    DreamLease,
    DreamPlanner,
    DreamPlanReceipt,
    DreamPlanRequest,
    DreamScheduler,
    dream_doctor,
    inspect_dream_journal,
)

if TYPE_CHECKING:
    from rosclaw.dream.replay import (
        BlindEvaluationEvidence,
        BlindGateEvaluator,
        BlindGateReport,
        CandidateFreezer,
        FrozenDreamCandidate,
        ReplayConsolidationAdapter,
        ReplayConsolidationReceipt,
        ReplayConsolidationResult,
        ReplayConsolidationStatus,
        ReplayDreamRunReceipt,
        ReplayDreamRunResult,
        ReplayDreamService,
        ReplaySnapshot,
        ReplaySnapshotBuilder,
        ReplayWorkset,
    )

__all__ = [
    "BlindEvaluationEvidence",
    "BlindGateEvaluator",
    "BlindGateReport",
    "CandidateFreezer",
    "DreamBudget",
    "DreamBudgetExceededError",
    "DreamBudgetUsage",
    "DreamCampaign",
    "DreamCampaignState",
    "DreamCampaignStatus",
    "DreamEpisode",
    "DreamLease",
    "DreamPlanReceipt",
    "DreamPlanRequest",
    "DreamPlanner",
    "DreamScheduler",
    "DreamType",
    "FrozenDreamCandidate",
    "ReplayConsolidationAdapter",
    "ReplayConsolidationReceipt",
    "ReplayConsolidationResult",
    "ReplayConsolidationStatus",
    "ReplayDreamRunReceipt",
    "ReplayDreamRunResult",
    "ReplayDreamService",
    "ReplaySnapshot",
    "ReplaySnapshotBuilder",
    "ReplayWorkset",
    "dream_doctor",
    "inspect_dream_journal",
]

_REPLAY_EXPORTS = {
    "BlindEvaluationEvidence",
    "BlindGateEvaluator",
    "BlindGateReport",
    "CandidateFreezer",
    "FrozenDreamCandidate",
    "ReplayConsolidationAdapter",
    "ReplayConsolidationReceipt",
    "ReplayConsolidationResult",
    "ReplayConsolidationStatus",
    "ReplayDreamRunReceipt",
    "ReplayDreamRunResult",
    "ReplayDreamService",
    "ReplaySnapshot",
    "ReplaySnapshotBuilder",
    "ReplayWorkset",
}


def __getattr__(name: str) -> Any:
    if name in _REPLAY_EXPORTS:
        from rosclaw.dream import replay

        return getattr(replay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
