"""Recoverable service boundaries for asynchronous continual learning.

Service implementations are imported lazily so lightweight consumers (for
example ``persistence``) do not require the optional training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.continual.services.experience import ExperienceService
    from rosclaw.continual.services.inference import (
        InferenceService,
        InferenceSlotReceipt,
        MotionVersionLease,
    )
    from rosclaw.continual.services.learner import (
        LearnerProduct,
        LearnerService,
        LearnerServiceReceipt,
        ResidualSACServiceExecutor,
    )
    from rosclaw.continual.services.rollout import (
        RolloutAssignment,
        RolloutService,
        RolloutState,
    )
    from rosclaw.continual.services.weight_update import (
        WeightUpdateService,
        WeightUpdateServiceReceipt,
    )

_LAZY_EXPORTS = {
    "ExperienceService": "rosclaw.continual.services.experience",
    "InferenceService": "rosclaw.continual.services.inference",
    "InferenceSlotReceipt": "rosclaw.continual.services.inference",
    "MotionVersionLease": "rosclaw.continual.services.inference",
    "LearnerProduct": "rosclaw.continual.services.learner",
    "LearnerService": "rosclaw.continual.services.learner",
    "LearnerServiceReceipt": "rosclaw.continual.services.learner",
    "ResidualSACServiceExecutor": "rosclaw.continual.services.learner",
    "RolloutAssignment": "rosclaw.continual.services.rollout",
    "RolloutService": "rosclaw.continual.services.rollout",
    "RolloutState": "rosclaw.continual.services.rollout",
    "WeightUpdateService": "rosclaw.continual.services.weight_update",
    "WeightUpdateServiceReceipt": "rosclaw.continual.services.weight_update",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ExperienceService",
    "InferenceService",
    "InferenceSlotReceipt",
    "LearnerProduct",
    "LearnerService",
    "LearnerServiceReceipt",
    "MotionVersionLease",
    "ResidualSACServiceExecutor",
    "RolloutAssignment",
    "RolloutService",
    "RolloutState",
    "WeightUpdateService",
    "WeightUpdateServiceReceipt",
]
