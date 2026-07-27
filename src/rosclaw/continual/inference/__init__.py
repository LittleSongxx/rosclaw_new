"""Read-only execution of versioned continual-learning candidates."""

from rosclaw.continual.inference.loader import (
    ResidualCandidateArtifact,
    load_residual_candidate,
)
from rosclaw.continual.inference.policy_runtime import (
    ResidualCandidateController,
    ResidualCandidatePolicy,
    build_g1_candidate_runtime,
)
from rosclaw.continual.inference.receipt import CandidateInferenceReceipt
from rosclaw.continual.inference.version_lock import CandidateVersionLock

__all__ = [
    "CandidateInferenceReceipt",
    "CandidateVersionLock",
    "ResidualCandidateArtifact",
    "ResidualCandidateController",
    "ResidualCandidatePolicy",
    "build_g1_candidate_runtime",
    "load_residual_candidate",
]
