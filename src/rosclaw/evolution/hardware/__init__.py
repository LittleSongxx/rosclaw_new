"""Hardware self-evolution acceptance (Evo-RPS, 真机自进化v2).

Public surface:

* :class:`EvoRpsConfig` / :func:`load_config` — experiment contract (§12)
* :class:`ExperimentNamespace` — isolated SeekDB/practice/trace/evidence roots (§2.7)
* :class:`EvidenceManifest` — hash-bound run evidence (§Phase 0.6)
* :func:`run_preflight` — hardware gates, no mock in formal mode (§2.2)
"""

from .contracts import (
    BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE,
    CandidateSpace,
    EvoRpsConfig,
    ValidationError,
    load_config,
)
from .evidence import EvidenceManifest
from .namespace import ExperimentNamespace
from .preflight import PreflightReport, run_preflight

__all__ = [
    "BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE",
    "CandidateSpace",
    "EvidenceManifest",
    "EvoRpsConfig",
    "ExperimentNamespace",
    "PreflightReport",
    "ValidationError",
    "load_config",
    "run_preflight",
]
