"""TwinTouch — dual-body fingertip contact lab (v4).

Phase-1 surface (PR-TT-1): interaction contracts and contact topology
only.  No motion logic lives here; dispatch arrives with the Bimanual
ActionGateway (PR-TT-2), effect verification with the Action Effect
Gate (PR-TT-3), and contact control with the Contact Supervisor
(PR-TT-4).
"""

from rosclaw.twintouch.choreography import (
    CANONICAL_MARQUEE_PAIRS,
    PERMIT_ALREADY_USED,
    PERMIT_EXPIRED,
    PERMIT_HASH_MISMATCH,
    PERMIT_OK,
    ContactChoreographyContract,
    SequencePermit,
)
from rosclaw.twintouch.config import (
    EVOLUTION_BOUNDS,
    EVOLUTION_FORBIDDEN_KEYS,
    TwinTouchConfig,
    validate_candidate_changes,
)
from rosclaw.twintouch.envelope import (
    SCHEMA_VERSION as ENVELOPE_SCHEMA_VERSION,
)
from rosclaw.twintouch.envelope import (
    BimanualActionEnvelope,
    BodyActionBlock,
    CoordinationBlock,
    SafetyBlock,
)
from rosclaw.twintouch.gateway import (
    BimanualActionGateway,
    BodyExecutor,
    DispatchReport,
    EstopReport,
    LeaseRegistry,
    PreconditionProbe,
)
from rosclaw.twintouch.pairs import (
    FORBIDDEN_FINGERTIP_PAIRS,
    REACHABILITY_CALIBRATED,
    REACHABILITY_MUTUAL_CURL_ONLY,
    REACHABILITY_UNKNOWN,
    VALID_PAIR_IDS,
    VALID_PAIRS,
    FingerPair,
    FingerPairReachabilityMatrix,
    ForbiddenCollisionMap,
    TwinTouchPhysicalLayout,
    is_valid_pair_id,
    pair_by_id,
)
from rosclaw.twintouch.receipt import (
    CONTACT_ANOMALY_OUTCOMES,
    OUTCOME_ABORTED_BEFORE_DISPATCH,
    OUTCOME_CONTACT_CONFIRMED,
    OUTCOME_DISPATCHED,
    OUTCOME_EARLY_CONTACT,
    OUTCOME_NO_CONTACT,
    OUTCOME_ONE_SIDED_FORCE,
    OUTCOME_PARTIAL_DISPATCH,
    OUTCOME_PEER_NOT_READY,
    OUTCOME_RELEASE_FAILED,
    OUTCOME_STALE_OBSERVATION,
    OUTCOME_THERMAL_ABORT,
    OUTCOME_TRANSPORT_FAILURE,
    OUTCOME_UNINTENDED_CONTACT,
    OUTCOME_VISUAL_FORCE_CONFLICT,
    OUTCOME_WRONG_FINGER_CONTACT,
    InteractionReceipt,
)

__all__ = [
    "CANONICAL_MARQUEE_PAIRS",
    "CONTACT_ANOMALY_OUTCOMES",
    "ENVELOPE_SCHEMA_VERSION",
    "EVOLUTION_BOUNDS",
    "EVOLUTION_FORBIDDEN_KEYS",
    "FORBIDDEN_FINGERTIP_PAIRS",
    "PERMIT_ALREADY_USED",
    "PERMIT_EXPIRED",
    "PERMIT_HASH_MISMATCH",
    "PERMIT_OK",
    "REACHABILITY_CALIBRATED",
    "REACHABILITY_MUTUAL_CURL_ONLY",
    "REACHABILITY_UNKNOWN",
    "VALID_PAIR_IDS",
    "VALID_PAIRS",
    "BimanualActionEnvelope",
    "BimanualActionGateway",
    "BodyActionBlock",
    "BodyExecutor",
    "ContactChoreographyContract",
    "CoordinationBlock",
    "DispatchReport",
    "EstopReport",
    "FingerPair",
    "FingerPairReachabilityMatrix",
    "ForbiddenCollisionMap",
    "InteractionReceipt",
    "LeaseRegistry",
    "OUTCOME_ABORTED_BEFORE_DISPATCH",
    "OUTCOME_CONTACT_CONFIRMED",
    "OUTCOME_DISPATCHED",
    "OUTCOME_EARLY_CONTACT",
    "OUTCOME_NO_CONTACT",
    "OUTCOME_ONE_SIDED_FORCE",
    "OUTCOME_PARTIAL_DISPATCH",
    "OUTCOME_PEER_NOT_READY",
    "OUTCOME_RELEASE_FAILED",
    "OUTCOME_STALE_OBSERVATION",
    "OUTCOME_THERMAL_ABORT",
    "OUTCOME_TRANSPORT_FAILURE",
    "OUTCOME_UNINTENDED_CONTACT",
    "OUTCOME_VISUAL_FORCE_CONFLICT",
    "OUTCOME_WRONG_FINGER_CONTACT",
    "PreconditionProbe",
    "SafetyBlock",
    "SequencePermit",
    "TwinTouchConfig",
    "TwinTouchPhysicalLayout",
    "is_valid_pair_id",
    "pair_by_id",
    "validate_candidate_changes",
]
