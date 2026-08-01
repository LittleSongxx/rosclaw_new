"""Fail-closed MotionDecode source qualification.

The adapter never downloads a dataset and never authorizes hardware.  It turns
an operator-managed local snapshot into content-addressed source evidence, then
audits motion CSVs without treating a kinematic reference as an RL transition.
"""

from rosclaw.collective.sources.motiondecode.audit import (
    MotionDecodeClipAudit,
    MotionDecodeIngestReport,
    MotionQualificationLevel,
    audit_motiondecode_snapshot,
)
from rosclaw.collective.sources.motiondecode.license import (
    MotionDecodeLicenseSnapshot,
    snapshot_license,
)
from rosclaw.collective.sources.motiondecode.manifest import (
    MotionDecodeFileRecord,
    MotionDecodeRegistration,
    MotionDecodeSourceManifest,
    register_motiondecode_source,
)
from rosclaw.collective.sources.motiondecode.parser import CanonicalMotionEpisode
from rosclaw.collective.sources.motiondecode.repair import (
    MotionDecodeRepairReport,
    MotionRepairDisposition,
    SegmentationRepairManifest,
    repair_motiondecode_snapshot,
    replay_segmentation_repair,
)
from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily

__all__ = [
    "CanonicalMotionEpisode",
    "MotionDecodeClipAudit",
    "MotionDecodeFileRecord",
    "MotionDecodeIngestReport",
    "MotionDecodeLicenseSnapshot",
    "MotionDecodeRegistration",
    "MotionDecodeRepairReport",
    "MotionDecodeSourceManifest",
    "MotionFamily",
    "MotionQualificationLevel",
    "MotionRepairDisposition",
    "SegmentationRepairManifest",
    "audit_motiondecode_snapshot",
    "register_motiondecode_source",
    "repair_motiondecode_snapshot",
    "replay_segmentation_repair",
    "snapshot_license",
]
