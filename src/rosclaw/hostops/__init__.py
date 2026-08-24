"""HostOps plane (Skill Runtime 2.0, doc §17-§23).

Host skills never route through the sandboxed harness shell, and never
gain an unrestricted root shell. The broker accepts **typed operations**
only (``package.install``, ``repository.enable``, …) and converts them
into safe argv itself; everything else fails closed.
"""

from rosclaw.hostops.planner import plan_hash  # noqa: F401
from rosclaw.hostops.policy import (  # noqa: F401
    ApprovalMismatchError,
    HostOpsPolicy,
    HostOpsPolicyError,
)
