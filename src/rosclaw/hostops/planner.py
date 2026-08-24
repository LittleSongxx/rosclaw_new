"""Plan hashing (doc §21).

Approval binds to the plan hash: the user approves *this exact plan*, not
"the agent may sudo". Any change to skill / target / operations changes
the hash and requires re-approval of the changed plan.
"""

from __future__ import annotations

import hashlib
import json

# Only these fields define what will actually happen on the host.
_HASH_FIELDS = ("skill", "domain", "target", "operations")


def canonical_plan(plan: dict) -> dict:
    return {k: plan.get(k) for k in _HASH_FIELDS}


def plan_hash(plan: dict) -> str:
    """SHA256 over the canonical plan (skill + target + operations)."""
    blob = json.dumps(canonical_plan(plan), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
