"""Production source adapters for the ContextCompiler (PR-NA-020 wiring).

SIM sources are honest: the sim body summary says it is simulated, the sim
self source increments its sequence per read. If a real body is linked via
``rosclaw.body.BodyResolver``, ``ResolverBodySource`` exposes it (fail
closed on any resolver error).
"""

from __future__ import annotations

from datetime import UTC, datetime

from rosclaw.agentd.context.sources import (
    BodyFacts,
    CapabilityInfo,
    ConsentFacts,
    MemoryItem,
    OrgFacts,
    SelfFacts,
)
from rosclaw.contracts.common import content_hash


class SimBodySource:
    """Deterministic simulated body, explicitly marked as simulation."""

    def __init__(self, body_id: str, summary: str | None = None) -> None:
        self._body_id = body_id
        self._summary = summary or (
            f"SIMULATED body {body_id} (evidence_class=simulated; "
            "not usable as REAL physical proof)"
        )
        self._hash = content_hash("body", {"body_id": body_id, "kind": "sim", "v": 1})

    def get_body(self, body_id: str) -> BodyFacts | None:
        if body_id != self._body_id:
            return None
        return BodyFacts(
            body_id=self._body_id,
            effective_body_hash=self._hash,
            summary=self._summary,
            calibrated=True,
        )

    @property
    def body_hash(self) -> str:
        return self._hash


class ResolverBodySource:
    """Real body via rosclaw.body.BodyResolver. Fail closed on errors."""

    def __init__(self) -> None:
        self._resolver = None
        try:
            from rosclaw.body.resolver import BodyResolver

            self._resolver = BodyResolver()
        except Exception:  # noqa: BLE001 - absence means "no real body"
            self._resolver = None

    def get_body(self, body_id: str) -> BodyFacts | None:
        if self._resolver is None:
            return None
        try:
            effective = self._resolver.get_effective_body()
            body_hash = effective.compute_hash()
            summary = f"EffectiveBody {body_id} (hash {body_hash[:18]}…)"
            return BodyFacts(
                body_id=body_id,
                effective_body_hash=body_hash,
                summary=summary,
                calibrated=True,
            )
        except Exception:  # noqa: BLE001 - resolver failure must fail closed
            return None


class SimSelfSource:
    def __init__(self) -> None:
        self._sequence = 0

    def get_self(self, body_id: str) -> SelfFacts | None:
        self._sequence += 1
        return SelfFacts(
            self_snapshot_hash=content_hash(
                "selfsnap", {"body_id": body_id, "seq": self._sequence}
            ),
            sequence=self._sequence,
            observed_at=datetime.now(UTC),
            health="OK",
            summary=f"SIMULATED self state seq={self._sequence} health=OK",
        )


class StaticCapabilitySource:
    def __init__(self, names: list[str]) -> None:
        self._names = sorted(names)

    def list_capabilities(self, query: str, limit: int) -> list[CapabilityInfo]:
        return [
            CapabilityInfo(name=n, kind="tool", summary=f"builtin tool {n}")
            for n in self._names[: limit * 2]
        ]


class EmptyMemorySource:
    def retrieve(self, query: str, limit: int) -> list[MemoryItem]:
        return []


class NullOrgSource:
    def get_org(self) -> OrgFacts:
        return OrgFacts()


class ConfigConsentSource:
    """Public consent facts from config. Grants arrive with Operator Broker."""

    def __init__(self, allowed_risk_tiers: tuple[str, ...] = ("LOW",)) -> None:
        self._tiers = allowed_risk_tiers
        self._policy_hash = content_hash(
            "pol", {"policy": "default_sim_only", "tiers": list(allowed_risk_tiers)}
        )

    @property
    def policy_hash(self) -> str:
        return self._policy_hash

    def get_consent(self, mission_id: str) -> ConsentFacts | None:
        return ConsentFacts(
            policy_hash=self._policy_hash,
            public_scope_summary=(
                "default policy: SIMULATION only, EXACT_ACTION authorization, "
                f"allowed_risk_tiers={list(self._tiers)}"
            ),
            allowed_risk_tiers=self._tiers,
        )
