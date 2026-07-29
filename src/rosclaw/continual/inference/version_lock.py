"""Episode-scoped no-switch lock for a continual candidate."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.continual.contracts import PolicyVersion


@dataclass(frozen=True)
class CandidateVersionLock:
    policy_version: int
    policy_version_hash: str
    artifact_hash: str
    body_hash: str

    @classmethod
    def pin(cls, policy: PolicyVersion) -> CandidateVersionLock:
        return cls(
            policy_version=policy.version,
            policy_version_hash=policy.version_hash,
            artifact_hash=policy.artifact_hash,
            body_hash=policy.body_hash,
        )

    def verify(self, policy: PolicyVersion) -> None:
        if (
            policy.version != self.policy_version
            or policy.version_hash != self.policy_version_hash
            or policy.artifact_hash != self.artifact_hash
            or policy.body_hash != self.body_hash
        ):
            raise RuntimeError("candidate policy version changed inside a locked motion segment")


__all__ = ["CandidateVersionLock"]
