"""Attribution manifest for derived MotionDecode artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class MotionDecodeAttribution:
    provider: str
    dataset: str
    source_uri: str
    revision: str
    attribution_text: str
    license_snapshot_hash: str
    schema_version: str = "rosclaw.collective.motiondecode_attribution.v1"

    def __post_init__(self) -> None:
        for label in (
            "provider",
            "dataset",
            "source_uri",
            "revision",
            "attribution_text",
            "license_snapshot_hash",
        ):
            if not getattr(self, label).strip():
                raise ValueError(f"{label} must not be empty")

    @property
    def attribution_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "dataset": self.dataset,
            "source_uri": self.source_uri,
            "revision": self.revision,
            "attribution_text": self.attribution_text,
            "license_snapshot_hash": self.license_snapshot_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeAttribution:
        return cls(
            provider=str(value["provider"]),
            dataset=str(value["dataset"]),
            source_uri=str(value["source_uri"]),
            revision=str(value["revision"]),
            attribution_text=str(value["attribution_text"]),
            license_snapshot_hash=str(value["license_snapshot_hash"]),
        )


__all__ = ["MotionDecodeAttribution"]
