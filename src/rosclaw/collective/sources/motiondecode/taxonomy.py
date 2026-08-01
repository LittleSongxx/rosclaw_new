"""Deterministic MotionDecode taxonomy parsing and pilot bucketing."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

MOTIONDECODE_INDEX_COLUMNS = (
    "Label",
    "Primary class name",
    "Secondary class name",
    "Tertiary class name",
    "Data Format",
    "Retargeted Model",
)
_MAX_CATALOG_BYTES = 10 * 1024 * 1024


class MotionFamily(StrEnum):
    FOOTBALL = "football"
    BALANCE = "balance"
    GAIT = "gait"
    TRANSITION_RECOVERY = "transition_recovery"
    OTHER = "other"


@dataclass(frozen=True)
class MotionDecodeTaxonomyRow:
    label: str
    primary: str
    secondary: str
    tertiary: str
    data_format: str
    retargeted_model: str
    family: MotionFamily

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "primary": self.primary,
            "secondary": self.secondary,
            "tertiary": self.tertiary,
            "data_format": self.data_format,
            "retargeted_model": self.retargeted_model,
            "family": self.family.value,
        }


@dataclass(frozen=True)
class MotionDecodeCatalogAudit:
    row_count: int
    duplicate_labels: tuple[str, ...]
    non_csv_rows: int
    non_g1_rows: int
    family_counts: dict[str, int]
    schema_version: str = "rosclaw.collective.motiondecode_catalog_audit.v1"

    @property
    def schema_valid(self) -> bool:
        return not self.duplicate_labels and self.non_csv_rows == 0 and self.non_g1_rows == 0

    @property
    def audit_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "duplicate_labels": list(self.duplicate_labels),
            "non_csv_rows": self.non_csv_rows,
            "non_g1_rows": self.non_g1_rows,
            "family_counts": dict(sorted(self.family_counts.items())),
            "schema_valid": self.schema_valid,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeCatalogAudit:
        counts = value["family_counts"]
        if not isinstance(counts, dict):
            raise ValueError("family_counts must be an object")
        return cls(
            row_count=int(value["row_count"]),
            duplicate_labels=tuple(str(item) for item in value["duplicate_labels"]),
            non_csv_rows=int(value["non_csv_rows"]),
            non_g1_rows=int(value["non_g1_rows"]),
            family_counts={str(key): int(count) for key, count in counts.items()},
        )


def parse_catalog(
    path: Path,
) -> tuple[tuple[MotionDecodeTaxonomyRow, ...], MotionDecodeCatalogAudit]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.stat().st_size > _MAX_CATALOG_BYTES:
        raise ValueError("MotionDecode metadata/index.csv exceeds the 10 MB safety limit")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MOTIONDECODE_INDEX_COLUMNS:
            raise ValueError("MotionDecode metadata/index.csv has an unexpected schema")
        rows = tuple(_taxonomy_row(row) for row in reader)
    if not rows:
        raise ValueError("MotionDecode metadata/index.csv is empty")
    labels = Counter(row.label for row in rows)
    duplicates = tuple(sorted(label for label, count in labels.items() if count > 1))
    family_counts = Counter(row.family.value for row in rows)
    audit = MotionDecodeCatalogAudit(
        row_count=len(rows),
        duplicate_labels=duplicates,
        non_csv_rows=sum(row.data_format.casefold() != "csv" for row in rows),
        non_g1_rows=sum(_normalized(row.retargeted_model) != "unitree_g1" for row in rows),
        family_counts=dict(family_counts),
    )
    return rows, audit


def classify_motion(*parts: str) -> MotionFamily:
    text = "_".join(_normalized(part) for part in parts)
    if re.search(r"(football|soccer)", text):
        return MotionFamily.FOOTBALL
    if re.search(r"(balance|single_leg|tiptoe|stability|push_recovery)", text):
        return MotionFamily.BALANCE
    if re.search(r"(gait|walking|jogging|running|sprint|turning|lateral)", text):
        return MotionFamily.GAIT
    if re.search(r"(transition|recovery|stand_to|to_stand|fall|lie_down|get_up)", text):
        return MotionFamily.TRANSITION_RECOVERY
    return MotionFamily.OTHER


def _taxonomy_row(row: dict[str, str]) -> MotionDecodeTaxonomyRow:
    values = {key: (row.get(key) or "").strip() for key in MOTIONDECODE_INDEX_COLUMNS}
    if not values["Label"]:
        raise ValueError("MotionDecode catalog contains an empty Label")
    family = classify_motion(
        values["Primary class name"],
        values["Secondary class name"],
        values["Tertiary class name"],
    )
    return MotionDecodeTaxonomyRow(
        label=values["Label"],
        primary=values["Primary class name"],
        secondary=values["Secondary class name"],
        tertiary=values["Tertiary class name"],
        data_format=values["Data Format"],
        retargeted_model=values["Retargeted Model"],
        family=family,
    )


def _normalized(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized


__all__ = [
    "MOTIONDECODE_INDEX_COLUMNS",
    "MotionDecodeCatalogAudit",
    "MotionDecodeTaxonomyRow",
    "MotionFamily",
    "classify_motion",
    "parse_catalog",
]
