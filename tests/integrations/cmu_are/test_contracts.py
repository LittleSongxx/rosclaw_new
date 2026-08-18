from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rosclaw.integrations.cmu_are.contracts import (
    CmuAreContractError,
    body_snapshot_hash,
    load_safety_contract,
    resolve_target,
)


def test_malformed_safety_card_fails_closed(tmp_path: Path) -> None:
    card = tmp_path / "bad.yaml"
    card.write_text("schema_version: rosclaw.embodiment_card.v1\n", encoding="utf-8")
    with pytest.raises(CmuAreContractError):
        load_safety_contract(card)


def test_place_resolution_and_geofence() -> None:
    safety = load_safety_contract()
    target = resolve_target(
        place="inspection_a",
        x=None,
        y=None,
        places={
            "inspection_a": type(
                "Place",
                (),
                {
                    "name": "inspection_a",
                    "x": 0.0,
                    "y": 6.0,
                    "z": 0.0,
                    "frame_id": "map",
                    "aliases": (),
                },
            )()
        },
        safety=safety,
    )
    assert target == {"frame_id": "map", "x": 0.0, "y": 6.0, "z": 0.0}
    with pytest.raises(CmuAreContractError):
        resolve_target(
            place=None,
            x=101.0,
            y=0.0,
            places={},
            safety=safety,
        )


def test_snapshot_hash_is_stable() -> None:
    assert body_snapshot_hash() == body_snapshot_hash()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_sequence_steps", 8.0),
        ("max_sequence_steps", "8"),
        ("max_speed", float("nan")),
        ("max_speed", True),
    ],
)
def test_safety_card_rejects_non_contract_numeric_values(
    tmp_path: Path, field: str, value: object
) -> None:
    source = Path(__file__).parents[3] / "src/rosclaw/connectors/ros/specs/cmu_are.yaml"
    card = yaml.safe_load(source.read_text(encoding="utf-8"))
    if field == "max_sequence_steps":
        card["operational_limits"][field] = value
    else:
        card["operational_limits"][field] = value
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(card, allow_unicode=True), encoding="utf-8")
    with pytest.raises(CmuAreContractError):
        load_safety_contract(target)


def test_resolve_target_rejects_bool_and_string_coordinates() -> None:
    safety = load_safety_contract()
    for x, y in [(True, 0.0), ("1.0", 0.0)]:
        with pytest.raises(CmuAreContractError):
            resolve_target(
                place=None,
                x=x,
                y=y,
                places={},
                safety=safety,
            )
