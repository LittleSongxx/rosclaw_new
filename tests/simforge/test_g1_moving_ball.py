from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios


def _scenario():
    base = generate_goalforge_scenarios(
        ledger=SeedLedger(
            task_id="g1_penalty_kick",
            secret=b"moving-ball-adapter-unit-test",
        ),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]
    return replace(
        base,
        ball_x_m=1.12,
        ball_velocity_x_mps=-0.08,
        ball_launch_delay_sec=4.0,
    )


def test_moving_ball_adapter_plans_inside_bounded_intercept_envelope() -> None:
    plan = MovingBallInterceptAdapter().plan(_scenario())

    assert plan.eligible
    assert plan.predicted_ball_x_m == pytest.approx(1.0176)
    assert plan.nominal_contact_error_m < 0.02
    assert plan.parameters.policy_type == "parameter"


def test_moving_ball_adapter_rejects_unvalidated_fast_pass() -> None:
    scenario = replace(_scenario(), ball_velocity_x_mps=-0.45)
    plan = MovingBallInterceptAdapter().plan(scenario)

    assert not plan.eligible
    assert "ball_speed_outside_validated_envelope" in plan.reasons
