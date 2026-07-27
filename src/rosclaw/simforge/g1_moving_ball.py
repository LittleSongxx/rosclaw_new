"""Bounded high-level intercept adapter around the frozen RoboNaldo prior."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario


@dataclass(frozen=True)
class MovingBallPlan:
    launch_time_sec: float
    predicted_contact_time_sec: float
    predicted_ball_x_m: float
    predicted_ball_y_m: float
    nominal_contact_error_m: float
    eligible: bool
    reasons: tuple[str, ...]
    parameters: ShotParameters
    schema_version: str = "rosclaw.g1_goalforge.moving_ball_plan.v1"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["parameters"] = self.parameters.to_dict()
        return value


class MovingBallInterceptAdapter:
    """Plan only gentle, reachable passes; never writes low-level joint targets."""

    nominal_contact_time_sec = 5.28
    nominal_ball_x_m = 1.0
    maximum_contact_position_error_m = 0.16
    maximum_ball_speed_mps = 0.20

    def plan(self, scenario: GoalForgeScenario) -> MovingBallPlan:
        moving_duration = max(0.0, self.nominal_contact_time_sec - scenario.ball_launch_delay_sec)
        predicted_x = scenario.ball_x_m + scenario.ball_velocity_x_mps * moving_duration
        predicted_y = scenario.ball_y_m + scenario.ball_velocity_y_mps * moving_duration
        error = math.hypot(predicted_x - self.nominal_ball_x_m, predicted_y)
        speed = math.hypot(scenario.ball_velocity_x_mps, scenario.ball_velocity_y_mps)
        reasons = []
        if scenario.ball_launch_delay_sec <= 0.0:
            reasons.append("launcher_delay_missing")
        if speed <= 0.0:
            reasons.append("ball_not_moving")
        if speed > self.maximum_ball_speed_mps:
            reasons.append("ball_speed_outside_validated_envelope")
        if error > self.maximum_contact_position_error_m:
            reasons.append("predicted_intercept_outside_kick_envelope")
        eligible = not reasons
        parameters = ShotParameters(
            pelvis_yaw_offset=0.1925,
            com_shift_y=0.015,
            foot_yaw_offset=0.03025,
            recovery_step_length=0.055,
            policy_type="parameter",
        )
        return MovingBallPlan(
            launch_time_sec=scenario.ball_launch_delay_sec,
            predicted_contact_time_sec=self.nominal_contact_time_sec,
            predicted_ball_x_m=predicted_x,
            predicted_ball_y_m=predicted_y,
            nominal_contact_error_m=error,
            eligible=eligible,
            reasons=tuple(reasons),
            parameters=parameters,
        )


__all__ = ["MovingBallInterceptAdapter", "MovingBallPlan"]
