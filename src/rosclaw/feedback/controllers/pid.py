"""General vector PID residual controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rosclaw.feedback.contracts import FeedbackFrame, canonical_hash


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0


class PIDResidualController:
    """Map each tracked signal to one output using explicit bounded gains."""

    def __init__(
        self,
        gains: Mapping[str, PIDGains],
        output_map: Mapping[str, str] | None = None,
    ) -> None:
        if not gains:
            raise ValueError("gains must not be empty")
        self.gains = dict(gains)
        self.output_map = dict(output_map or {signal: signal for signal in gains})
        if set(self.gains).difference(self.output_map):
            raise ValueError("output_map must cover every configured PID signal")

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        return None

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]:
        del base_action
        residual: dict[str, float] = {}
        for signal, gains in self.gains.items():
            residual[self.output_map[signal]] = (
                gains.kp * frame.error.value[signal]
                + gains.ki * frame.error.integral[signal]
                + gains.kd * frame.error.derivative[signal]
            )
        return residual

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "pid_residual",
            "version": 1,
            "gains": {signal: asdict(gain) for signal, gain in sorted(self.gains.items())},
            "output_map": dict(sorted(self.output_map.items())),
        }
