"""Controller protocol for the synchronous Feedback Plane."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from rosclaw.feedback.contracts import FeedbackFrame, canonical_hash


class FeedbackController(Protocol):
    """A controller must be deterministic and free of asynchronous I/O."""

    @property
    def controller_hash(self) -> str: ...

    def reset(self) -> None: ...

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]: ...

    def config_dict(self) -> dict[str, object]: ...


class ZeroResidualController:
    """Control-plane-off baseline that still exercises timing and evidence."""

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
        del frame, base_action
        return {}

    def config_dict(self) -> dict[str, object]:
        return {"controller_type": "zero_residual", "version": 1}
