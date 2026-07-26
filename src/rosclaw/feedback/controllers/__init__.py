"""Controllers that produce bounded residuals for the Feedback Plane."""

from rosclaw.feedback.controllers.balance import G1BalanceReflexConfig, G1BalanceReflexController
from rosclaw.feedback.controllers.base import FeedbackController, ZeroResidualController
from rosclaw.feedback.controllers.kick_skill import (
    G1KickSkillFeedbackConfig,
    G1KickSkillFeedbackController,
)
from rosclaw.feedback.controllers.phase import LatchedPhaseGate
from rosclaw.feedback.controllers.pid import PIDGains, PIDResidualController

__all__ = [
    "FeedbackController",
    "G1BalanceReflexConfig",
    "G1BalanceReflexController",
    "G1KickSkillFeedbackConfig",
    "G1KickSkillFeedbackController",
    "LatchedPhaseGate",
    "PIDGains",
    "PIDResidualController",
    "ZeroResidualController",
]
