"""Qualified Feedback Plane profiles for supported simulated bodies."""

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.profiles.g1_cerebellum import build_g1_cerebellum_runtime
from rosclaw.feedback.profiles.g1_skill import build_g1_kick_skill_runtime

__all__ = [
    "build_g1_balance_runtime",
    "build_g1_cerebellum_runtime",
    "build_g1_kick_skill_runtime",
]
