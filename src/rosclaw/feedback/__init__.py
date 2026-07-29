"""ROSClaw user-feedback telemetry and synchronous control-feedback planes."""

from __future__ import annotations

from rosclaw.feedback.config import FeedbackConfig, TelemetryConfig
from rosclaw.feedback.contracts import (
    FeedbackFrame,
    FeedbackLoopSpec,
    FeedbackReceipt,
    ResidualCommand,
)
from rosclaw.feedback.directories import ensure_feedback_dirs
from rosclaw.feedback.installation import Installation, InstallationManager
from rosclaw.feedback.store import append_event, count_events, directory_size_mb, read_events

__all__ = [
    "FeedbackConfig",
    "FeedbackFrame",
    "FeedbackLoopSpec",
    "FeedbackReceipt",
    "TelemetryConfig",
    "ensure_feedback_dirs",
    "Installation",
    "InstallationManager",
    "append_event",
    "count_events",
    "directory_size_mb",
    "read_events",
    "ResidualCommand",
]
