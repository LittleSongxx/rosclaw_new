"""Strict deterministic replay for Feedback Plane traces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rosclaw.feedback.contracts import FeedbackInput
from rosclaw.feedback.runtime import FeedbackRuntime


@dataclass(frozen=True)
class FeedbackReplayReport:
    samples: int
    matched: int
    first_mismatch: int | None
    strict_replay: bool


class RecordedLatencyClock:
    """Replay recorded compute durations without depending on host jitter."""

    def __init__(self, latency_ns: Sequence[int]) -> None:
        self._latencies = tuple(int(value) for value in latency_ns)
        if any(value < 0 for value in self._latencies):
            raise ValueError("recorded latency must be non-negative")
        self._index = 0
        self._start = True
        self._cursor = 0

    def __call__(self) -> int:
        if self._index >= len(self._latencies):
            raise RuntimeError("recorded latency clock exhausted")
        if self._start:
            self._start = False
            return self._cursor
        self._cursor += self._latencies[self._index]
        self._index += 1
        self._start = True
        return self._cursor


def verify_feedback_replay(
    runtime_factory: Callable[[], FeedbackRuntime],
    inputs: Sequence[FeedbackInput],
    expected_command_hashes: Sequence[str],
) -> FeedbackReplayReport:
    if len(inputs) != len(expected_command_hashes):
        raise ValueError("replay inputs and expected hashes must have equal length")
    runtime = runtime_factory()
    matched = 0
    first_mismatch: int | None = None
    for index, (sample, expected) in enumerate(zip(inputs, expected_command_hashes, strict=True)):
        command = runtime.tick(
            timestamp_ns=sample.timestamp_ns,
            observation_timestamp_ns=sample.observation_timestamp_ns,
            phase=sample.phase,
            reference=sample.reference,
            actual=sample.actual,
            base_action=sample.base_action,
        )
        if command.command_hash == expected:
            matched += 1
        elif first_mismatch is None:
            first_mismatch = index
    return FeedbackReplayReport(
        samples=len(inputs),
        matched=matched,
        first_mismatch=first_mismatch,
        strict_replay=matched == len(inputs),
    )
