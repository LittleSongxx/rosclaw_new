"""Deadline-aware synchronous runtime for ROSClaw's Feedback Plane."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

import numpy as np

from rosclaw.data.ring_buffer import RingBuffer
from rosclaw.feedback.contracts import (
    ControllerSnapshot,
    FallbackMode,
    FeedbackFrame,
    FeedbackInput,
    FeedbackLoopSpec,
    FeedbackReceipt,
    FeedbackTickRecord,
    ResidualCommand,
    canonical_hash,
)
from rosclaw.feedback.controllers.base import FeedbackController
from rosclaw.feedback.safety_projector import FeedbackSafetyProjector
from rosclaw.feedback.state_estimator import FeedbackStateEstimator


def _finite_or_zero(values: Mapping[str, float]) -> dict[str, float]:
    """Return a finite copy of a signal mapping, zeroing invalid entries."""

    clean: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        clean[str(key)] = number if math.isfinite(number) else 0.0
    return clean


class FeedbackRuntime:
    """Execute one deterministic controller synchronously at L1/L2 rates.

    No EventBus, network call, model service or evidence write occurs in
    :meth:`tick`.  Ring buffers are preallocated; receipts are assembled only
    after the loop.  This runtime augments rather than replaces the L0 servo.
    """

    def __init__(
        self,
        *,
        spec: FeedbackLoopSpec,
        controller: FeedbackController,
        capacity: int = 4096,
        safety_projector: FeedbackSafetyProjector | None = None,
        compute_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if controller.controller_hash != spec.controller_hash:
            raise ValueError("controller hash does not match pinned FeedbackLoopSpec")
        self.spec = spec
        self.controller = controller
        self.projector = safety_projector or FeedbackSafetyProjector(spec)
        self._compute_clock_ns = compute_clock_ns
        self._estimator = FeedbackStateEstimator(spec.reference_signals)
        self._error_buffer = RingBuffer(capacity, (len(spec.reference_signals),))
        self._residual_buffer = RingBuffer(capacity, (len(spec.output_limits),))
        self._output_order = tuple(sorted(spec.output_limits))
        self._capacity = capacity
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self._estimator.reset()
        self._error_buffer.clear()
        self._residual_buffer.clear()
        self._records: list[FeedbackTickRecord] = []
        self._sequence = 0
        self._last_safe_projected: dict[str, float] = {}

    @property
    def records(self) -> tuple[FeedbackTickRecord, ...]:
        return tuple(self._records)

    def tick(
        self,
        *,
        timestamp_ns: int,
        observation_timestamp_ns: int,
        phase: float,
        reference: Mapping[str, float],
        actual: Mapping[str, float],
        base_action: Mapping[str, float],
    ) -> ResidualCommand:
        """Compute and project one residual or remove it on any invalid input."""

        sequence = self._sequence
        self._sequence += 1
        started_ns = self._compute_clock_ns()
        invalid_reason: str | None = None
        try:
            input_sample = FeedbackInput(
                timestamp_ns=timestamp_ns,
                observation_timestamp_ns=observation_timestamp_ns,
                phase=phase,
                reference=reference,
                actual=actual,
                base_action=base_action,
            )
        except ValueError as exc:
            # Fail closed on non-finite policy/sensor input instead of raising
            # out of the synchronous control loop.
            invalid_reason = f"invalid_input:{exc}"
            reference = _finite_or_zero(reference)
            actual = _finite_or_zero(actual)
            base_action = _finite_or_zero(base_action)
            phase = phase if math.isfinite(phase) else 0.0
            input_sample = FeedbackInput(
                timestamp_ns=timestamp_ns,
                observation_timestamp_ns=observation_timestamp_ns,
                phase=phase,
                reference=reference,
                actual=actual,
                base_action=base_action,
            )
        age_ns = timestamp_ns - observation_timestamp_ns
        stale = age_ns < 0 or age_ns > int(self.spec.max_observation_age_ms * 1_000_000.0)
        error = None
        if invalid_reason is not None:
            command = self.projector.fallback(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                base_action=base_action,
                reason=invalid_reason,
                mode=self.spec.fallback_unsafe_projection,
                deadline_met=True,
            )
        elif stale:
            command = self.projector.fallback(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                base_action=base_action,
                reason=f"stale_observation:{age_ns}",
                mode=self.spec.fallback_stale_observation,
                deadline_met=True,
            )
        else:
            try:
                missing = set(self.spec.observation_signals).difference(actual)
                if missing:
                    raise ValueError(f"missing observation signals: {sorted(missing)}")
                error = self._estimator.update(reference, actual, timestamp_ns)
                frame = FeedbackFrame(
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    observation_timestamp_ns=observation_timestamp_ns,
                    phase=phase,
                    reference=reference,
                    actual=actual,
                    error=error,
                )
                raw = self.controller.compute(frame, base_action)
                command = self.projector.project(
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    base_action=base_action,
                    raw=raw,
                )
            except Exception as exc:  # noqa: BLE001 - controller faults must fail closed
                command = self.projector.fallback(
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    base_action=base_action,
                    raw={},
                    reason=f"controller_invalid:{type(exc).__name__}",
                    mode=self.spec.fallback_unsafe_projection,
                    deadline_met=True,
                )
        elapsed_ns = max(0, self._compute_clock_ns() - started_ns)
        deadline_ns = int(self.spec.deadline_ms * 1_000_000.0)
        if elapsed_ns > deadline_ns:
            freeze = (
                self._last_safe_projected
                if self.spec.fallback_deadline_miss is FallbackMode.FREEZE_AND_STABILIZE
                else None
            )
            command = self.projector.fallback(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                base_action=base_action,
                raw=command.raw,
                reason=f"deadline_miss:{elapsed_ns}",
                mode=self.spec.fallback_deadline_miss,
                deadline_met=False,
                projected_override=freeze,
            )
        elif command.fallback is None:
            self._last_safe_projected = dict(command.projected)
        error_values = (
            np.asarray([error.value[signal] for signal in self.spec.reference_signals])
            if error is not None
            else np.full(len(self.spec.reference_signals), np.nan, dtype=np.float64)
        )
        residual_values = np.asarray(
            [command.projected.get(signal, 0.0) for signal in self._output_order],
            dtype=np.float64,
        )
        timestamp_sec = timestamp_ns / 1_000_000_000.0
        self._error_buffer.append(error_values, timestamp=timestamp_sec)
        self._residual_buffer.append(residual_values, timestamp=timestamp_sec)
        self._records.append(
            FeedbackTickRecord(
                input=input_sample,
                error_rms=error.rms if error is not None else math.nan,
                command_hash=command.command_hash,
                latency_ns=elapsed_ns,
                deadline_met=command.deadline_met,
                stale_observation=stale,
                saturation_count=command.saturation_count,
                residual_norm=float(np.linalg.norm(residual_values)),
                correction_applied=bool(np.any(np.abs(residual_values) > 0.0)),
                safety_projection_applied=bool(
                    command.saturation_count or command.fallback is not None
                ),
                reasons=command.reasons,
            )
        )
        if len(self._records) > self._capacity:
            del self._records[0]
        return command

    def build_receipt(
        self,
        *,
        action_id: str,
        strict_replay: bool,
        evidence_domain: str = "SHADOW",
    ) -> FeedbackReceipt:
        """Summarize the bounded trace after execution, outside the hot path."""

        if not self._records:
            raise ValueError("cannot build a feedback receipt without ticks")
        finite_indices = [
            index for index, record in enumerate(self._records) if math.isfinite(record.error_rms)
        ]
        valid_errors = np.asarray(
            [self._records[index].error_rms for index in finite_indices],
            dtype=np.float64,
        )
        correction_indices = [
            index for index, record in enumerate(self._records) if record.correction_applied
        ]
        if correction_indices:
            window = max(1, min(50, len(correction_indices) // 5))
            first = correction_indices[0]
            last = correction_indices[-1]
            before_values = [
                record.error_rms
                for record in self._records[max(0, first - window) : first]
                if math.isfinite(record.error_rms)
            ]
            after_values = [
                record.error_rms
                for record in self._records[max(first, last - window + 1) : last + 1]
                if math.isfinite(record.error_rms)
            ]
            initial = float(np.mean(before_values or [self._records[first].error_rms]))
            final = float(np.mean(after_values or [self._records[last].error_rms]))
        else:
            window = max(1, int(math.ceil(len(valid_errors) * 0.2)))
            initial = float(np.mean(valid_errors[:window])) if len(valid_errors) else None
            final = float(np.mean(valid_errors[-window:])) if len(valid_errors) else None
        latency_ms = np.asarray(
            [record.latency_ns / 1_000_000.0 for record in self._records],
            dtype=np.float64,
        )
        timestamps_ns = np.asarray(
            [record.input.timestamp_ns for record in self._records],
            dtype=np.int64,
        )
        intervals_ns = np.diff(timestamps_ns)
        jitter_ms = np.abs(intervals_ns - self.spec.period_ns) / 1_000_000.0
        dropped_frames = int(
            sum(max(0, int(round(interval / self.spec.period_ns)) - 1) for interval in intervals_ns)
        )
        reference_hash = canonical_hash(
            {
                "reference": [
                    dict(sorted(record.input.reference.items())) for record in self._records
                ]
            }
        )
        observation_hash = canonical_hash(
            {"actual": [dict(sorted(record.input.actual.items())) for record in self._records]}
        )
        snapshot = ControllerSnapshot(
            controller_id=self.spec.loop_id + ":controller",
            controller_type=type(self.controller).__name__,
            body_hash=self.spec.body_hash,
            loop_spec_hash=self.spec.spec_hash,
            config=self.controller.config_dict(),
        )
        return FeedbackReceipt(
            loop_id=self.spec.loop_id,
            loop_spec_hash=self.spec.spec_hash,
            controller_hash=self.spec.controller_hash,
            controller_snapshot_hash=snapshot.snapshot_hash,
            body_hash=self.spec.body_hash,
            action_id=action_id,
            reference_hash=reference_hash,
            observation_hash=observation_hash,
            trace_hash=self.trace_hash,
            samples=len(self._records),
            initial_error_rms=initial,
            final_error_rms=final,
            deadline_miss_count=sum(not record.deadline_met for record in self._records),
            stale_observation_count=sum(record.stale_observation for record in self._records),
            saturation_count=sum(record.saturation_count for record in self._records),
            latency_p50_ms=float(np.percentile(latency_ms, 50)) if len(latency_ms) else math.nan,
            latency_p99_ms=float(np.percentile(latency_ms, 99)) if len(latency_ms) else math.nan,
            jitter_p99_ms=float(np.percentile(jitter_ms, 99)) if len(jitter_ms) else 0.0,
            observation_age_max_ms=(
                max(
                    (record.input.timestamp_ns - record.input.observation_timestamp_ns)
                    / 1_000_000.0
                    for record in self._records
                )
                if self._records
                else 0.0
            ),
            dropped_frame_count=dropped_frames,
            correction_applied=bool(correction_indices),
            safety_projection_applied=any(
                record.safety_projection_applied for record in self._records
            ),
            strict_replay=strict_replay,
            evidence_domain=evidence_domain,
        )

    @property
    def trace_hash(self) -> str:
        return canonical_hash(
            {
                "loop_spec_hash": self.spec.spec_hash,
                "ticks": [
                    {
                        "timestamp_ns": record.input.timestamp_ns,
                        "observation_timestamp_ns": record.input.observation_timestamp_ns,
                        "phase": record.input.phase,
                        "reference": dict(sorted(record.input.reference.items())),
                        "actual": dict(sorted(record.input.actual.items())),
                        "base_action": dict(sorted(record.input.base_action.items())),
                        "error_rms": (
                            record.error_rms if math.isfinite(record.error_rms) else None
                        ),
                        "command_hash": record.command_hash,
                    }
                    for record in self._records
                ],
            }
        )

    def recent_error(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        return self._error_buffer.get_last_n(count)

    def recent_residual(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        return self._residual_buffer.get_last_n(count)
