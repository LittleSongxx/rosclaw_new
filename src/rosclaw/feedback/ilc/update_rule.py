"""Body-bound, bounded iterative learning control update rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosclaw.feedback.contracts import AdaptationSnapshot
from rosclaw.feedback.ilc.trajectory_memory import ILCFeedforward


@dataclass(frozen=True)
class ILCUpdate:
    trial: int
    previous_error_rms: float
    current_error_rms: float
    residual_peak: float
    converged: bool
    snapshot: AdaptationSnapshot


class BoundedILC:
    """Learn a feed-forward residual only between completed trials.

    ``u[k+1] = clip(retention*u[k] + learning_gain*e[k])``.  Shape and body
    identity are pinned on first use; no online safety boundary is learnable.
    """

    def __init__(
        self,
        *,
        body_hash: str,
        learning_gain: float = 0.2,
        retention: float = 0.95,
        residual_limit: float = 0.12,
        convergence_tolerance: float = 1e-3,
    ) -> None:
        if not 0.0 < learning_gain <= 1.0:
            raise ValueError("learning_gain must be in (0, 1]")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        if residual_limit <= 0.0 or convergence_tolerance < 0.0:
            raise ValueError("residual_limit must be positive and tolerance non-negative")
        self.body_hash = body_hash
        self.learning_gain = float(learning_gain)
        self.retention = float(retention)
        self.residual_limit = float(residual_limit)
        self.convergence_tolerance = float(convergence_tolerance)
        self.reset()

    def reset(self) -> None:
        self._residual: np.ndarray | None = None
        self._previous_error_rms: float | None = None
        self._trial = 0

    @property
    def residual(self) -> np.ndarray | None:
        return None if self._residual is None else self._residual.copy()

    def update(
        self,
        tracking_error: np.ndarray,
        *,
        source_receipt_hash: str,
    ) -> ILCUpdate:
        error = np.asarray(tracking_error, dtype=np.float64)
        if error.ndim != 2 or error.size == 0 or not np.all(np.isfinite(error)):
            raise ValueError("tracking_error must be a finite non-empty [time, signal] array")
        if self._residual is None:
            self._residual = np.zeros_like(error)
        if self._residual.shape != error.shape:
            raise ValueError("tracking_error shape is pinned by the first ILC trial")
        previous = self._previous_error_rms
        current = float(np.sqrt(np.mean(np.square(error))))
        candidate = self.retention * self._residual + self.learning_gain * error
        self._residual = np.clip(candidate, -self.residual_limit, self.residual_limit)
        self._trial += 1
        converged = previous is not None and abs(previous - current) <= self.convergence_tolerance
        snapshot = AdaptationSnapshot(
            adaptation_id=f"ilc-trial-{self._trial}",
            body_hash=self.body_hash,
            source_receipt_hashes=(source_receipt_hash,),
            update={
                "trial": self._trial,
                "shape": list(error.shape),
                "learning_gain": self.learning_gain,
                "retention": self.retention,
                "residual_limit": self.residual_limit,
                "residual_peak": float(np.max(np.abs(self._residual))),
            },
            bounded=bool(np.max(np.abs(self._residual)) <= self.residual_limit),
        )
        update = ILCUpdate(
            trial=self._trial,
            previous_error_rms=previous if previous is not None else current,
            current_error_rms=current,
            residual_peak=float(np.max(np.abs(self._residual))),
            converged=converged,
            snapshot=snapshot,
        )
        self._previous_error_rms = current
        return update


class BoundedTrajectoryILC:
    """Trial-to-trial joint feed-forward update with temporal smoothing."""

    def __init__(
        self,
        *,
        body_hash: str,
        regime_hash: str,
        joint_names: tuple[str, ...],
        learning_gain: float = 0.12,
        retention: float = 0.98,
        residual_limit: float = 0.04,
        smoothing_passes: int = 2,
    ) -> None:
        if not 0.0 < learning_gain <= 1.0:
            raise ValueError("learning_gain must be in (0, 1]")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        if residual_limit <= 0.0 or smoothing_passes < 0:
            raise ValueError("residual_limit must be positive and smoothing_passes non-negative")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        self.body_hash = body_hash
        self.regime_hash = regime_hash
        self.joint_names = joint_names
        self.learning_gain = float(learning_gain)
        self.retention = float(retention)
        self.residual_limit = float(residual_limit)
        self.smoothing_passes = smoothing_passes

    def update(
        self,
        *,
        previous: ILCFeedforward | None,
        tracking_error: np.ndarray,
        source_receipt_hash: str,
        learning_scale: float = 1.0,
    ) -> ILCFeedforward:
        error = np.asarray(tracking_error, dtype=np.float64)
        if error.ndim != 2 or error.shape[1] != len(self.joint_names) or not error.size:
            raise ValueError("tracking_error must be non-empty [time, joint] data")
        if not np.all(np.isfinite(error)):
            raise ValueError("tracking_error must be finite")
        if not 0.0 < learning_scale <= 10.0:
            raise ValueError("learning_scale must be in (0, 10]")
        if previous is None:
            prior = np.zeros_like(error)
            trial = 1
            sources = (source_receipt_hash,)
        else:
            previous.require_compatible(
                body_hash=self.body_hash,
                regime_hash=self.regime_hash,
                joint_names=self.joint_names,
            )
            if previous.values.shape != error.shape:
                raise ValueError("tracking_error shape differs from the pinned feedforward")
            prior = previous.values
            trial = previous.trial + 1
            sources = (*previous.source_receipt_hashes[-3:], source_receipt_hash)
        candidate = self.retention * prior + self.learning_gain * learning_scale * error
        for _ in range(self.smoothing_passes):
            padded = np.pad(candidate, ((1, 1), (0, 0)), mode="edge")
            candidate = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
        bounded = np.clip(candidate, -self.residual_limit, self.residual_limit)
        return ILCFeedforward(
            body_hash=self.body_hash,
            regime_hash=self.regime_hash,
            joint_names=self.joint_names,
            values=bounded,
            residual_limit=self.residual_limit,
            trial=trial,
            source_receipt_hashes=sources,
        )
