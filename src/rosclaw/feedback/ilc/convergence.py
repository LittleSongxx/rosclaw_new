"""Promotion-oriented convergence checks for iterative learning control."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.feedback.ilc.trajectory_memory import ILCTrajectory


@dataclass(frozen=True)
class ILCConvergence:
    monotonic_error: bool
    safety_not_increased: bool
    energy_within_limit: bool
    error_reduction: float
    passed: bool


def assess_ilc_convergence(
    trajectories: tuple[ILCTrajectory, ...],
    *,
    energy_growth_limit: float = 0.10,
    minimum_error_reduction: float = 0.01,
    tolerance: float = 1e-9,
) -> ILCConvergence:
    if len(trajectories) < 2:
        raise ValueError("ILC convergence requires at least two trials")
    if not 0.0 <= minimum_error_reduction < 1.0:
        raise ValueError("minimum_error_reduction must be in [0, 1)")
    errors = [item.error_rms for item in trajectories]
    monotonic = all(
        current <= previous + tolerance
        for previous, current in zip(errors, errors[1:], strict=False)
    )
    interventions = [item.safety_interventions for item in trajectories]
    safe = all(
        current <= previous
        for previous, current in zip(interventions, interventions[1:], strict=False)
    )
    baseline_energy = max(trajectories[0].energy, 1e-12)
    energy_ok = max(item.energy for item in trajectories) <= baseline_energy * (
        1.0 + energy_growth_limit
    )
    reduction = (errors[0] - errors[-1]) / max(errors[0], 1e-12)
    return ILCConvergence(
        monotonic,
        safe,
        energy_ok,
        reduction,
        monotonic and safe and energy_ok and reduction >= minimum_error_reduction,
    )
