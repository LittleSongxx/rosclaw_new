"""Candidate gate pipeline (PR-EVO-HW-3, 真机自进化v2 §Phase 5).

Every candidate must pass, in order:

    Schema Gate → Applicability Gate → Choreography Gate
    → Timeline Replay (L1) → RH56 Shadow (L2) → Resource Budget
    → Safety Invariants

Any failure → ``REJECTED`` with the failed gate named; a rejected
candidate never reaches hardware.  The L1/L2 layers are the
TASK-RELEVANT sandbox the v2 doc demands (§2.3/§7.14): L1 replays the
real baseline session's round timings; L2 runs the real task code path
with mock executors and asserts ``hardware_actions_executed == 0``.

Shadow disclosure: L2 uses mock hands/camera BY DESIGN — it makes no
perception or hardware claims; it validates the candidate parameter
lifecycle (cooldowns actually slept, rehome actually invoked) plus
timing/deadline behavior.  Formal acceptance runs (baseline/canary/
recurrence) never use mocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate
from .contracts import EvoRpsConfig

GATE_ORDER = (
    "schema",
    "applicability",
    "choreography",
    "timeline_replay",
    "shadow",
    "resource_budget",
    "safety_invariants",
)

SHADOW_DISCLOSURE = (
    "L2 shadow: mock hands + mock camera BY DESIGN (no perception/hardware "
    "claims; validates candidate lifecycle + timing only)"
)


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    passed: bool
    failed_gate: str | None
    verdicts: tuple[GateVerdict, ...]


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def schema_gate(candidate: Candidate) -> GateVerdict:
    missing = [
        key
        for key in ("candidate_id", "changes", "source_failure", "current_regime", "constraints")
        if not getattr(candidate, key, None) and key != "changes"
    ]
    if "max_round_duration_ms" not in candidate.constraints:
        missing.append("constraints.max_round_duration_ms")
    if not candidate.constraints.get("no_servo_speed_change"):
        missing.append("constraints.no_servo_speed_change")
    if not candidate.constraints.get("no_trajectory_change"):
        missing.append("constraints.no_trajectory_change")
    if missing:
        return GateVerdict("schema", False, f"missing: {missing}")
    return GateVerdict("schema", True, "candidate carries id/changes/provenance/constraints")


def applicability_gate(candidate: Candidate, config: EvoRpsConfig) -> GateVerdict:
    errors = config.candidate_space.validate_candidate(candidate.changes)
    forbidden = set(candidate.changes) & set(config.forbidden_parameters)
    if forbidden:
        errors.append(f"forbidden parameters present: {sorted(forbidden)}")
    if errors:
        return GateVerdict("applicability", False, "; ".join(errors))
    return GateVerdict("applicability", True, "in-space, non-stacked, no forbidden params")


def choreography_gate(candidate: Candidate, validator: Any, timing_model: Any) -> GateVerdict:
    """The production ChoreographyValidator decides (SAFE-2)."""
    if not candidate.changes:
        return GateVerdict("choreography", True, "empty patch (C0) — timing identity")
    validation = validator.validate(candidate.changes, timing_model)
    if validation.allowed:
        return GateVerdict(
            "choreography",
            True,
            "contract allows the patch",
            {
                "reveal_window_preserved": validation.reveal_window_preserved,
                "total_round_budget_preserved": validation.total_round_budget_preserved,
            },
        )
    return GateVerdict(
        "choreography",
        False,
        f"contract blocked: {validation.violations}",
        {"violations": validation.violations},
    )


def timeline_replay_gate(
    candidate: Candidate,
    round_durations_ms: list[float],
    *,
    max_round_duration_ms: float,
    budget_utilization_max: float = 1.0,
) -> GateVerdict:
    """L1: counterfactual replay over the REAL baseline round timings.

    A cooldown candidate adds its cooldown to every (or every Nth) round;
    the replayed durations must still fit the round budget, and the added
    overhead is reported honestly — a candidate that 'works' only by
    stalling is visible in the numbers.
    """
    if not round_durations_ms:
        return GateVerdict("timeline_replay", False, "no baseline round timings available")
    cooldown_s = float(candidate.changes.get("inter_round_cooldown_sec") or 0.0)
    every_n = int(candidate.changes.get("cooldown_every_n_rounds") or 0)
    replayed: list[float] = []
    overhead_s = 0.0
    for index, duration in enumerate(round_durations_ms, start=1):
        extra = 0.0
        if cooldown_s > 0:
            extra = cooldown_s
        elif every_n > 0 and index % every_n == 0:
            extra = 5.0  # cooldown_every_n_rounds uses the 5s pause (run2 semantics)
        replayed.append(duration + extra * 1000.0)
        overhead_s += extra
    max_duration = max(replayed)
    utilization = max_duration / max_round_duration_ms
    metrics = {
        "baseline_rounds": len(round_durations_ms),
        "baseline_max_ms": max(round_durations_ms),
        "replayed_max_ms": max_duration,
        "budget_utilization": round(utilization, 4),
        "cooldown_overhead_s_per_session": round(overhead_s, 2),
    }
    if utilization > budget_utilization_max:
        return GateVerdict(
            "timeline_replay",
            False,
            f"replayed max round {max_duration:.0f}ms exceeds budget "
            f"{max_round_duration_ms:.0f}ms",
            metrics,
        )
    return GateVerdict("timeline_replay", True, "replayed durations fit the round budget", metrics)


def shadow_gate(
    candidate: Candidate,
    shadow_run: Any | None = None,
) -> GateVerdict:
    """L2: the candidate ran the real task code path with ZERO hardware actions.

    ``shadow_run`` is the result of the driver's mock-executor session
    (injected for tests; produced by the orchestrator in production).
    """
    if shadow_run is None:
        return GateVerdict("shadow", False, "no shadow run evidence produced")
    if shadow_run.get("hardware_actions_executed") != 0:
        return GateVerdict(
            "shadow",
            False,
            f"hardware actions executed during shadow: "
            f"{shadow_run.get('hardware_actions_executed')}",
        )
    if not shadow_run.get("rounds_completed"):
        return GateVerdict("shadow", False, "shadow session completed no rounds")
    lifecycle = shadow_run.get("candidate_lifecycle") or {}
    expected_cooldown = float(candidate.changes.get("inter_round_cooldown_sec") or 0.0)
    if expected_cooldown > 0 and not lifecycle.get("cooldown_applied"):
        return GateVerdict(
            "shadow",
            False,
            f"cooldown {expected_cooldown}s never applied in the shadow session",
        )
    return GateVerdict(
        "shadow",
        True,
        SHADOW_DISCLOSURE,
        {"rounds_completed": shadow_run["rounds_completed"], "lifecycle": lifecycle},
    )


def resource_budget_gate(
    candidate: Candidate,
    *,
    baseline_runtime_s: float,
    cooldown_overhead_s: float,
    max_factor: float = 2.0,
) -> GateVerdict:
    estimated = baseline_runtime_s + cooldown_overhead_s
    metrics = {
        "baseline_runtime_s": round(baseline_runtime_s, 1),
        "estimated_runtime_s": round(estimated, 1),
        "max_factor": max_factor,
    }
    if estimated > baseline_runtime_s * max_factor:
        return GateVerdict(
            "resource_budget",
            False,
            f"estimated session runtime {estimated:.0f}s > {max_factor}× baseline "
            f"{baseline_runtime_s:.0f}s",
            metrics,
        )
    return GateVerdict("resource_budget", True, "session runtime stays within budget", metrics)


def safety_invariants_gate(candidate: Candidate, config: EvoRpsConfig) -> GateVerdict:
    problems: list[str] = []
    telemetry = candidate.changes.get("telemetry_hz")
    if telemetry is not None and float(telemetry) < 1:
        problems.append(f"telemetry_hz {telemetry} < 1 — regime features would starve")
    if float(candidate.changes.get("inter_round_cooldown_sec") or 0.0) > 8.0:
        problems.append("cooldown above the documented 8s bound (§4.1)")
    if problems:
        return GateVerdict("safety_invariants", False, "; ".join(problems))
    return GateVerdict(
        "safety_invariants",
        True,
        "abort thresholds untouched, telemetry floor kept, documented bounds held",
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def evaluate_candidate(
    candidate: Candidate,
    config: EvoRpsConfig,
    *,
    validator: Any,
    timing_model: Any,
    round_durations_ms: list[float],
    baseline_runtime_s: float,
    shadow_run: Any | None = None,
) -> CandidateEvaluation:
    """Run the full gate pipeline in order; stop at the first failure."""
    verdicts: list[GateVerdict] = []
    for gate in GATE_ORDER:
        if gate == "schema":
            verdict = schema_gate(candidate)
        elif gate == "applicability":
            verdict = applicability_gate(candidate, config)
        elif gate == "choreography":
            verdict = choreography_gate(candidate, validator, timing_model)
        elif gate == "timeline_replay":
            verdict = timeline_replay_gate(
                candidate,
                round_durations_ms,
                max_round_duration_ms=float(candidate.constraints["max_round_duration_ms"]),
            )
        elif gate == "shadow":
            verdict = shadow_gate(candidate, shadow_run)
        elif gate == "resource_budget":
            cooldown_overhead = 0.0
            cooldown_s = float(candidate.changes.get("inter_round_cooldown_sec") or 0.0)
            every_n = int(candidate.changes.get("cooldown_every_n_rounds") or 0)
            if cooldown_s > 0:
                cooldown_overhead = cooldown_s * len(round_durations_ms)
            elif every_n > 0:
                cooldown_overhead = 5.0 * (len(round_durations_ms) // every_n)
            verdict = resource_budget_gate(
                candidate,
                baseline_runtime_s=baseline_runtime_s,
                cooldown_overhead_s=cooldown_overhead,
            )
        else:
            verdict = safety_invariants_gate(candidate, config)
        verdicts.append(verdict)
        if not verdict.passed:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                passed=False,
                failed_gate=gate,
                verdicts=tuple(verdicts),
            )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        passed=True,
        failed_gate=None,
        verdicts=tuple(verdicts),
    )


# ---------------------------------------------------------------------------
# Baseline timing extraction (for L1 + the choreography timing model)
# ---------------------------------------------------------------------------


def round_durations_from_events(events_path: Path) -> list[float]:
    """Round PERIODS (start → next start, ms) from practice events.

    The contract's round budget governs ``start → next start`` — pairing
    started/resolved would undercount the budget-relevant period (result
    display + ready + interval are part of the round).
    """
    starts: list[float] = []
    if not Path(events_path).is_file():
        return []
    with open(events_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = str(event.get("event_type") or event.get("type") or "")
            if not etype.endswith("round.started"):
                continue
            ts = event.get("timestamp") or event.get("ts")
            if ts is None and isinstance(event.get("timestamp_ns"), (int, float)):
                ts = float(event["timestamp_ns"]) / 1e9
            if isinstance(ts, (int, float)):
                starts.append(float(ts))
    return [max(0.0, (b - a) * 1000.0) for a, b in zip(starts, starts[1:], strict=False)]
