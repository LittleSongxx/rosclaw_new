# ROSClaw Feedback Plane — Phase 6 Foundation

The Feedback Plane is ROSClaw's synchronous residual-control layer. It runs
between a qualified base policy and the existing low-level servo/safety
boundary:

```text
base policy + feedback residual + learned feed-forward residual
                         |
                         v
                  Safety Projector
                         |
                         v
             existing PD/torque-limited servo
```

It does not replace Unitree's L0 servo, use the EventBus in the hot path, call
an LLM, write Practice evidence synchronously, or grant hardware authority.
The pre-existing `rosclaw.feedback` user-feedback/telemetry API remains
compatible; control-plane types are additive.

## Implemented foundation

- Immutable, content-addressed `FeedbackLoopSpec`, `FeedbackFrame`,
  `ErrorState`, `ResidualCommand`, `FeedbackReceipt`, `ControllerSnapshot`,
  and `AdaptationSnapshot` contracts.
- Absolute-deadline `FixedRateScheduler` and a synchronous `FeedbackRuntime`
  with observation-age checks, deadline monitoring, deterministic estimator
  state, preallocated NumPy ring buffers, and post-run receipt generation.
- `FeedbackSafetyProjector` for signal allowlisting, finite-value validation,
  residual limits, optional absolute-action limits, saturation accounting, and
  fail-closed fallback.
- General vector PID and phase-latched controller primitives.
- A G1 kick-balance reflex that observes torso orientation and COM/support
  error, then emits bounded hip, ankle, and waist target residuals.
- A bounded GoalForge L2 skill-feedback contract for pre-contact phase/aim
  directives and post-contact recovery residuals. Its phase-rate output is not
  yet coupled to the MuJoCo policy clock, so no task-improvement claim is made.
- Strict command/trajectory replay using recorded deadline decisions, so a
  host scheduling outlier is evidence rather than hidden nondeterminism.
- Body- and regime-bound ILC trajectory memory, bounded update rules, and
  convergence checks for monotonic error, safety interventions, and energy.
- A transactional ten-trial G1 ILC campaign with raw error trajectories,
  rollback, strict final replay, wrong-regime rejection, and a reloadable,
  content-addressed selected feed-forward artifact.
- An offline self-evolution evaluator that binds the current controller,
  selected ILC artifact, A/B, multi-regime Holdout, historical replay, and DDS
  chaos evidence into one candidate and evaluates F1-F15 without activating it.

## G1 reflex behavior

The qualified RoboNaldo kick prior stays frozen. The reflex is deliberately
transparent during normal motion. It can latch only during the configured
disturbance-detection phase and only after roll, pitch, or COM displacement
crosses the pinned emergency threshold. COM-triggered correction additionally
requires transient motion, avoiding a low-friction quasi-static false trigger.
Once latched, it fades out before the
recovery phase. Every output is projected to a maximum joint-target residual
between `0.04` and `0.08 rad`, depending on the joint.

This trigger-window design is important. A continuously active balance
controller can fight intentional whole-body kick motion even when each local
correction looks reasonable. The emergency gate makes the base policy own its
qualified regime and gives the residual controller only the out-of-envelope
recovery case.

## Evidence path

The control loop retains compact, bounded in-memory records. After execution,
the asynchronous path builds hashes for reference, observation, controller,
specification, commands, and the full trace. `FeedbackReceipt` also reports
latency, jitter, dropped frames, observation age, deadline misses, stale
frames, corrections, safety projections, and saturation.

Raw simulation evidence must remain outside the source checkout. Run the
same-scenario A/B validator with:

```bash
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-reflex \
  --asset-root /path/to/RoboNaldo_Deploy \
  --output /tmp/rosclaw-feedback-validation.json
```

The profile executes 0, 35, 65, and 80 N lateral-disturbance pairs with the
feedback plane off and on, then replays every feedback trajectory using its
recorded deadline decisions. It requires no nominal regression, no feedback
fall/joint/torque regression, at least one same-scenario rescue, at least
99.9% deadline compliance, and strict replay for every pair.

Additional profiles are available:

```bash
# 11 friction/latency/bias/disturbance regimes plus 12 historical motions
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-holdout --asset-root /path/to/RoboNaldo_Deploy \
  --output /tmp/rosclaw-feedback-holdout.json

# Ten selected ILC trials plus disclosed line-search probes
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-ilc --asset-root /path/to/RoboNaldo_Deploy \
  --output /tmp/rosclaw-ilc-validation.json

# Reload the selected ILC artifact, build a candidate, and evaluate F1-F15
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-evolution \
  --feedback-evidence /tmp/rosclaw-feedback-validation.json \
  --holdout-evidence /tmp/rosclaw-feedback-holdout.json \
  --ilc-evidence /tmp/rosclaw-ilc-validation.json \
  --chaos-evidence /tmp/rosclaw-goalforge-chaos.json \
  --output /tmp/rosclaw-feedback-evolution.json
```

`feedback-evolution` deliberately returns a non-zero status for
`NEED_MORE_EVIDENCE` or `REJECTED`. A successfully constructed candidate is
not the same as a promoted controller. Evaluation never writes the Registry,
opens hardware transport, or activates a residual. The result records the
parent snapshot as its rollback target.

## Safety and claim boundary

- Evidence domain is `SIM`; no real robot, real DDS transport, or hardware
  permit is opened.
- MuJoCo at 500 Hz remains task truth. The Feedback Plane profile runs at
  250 Hz and feeds the existing torque-limited PD loop.
- A `passed` A/B, Holdout, or ILC report is not hardware Controller Promotion.
  The reflex-local error gate, hard-real-time qualification, and a separately
  authorized real-body Canary remain incomplete.
- The self-evolution evaluator has no caller-supplied `canary_passed` override.
  F15 remains missing until a future candidate-bound, independently attested,
  and explicitly authorized canary workflow exists.
- The ten-trial ILC claim is MuJoCo physical simulation, not real hardware.
  Candidate step sizes are simulation probes and must not be copied to a real
  robot without digital-twin/shadow screening.
- This is a Python simulation reference runtime, not a hard-real-time or
  allocation-free executor. Contract/trace objects still allocate on the hot
  path; shared-memory/C++ or Rust execution and real scheduler qualification
  remain necessary before hardware promotion.
- Learned residual, online embodiment latent, MPC, KickTwin online
  identification, and physical coupling of L2 skill phase modulation remain
  later Phase 6 work. Canonical Unitree DDS has been validated only on an
  isolated loopback simulator domain.

## Public-project provenance

External assets are qualified by content hash and Git commit and are not
vendored into ROSClaw:

- OpenDriveLab RoboNaldo deployment prior:
  `f60f24459aaabc3aea9187a2b13f8923049b629c`.
- Unitree `unitree_mujoco` timing/model/DDS reference:
  `ae6a8403e272733e9996ef59990880330496177f`.
- Unitree `unitree_rl_mjlab` policy decimation and G1 deployment reference:
  `1425b15f73bd4095f0df53709d7c389c3eb9e790`.

ROSClaw imports none of their hardware transports into this simulation-only
path. Their checked-out code was used to verify the official 2 ms MuJoCo step,
29-DoF joint order, policy/servo separation, and fixed-rate deployment model.
