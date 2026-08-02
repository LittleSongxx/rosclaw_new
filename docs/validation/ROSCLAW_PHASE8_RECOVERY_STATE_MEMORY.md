# Phase 8 recovery-state memory: closed-loop review

## Outcome

This increment implements a SIM-only, short-horizon recovery-state memory for
G1 post-kick control.  It is useful infrastructure, but the learned candidate
is **not promoted**.  DEVELOPMENT and private fallback checks passed; sealed
local-physics VALIDATION exposed unsafe generalization of the discrete
open-loop recovery primitives.  The promotion gate returned rejection and the
retained recovery controller remains authoritative.

The final post-rejection hardening requires two of two nearby DEVELOPMENT
examples to carry positive evidence.  On the same eight v6 stress cases this
changed two unsafe learned routes into fallback, producing eight of eight
strict, exact parent replays.  That replay is a safety check, not a new
performance qualification.

## What was added

- `g1_recovery_state_memory.py` defines a content-addressed artifact and a
  deterministic evidence-neighborhood policy.
- A five-policy-frame sequence records both the mean state and its trend.
- The observation contract combines proprioception with ball-relative
  position and velocity: 22 raw observations, 16 descriptor features, and a
  32-value mean-plus-trend descriptor.
- Positive examples carry a bounded recovery primitive and measured composite
  and component lower bounds.  Failed searches are explicit negative
  prototypes, not discarded data.
- Missing, non-numeric, non-finite, feature-envelope, distance, negative-nearest,
  consensus, advantage, and component gates all fail closed.
- Selection is allowed immediately after observed ball contact and right-foot
  landing, but target adaptation remains gated by the primitive's original
  start frame.  This lets the controller observe before acting without
  introducing a target discontinuity.
- `G1CerebellarRecoveryController` accepts exactly one learned recovery mode at
  a time, binds artifacts to Body/motion/baseline/fallback hashes, and records
  the recovery-state receipt.
- The MuJoCo backend records observation-update and policy-frame evidence plus
  the expanded environment-and-body state.  It remains CPU physics and does
  not open ROS, DDS, simulator bridges, or hardware transport.
- The CLI adds `goalforge muscle-memory state-train` and `state-inspect`.

## Why the earlier router failed

The first implementation revealed two train/deploy mismatches:

1. The offline window required right-foot support on every sample even though
   runtime landing was latched.  Some moving-ball descriptors therefore came
   from different policy frames.  Recording the exact observation-update
   event and mirroring the landing latch reduced same-case descriptor distance
   to exactly zero.
2. Collecting five frames starting at recovery frame 300 delayed execution to
   roughly frame 304.  In this contact-sensitive motion, four frames were
   enough to turn a positive primitive into a large regression.  Moving
   observation before the action gate restored byte-identical equivalence
   between a forced primitive route and direct baseline execution.

The later v5 test found a representation gap: states with almost identical
body descriptors (distance 0.0017--0.0024) could diverge from success to fall
after small ball-restitution changes.  Ball-relative position and velocity
were therefore added in v6, and the failed restitution points became the next
generation's positive/negative curriculum.

## v6 evidence

All figures below are aggregate, frozen-policy CPU MuJoCo evidence from
`/code/rosclaw/phase8_evidence/g1-recovery-state-v6`.

| Suite | Result | Learned / fallback | Key measurements |
|---|---:|---:|---|
| DEVELOPMENT | pass, 27/27 | 6 / 21 | backstep +28.97%, tail wobble +25.18%, leg jerk +6.58%, composite +22.98%, bootstrap lower +7.36% |
| sealed VALIDATION | **reject**, 6/8 passed | 2 / 6 | routed mean tail wobble -5911.76%, leg jerk -100.56%, composite -2363.17%; 6/8 parent-valid |
| private HOLDOUT | pass, 3/3 | 0 / 3 | 3/3 strict and exact parent fallback |

The large validation regressions are not clipped or omitted.  They demonstrate
that a nearest-state selector over discrete open-loop primitives does not yet
model the action-conditioned future of the contact dynamics.  More features
alone are insufficient.

After rejection, the consensus gate was raised from 0.5 to 1.0.  A safety
replay using artifact hash
`sha256:64bfa6b216caf5115d34be936c3af7040fab161c84b068575d258c1df23d3754`
produced 0 learned routes, 8 fallback routes, 8 strict replays, and 8 exact
parent replays on the v6 validation suite.  This hardened policy is not claimed
as a performance candidate.

## Stability-plasticity interpretation

The useful result is not that the G1 has learned a generally better kick; it
has not.  The useful result is an operational stability-plasticity loop:

1. search bounded actions on DEVELOPMENT;
2. store success and failure as signed episodic memory;
3. select only inside a content-bound evidence neighborhood;
4. test the frozen policy on unseen local physics;
5. reject unsafe generalization;
6. convert the failure boundary into the next generation's curriculum;
7. tighten the safety gate immediately while preserving the old parent.

This loop prevented all failed candidates from silently replacing the known
controller.  It also showed where the next model must be qualitatively
different.

## Next implementation

The next candidate should replace nearest-primitive routing with an
action-conditioned recurrent critic.  For each bounded primitive (including
the retained parent), it should predict the vector of future backstep, wobble,
jerk, safety, and goal outcomes plus epistemic uncertainty from the joint
body-ball history.  A primitive may run only when its conservative lower bound
dominates the parent on every guardrail.  The actor should then modulate the
selected primitive through bounded closed-loop target residuals rather than
execute a purely open-loop switch.

Dream/replay training should prioritize validation-failure boundaries, retain
parent-anchor and negative exemplars, and use a fresh sealed interpolation
suite for every generation.  No promotional video should be generated from a
rejected candidate; the existing qualified contextual-recovery video remains
the current visual result.

## Safety status

- Activation ceiling: `SIM_ONLY`
- Physics authority: `CPU_MUJOCO`
- Hardware command sent: `false`
- Candidate promotion: `REJECTED`
- Retained parent: unchanged

## Verification

- Focused recovery/controller/GoalForge suite: `53 passed, 2 deselected`.
- Ruff over all `src` and `tests`: passed.
- CI mypy gate: `203` source files passed.
- Changed recovery/backend source files under mypy: passed.
- Full pytest with the explicit isolated LeRobot runtime:
  `5416 passed, 59 skipped, 27 deselected, 26 warnings` in `1043.67 s`.
- v6 training/evaluation: `312` training-side, `32` validation, and `12`
  holdout CPU MuJoCo rollouts; no CUDA physics authority and no hardware
  command.
