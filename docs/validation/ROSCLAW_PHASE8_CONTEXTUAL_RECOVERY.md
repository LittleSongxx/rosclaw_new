# Phase 8 contextual G1 recovery: implementation and falsification report

## Status

This increment is **SIM-only** and **not promoted**.

It demonstrates a material learned contribution on the DEVELOPMENT regimes,
but the frozen policy did not generalize to the sealed VALIDATION regimes.
The runtime therefore routes those states to the retained parent.  That safe
fallback is correct behavior, but it is not evidence of an independently
generalizing learned skill.

## What changed

The earlier temporal muscle-memory actor added a small joint residual after a
single structured recovery primitive.  Its isolated learned contribution was
only about `0.008%` to `0.014%`, far below the Phase 8 `5%` gate.

This increment adds a content-addressed contextual motor memory:

1. CPU MuJoCo evaluates a bounded library of target-space recovery primitives.
2. The learner records the causal proprioceptive state at the first recovery
   frame, after contact and kick-foot landing.
3. A nearest-prototype actor selects the primitive associated with that state.
4. The selection is latched for the episode.
5. Missing, non-finite, incompatible, or out-of-distribution observations
   fail closed to the retained parent primitive.

The actor cannot emit torque, raw joint targets, ROS/DDS commands, leases, or
hardware authorization.  Its activation ceiling is `SIM_ONLY`.

The model artifact binds:

- qualified Body and motion hashes;
- fixed structured and retained fallback configuration hashes;
- normalization and proprioceptive feature contracts;
- DEVELOPMENT trajectory and primitive-trial commitments;
- validation-suite commitment;
- bounded primitives and their prototypes.

## Controller defect fixed

The slow structured controller previously stored the target *after* adding
the learned muscle residual as its next smoothing state.  This unintentionally
integrated a supposedly bounded per-frame residual and could create drift or
twitching.

The smoothing state now stores only the structured target.  A regression test
applies the same learned residual on consecutive frames and proves that it is
not integrated into the next frame.

## Closed-loop qualification

The qualification distinguishes three comparisons:

- **Safety and retention:** candidate versus the retained parent.
- **Learned causal contribution:** contextual candidate versus the same fixed
  structured controller with zero learning.
- **Generalization:** frozen candidate on a precommitted VALIDATION suite.

Promotion gates include:

- mean learned composite contribution at least `5%`;
- deterministic bootstrap 95% lower bound greater than zero;
- no learned component regressing more than `2%` on DEVELOPMENT;
- goal, fall, joint, torque, saturation, slip, bilateral-support, recovery-time,
  and naturalness guards;
- exact deterministic replay;
- private HOLDOUT routing to and exactly replaying the retained parent.

## Results

The strongest frozen DEVELOPMENT run is
`g1-contextual-recovery-v4`.

| Metric | Learned contribution versus fixed structured baseline |
| --- | ---: |
| Backward reversal | `22.65%` reduction |
| Tail wobble | `18.86%` reduction |
| Leg jerk | `6.08%` reduction |
| Weighted composite | `17.82%` improvement |
| Bootstrap 95% lower bound | `3.51%` |
| Worst DEVELOPMENT case | `1.98%` improvement |

All five DEVELOPMENT cases were safe, goal-preserving, naturalness-preserving,
and strict-replay deterministic.

The private retention suite passed:

| Retention check | Result |
| --- | ---: |
| Cases | `3 / 3` |
| OOD fallback routes | `3 / 3` |
| Strict replays | `3 / 3` |
| Exact retained-parent trajectories | `3 / 3` |

The sealed VALIDATION suite did **not** pass.  The richer delayed
proprioceptive router correctly classified all three cases as outside its
covered state set and selected the retained parent.  Because the learned
primitive was not applied, this is a safety success but a learning
generalization failure.  The candidate remains rejected.

Earlier frozen variants that used the contact/landing snapshot did select an
expert primitive on novel states, but produced large wobble regression.  This
falsified the assumption that a single impact snapshot is sufficient for
reliable recovery selection.

## Stage video

The external evidence directory contains a split-screen DEVELOPMENT replay:

`/code/rosclaw/phase8_evidence/g1-contextual-recovery-v4/g1-contextual-recovery-development.mp4`

It compares:

- left: fixed structured recovery with zero learning;
- right: learned contextual router.

The selected DEVELOPMENT case measures:

| Metric | Improvement |
| --- | ---: |
| Backward reversal | `84.56%` |
| Tail wobble | `74.17%` |
| Leg jerk | `19.13%` |
| Composite | `67.32%` |

The frame overlay and hash-bound manifest explicitly say:

`DEVELOPMENT ONLY · SEALED VALIDATION FAILED · NOT PROMOTED`

The video is visualization downstream of strict MuJoCo evidence and cannot
create or upgrade task evidence.

## Interpretation

The result is useful precisely because it separates memorization from growth.
ROSClaw can now:

- discover materially different recovery strategies from data;
- bind them into a portable, auditable motor-memory artifact;
- use proprioception to select them online;
- reproduce large local improvements;
- recognize when a novel body state is not covered;
- preserve historical behavior by exact parent fallback.

It has not yet shown that the selected motor memory generalizes through the
highly discontinuous RoboNaldo contact dynamics.  More optimizer budget does
not solve that representation problem.

## Next implementation target

The next candidate should replace hard nearest-prototype selection with a
short-horizon recovery-state model trained from a denser applicability-gated
curriculum.  Promotion still requires CPU MuJoCo truth and a new sealed suite.
The model should estimate both expected return and epistemic uncertainty, and
it should abstain unless the lower confidence bound beats the fixed structured
baseline.  The current artifact, failed validation reports, and exact fallback
suite provide the anchors needed to prevent forgetting while that model is
trained.

That follow-up has now been implemented and evaluated through six evidence
generations.  Its architecture, rejected local-physics validation, fail-closed
hardening, and action-conditioned critic follow-up are documented in
`ROSCLAW_PHASE8_RECOVERY_STATE_MEMORY.md`.  The rejected state-memory artifact
does not supersede the results in this document.

## Terminal-damping follow-up and multi-challenge video

The `g1-contextual-recovery-v5-terminal-damping` follow-up adds a third,
smoothstep-gated recovery stage after unloading and upright settling.  It
reduces leg proportional gain to `0.93x` and raises leg derivative damping to
`1.50x`, starting at policy frame `500` over `80` frames.  It does not change
the kick, emit torque, or cross the `SIM_ONLY` boundary.

The parameters were selected from DEVELOPMENT-only sweeps and then evaluated
against an exact ablation with the same contextual primitive and terminal
damping disabled:

| Terminal-damping metric | Five-case DEVELOPMENT result |
| --- | ---: |
| Passed cases | `5 / 5` |
| Mean backward reversal | `5.82%` reduction |
| Mean tail joint jerk | `21.05%` reduction |
| Minimum tail joint jerk | `14.33%` reduction |
| Mean tail wobble | `0.89%` regression |
| Worst tail wobble | `2.24%` regression |

This is a deliberate stability/plasticity trade: visible late joint twitch is
materially lower and backward reversal improves, while the aggregate and
per-case wobble regressions remain inside the predeclared `2%`/`5%` limits.
Gentle upper-body and whole-body damping were also tested; neither matched the
leg-only controller's combined backstep and tail-jerk gains.

The contextual DEVELOPMENT result also improved relative to the matched fixed
structured baseline:

| Metric | v5 learned contribution |
| --- | ---: |
| Backward reversal | `28.56%` reduction |
| Tail wobble | `20.22%` reduction |
| Leg jerk | `6.46%` reduction |
| Weighted composite | `20.81%` improvement |
| Bootstrap 95% lower bound | `5.81%` |
| Worst DEVELOPMENT case | `4.41%` improvement |

The candidate remains **rejected** because sealed VALIDATION again routed all
three states to the retained parent (`0 / 3` contextual routes).  Private
retention remained exact (`3 / 3`).

The new external video is:

`/code/rosclaw/phase8_evidence/g1-contextual-recovery-v5-terminal-damping/g1-contextual-recovery-multichallenge.mp4`

It is `49.83 s`, H.264, `1280x720`, `30 fps`, and contains five strict-replay
experiments: a `0.55 m` high target with `0.10 m` lateral ball offset, nominal
moving ball, lateral moving ball, speed shift, and lighter ball.  The high
target demonstrates exact OOD fallback; the other four compare the same
contextual primitive with damping off versus on.  The manifest binds all ten
MuJoCo trajectories and keeps the `SEALED VALIDATION FAILED · NOT PROMOTED`
label visible throughout.

## Verification

- Ruff check on all `src` and `tests`: passed.
- Ruff format on the seven changed files: passed.  The older chained base has
  108 unrelated pre-existing files outside this change that do not match the
  current formatter; they were not rewritten.
- Mypy on the five changed source files: passed.
- Repository CI mypy gate: `203` source files passed.
- Full pytest with the existing isolated LeRobot runtime:
  `5406 passed`, `59 skipped`, `27 deselected`, `26 warnings` in `1067.60 s`.
- Contextual/cerebellar focused suite: `27 passed`.
- Multi-challenge video: H.264, `1280x720`, `30 fps`, `1495` frames,
  `49.83 s`; manifest and ten trajectory hashes verified.
