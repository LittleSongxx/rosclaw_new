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

## Verification

- Ruff on `src` and `tests`: passed.
- Mypy: `895` source files passed.
- Full pytest: `5389 passed`, `67 skipped`, `27 deselected`; four LeRobot
  tests initially failed because collection saw a persisted runtime while the
  isolated test home had no runtime registration.
- LeRobot failure set rerun with the existing real runtime at
  `/code/rosclaw/phase6_runtime/lerobot-0.6.1/bin/python`: `4 passed`.
- Contextual recovery focused suite after final fail-closed hardening:
  `37 passed`.
- Video: H.264, `1280x720`, `30 fps`, `285` frames, `9.5 s`; manifest and
  trajectory hashes verified.
