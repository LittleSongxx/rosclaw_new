# MotionDecode source qualification

ROSClaw treats MotionDecode CSVs as external motion references, not as trusted
robot actions or offline-RL transitions. The adapter does not download data,
authorize hardware, persist derived motion, or activate a policy. Its repair
stage can only dry-run a bounded terminal-reset trim in memory.

## Evidence loop

Use an operator-managed local snapshot and pin a 40-character dataset commit.
Write all generated evidence outside the source checkout.

```bash
.venv/bin/python -m rosclaw.entrypoint collective source add motiondecode \
  --dataset-root /data/MotionDecode \
  --revision <pinned-commit> \
  --usage research_noncommercial \
  --license-decision pending \
  --terms-file /data/MotionDecode/LICENSE \
  --terms-uri <revision-pinned-license-url> \
  --families football,balance,gait,transition_recovery \
  --limit 400 \
  --output /evidence/motiondecode-registration.json

.venv/bin/python -m rosclaw.entrypoint collective source inspect motiondecode \
  --registration /evidence/motiondecode-registration.json

.venv/bin/python -m rosclaw.entrypoint collective ingest motiondecode \
  --registration /evidence/motiondecode-registration.json \
  --dataset-root /data/MotionDecode \
  --target-model e-urdf-zoo/g1/robot.mjcf.xml \
  --output /evidence/motiondecode-ingest.json

.venv/bin/python -m rosclaw.entrypoint collective repair motiondecode \
  --registration /evidence/motiondecode-registration.json \
  --ingest-report /evidence/motiondecode-ingest.json \
  --dataset-root /data/MotionDecode \
  --target-model e-urdf-zoo/g1/robot.mjcf.xml \
  --output /evidence/motiondecode-repair.json
```

`source add` hashes the catalog and a bounded set of selected CSVs. `inspect`
replays the content-addressed registration without opening the dataset.
`ingest` rehashes every registered file before parsing it, checks the exact
Unitree HG 29-joint mapping, constructs read-only canonical episodes, and
performs finite-value, quaternion, discontinuity, joint-limit, velocity, and
acceleration checks.

`repair` replays the exact registration, source hashes, target body, target
model, ingest report, and audit thresholds. It detects a unique high-speed
reset to the initial pose within the bounded tail window, creates a
content-addressed `SegmentationRepairManifest`, trims the reset frame and
everything after it in memory, recomputes time derivatives, and runs the full
kinematic audit again. The output contains hashes, frame ranges, measurements,
and before/after audits, but no raw or repaired motion arrays.
Downstream simulation must call `replay_segmentation_repair`; it reconstructs
the retained prefix from the immutable source, replays every lineage and
detector commitment, and refuses a prefix that no longer passes Q1.

## Fail-closed rules

- A dataset card or public download does not prove training permission.
  `PERMITTED` requires a non-empty terms snapshot, revision-pinned URI,
  attribution, and an explicit legal decision.
- An empty or absent terms file can only produce `PENDING`.
- MotionDecode v1 CSVs have no action, reward, transition, contact, or
  synchronized ball semantics. They cannot be labeled as behavior-cloning,
  football-contact, or offline-RL data.
- Kinematic audit stops at Q1. Only later Q3/Q4 MuJoCo qualification may allow
  motion-tracker training.
- Automatic repair supports only `trim_tail_before_terminal_reset`. Multiple
  reset candidates, a reset outside the bounded tail, an explicit timeline,
  too few retained frames, root jumps outside the reset boundary, parser
  errors, and unrelated kinematic errors are rejected.
- A repaired clip is Q1 only after its retained frames pass the same audit.
  Repair never implies Q2/Q3 physics trackability or training permission.
- External evidence may discover candidates but cannot be Promotion truth.
- Every receipt declares `hardware_authorized=false`; no ROS, DDS, motor, or
  real-robot path exists in this adapter.

Loop-reset or clip-concatenation boundaries are errors, even if the reset is
near the end of a file. The separate repair manifest preserves the original
source hash, revision, attribution, motion family, retained prefix, and
left/right joint semantics. It cannot weaken these invariants.
