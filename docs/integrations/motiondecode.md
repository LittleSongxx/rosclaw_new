# MotionDecode source qualification

ROSClaw treats MotionDecode CSVs as external motion references, not as trusted
robot actions or offline-RL transitions. The adapter does not download data,
authorize hardware, repair clips, or activate a policy.

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
```

`source add` hashes the catalog and a bounded set of selected CSVs. `inspect`
replays the content-addressed registration without opening the dataset.
`ingest` rehashes every registered file before parsing it, checks the exact
Unitree HG 29-joint mapping, constructs read-only canonical episodes, and
performs finite-value, quaternion, discontinuity, joint-limit, velocity, and
acceleration checks.

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
- External evidence may discover candidates but cannot be Promotion truth.
- Every receipt declares `hardware_authorized=false`; no ROS, DDS, motor, or
  real-robot path exists in this adapter.

Loop-reset or clip-concatenation boundaries are errors, even if the reset is
near the end of a file. A future repair stage must emit a separate,
content-addressed repair manifest, preserve the source hash and semantics, and
pass the full audit again before MuJoCo qualification.
