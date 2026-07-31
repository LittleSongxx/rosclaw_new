# TY1200 Platform — ROSClaw Platform Pack for the Iluvatar TY1200

[中文文档](README.zh-CN.md)

A complete ROSClaw platform-operations package for the **Iluvatar TY1200**
embodied-AI compute box (Intel Core Ultra 7 255H + Iluvatar GPGPU MXM,
16/32 GB HBM2e). It turns the vendor's manual `docker exec` model-serving
setup into a **controlled, auditable, agent-operable stack**.

## What's inside

| Asset | Description |
| --- | --- |
| `rosclaw_ty1200` Python package | Platform detection, ixsmi telemetry, evidence bundles, `modeld` controlled-ops daemon, OpenAI-compatible provider runtime, executable skill runner |
| `skills/ty1200-platform-ops` | Platform Operations **Skill** (declarative assets + executable runner) |
| `providers/` | 4 provider assets: 2 local vLLM services + 2 remote LLM fallbacks |
| `deployment/` | Docker Compose (digest-pinned), systemd units, installer |
| `scripts/` | Ops entry points (preflight / verify / start / stop / support-bundle) |

## The two local model services

| Port | Service | Capabilities |
| --- | --- | --- |
| **8000** | Qwen3-Embedding-0.6B | `embedding.text`, `memory.embed`, `retrieval.embed_*` |
| **8001** | Cosmos-Reason2-2B | `vlm.physical_reasoning`, `vlm.video_question_answering`, `critic.*` (advisory only — never drives robots directly) |

Measured on TY1200 (16 GB): embedding **893 texts/s** @batch128, Chinese
retrieval top1 3/3; Cosmos TTFT **36 ms**, **363 tok/s**, 100% success over
20-request runs; 8 s video reasoning in **6.3 s** end to end.

## Hardware coverage (read-only)

The skill inventories the whole box, not just the model stack:

| Class | Detected on this unit |
| --- | --- |
| Layered compute | GPGPU (MRC-V100) + NPU (`intel_vpu`, `/dev/accel/accel0`) + Arc iGPU (`renderD128`) + P/E-core topology |
| EtherCAT | 2× Intel igb RJ45 (`enp5s0/enp6s0`) — link state reported |
| CAN-FD | PCI CANBUS controller detected; `hardware_present_no_driver` surfaced honestly |
| Serial | RS232/485 UARTs (real-IRQ filtered) |
| I2C / SPI / GPIO | bus/adapters enumerated; unexposed GPIO/SPI reported as WARN, never silently skipped |
| Display / wireless | HDMI connector state, Wi-Fi (`rtw89`), BT (`hci0`), 5G slot, ATSHA204A secure chip |

Ops: `peripheral_inventory`, `accelerator_check`, `rt_check` (PREEMPT_RT
assessment). **Boundary:** GPIO writes, CAN/EtherCAT frames, serial/I2C/SPI
transactions and any robot motion are *robot-domain* operations and are
explicitly out of scope for this skill.

## Architecture

```text
Agent
  ↓
ty1200-platform-ops Skill  (ty1200-ops CLI / SkillRunner)
  ↓  Unix socket, fixed allowlist — no arbitrary shell/docker
rosclaw-ty1200-modeld  (systemd, unprivileged service account)
  ↓  compose file hash-pinned, image digests pinned, approval tokens sha256-checked
Docker Compose model stack
  ↓
vLLM :8000 (embedding)  /  :8001 (cosmos, video via read-only Media Store)
  ↑
Provider Router — capability routing with remote fallback;
media inputs NEVER leave the box
```

## Quick start

```bash
# 1. install (setuptools < 61 targets need the shim; no PyPI required)
pip install --no-build-isolation ./ty1200-platform

# 2. preflight & verify live services
./scripts/ty1200-preflight --json
./scripts/ty1200-model-verify --json --with-inference

# 3. full install: modeld + systemd + compose (requires sudo)
sudo ./deployment/install.sh

# 4. use the skill
ty1200-ops '{"operation":"doctor"}'
ty1200-ops '{"operation":"provider_verify","target_service":"embedding"}'
ty1200-ops '{"operation":"model_stack_restart","target_service":"cosmos","approval_token":"<token>"}'
```

## Security model (L0–L6)

- **L0/L1** (inspect, doctor, verify, benchmark) — agent may run directly
- **L2** (start/stop/restart) — one-shot approval token, verified by modeld as sha256
- **L3** (image pull) — only digest-pinned images from the hash-pinned compose file
- **L4+** (compose changes, drivers, kernel, firmware) — humans only, outside the skill
- Cosmos output is **advisory** (`executable: false`); motion stays behind
  ROSClaw Action/Permit/Lease/Sandbox
- Every execution writes an evidence bundle (manifest + sha256); tokens,
  credentials and chain-of-thought are redacted (hash-only)

## Tested on

- TY1200, Ubuntu 22.04, CoreX SDK 4.4.0, Docker 29, Python 3.10
- 70 unit/integration/fault-injection/acceptance tests + 13 skill black-box tests
- Live fault injection: `docker kill` → honest unhealthy → modeld recovery in ~70 s
- Dojo evaluation: `ty1200-dojo` executes the declared scenarios (8/8 pass on-device)
- 24h availability: `rosclaw-ty1200-health.timer` samples every 5 min;
  `ty1200-availability --hours 24` evaluates the ≥99.5% gate

## Docs

- [Peripheral enablement runbook](docs/runbook-peripherals.md) — CAN driver,
  EtherCAT (IgH/SOEM), GPIO pinctrl, SPI, PREEMPT_RT, 5G, NPU/OpenVINO, ATSHA204A

## Notes for deployment at your site

- `providers/deepseekv4` and `providers/qwen3.6-27b` contain **site-specific
  endpoints** (`[SITE-LOCAL-ADDR]:*`) used by the origin deployment — replace with
  your own OpenAI-compatible services or delete these two provider packs.
- Image references point at the vendor Harbor registry; digests are pinned in
  `deployment/env/ty1200-models.env.example`.
- Model weights are **not** distributed (Cosmos: NVIDIA Open Model License;
  Qwen3-Embedding: Apache-2.0).

## License

MIT (code). Model weights are covered by their own licenses and are not
redistributed.
