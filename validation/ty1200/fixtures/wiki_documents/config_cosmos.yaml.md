name: ty1200_cosmos_reason2_2b
version: "0.1.0"
type: vlm

description: >
  On-device Cosmos-Reason2-2B served by vLLM on the TY1200 GPGPU.
  OpenAI-compatible /chat/completions endpoint. Physical-reasoning VLM;
  advisory only - can suggest or veto, never actuates.

capabilities:
  - vlm.physical_reasoning
  - critic.action_safety
  - critic.failure_analysis

modalities:
  input: [text, image, video]
  output: [text]

runtime:
  backend: openai_compatible
  protocol: http
  endpoint: http://127.0.0.1:8001/v1
  device: local_gpgpu
  env:
    api_kind: chat_completions
    model: cosmos-reason2-2b
    model_fallback: /models/nv-community/Cosmos-Reason2-2B
    timeout_sec: "120"
    retries: "1"

safety:
  executable: false
  requires_guard: true

observability:
  log_inputs: false
  log_outputs: true
  trace_level: standard
  redact_reasoning: true

data_policy:
  allow_text: true
  allow_images: true   # on-device only; media never leaves the box
  allow_video: true
  allow_raw_robot_logs: false
  allow_credentials: false
