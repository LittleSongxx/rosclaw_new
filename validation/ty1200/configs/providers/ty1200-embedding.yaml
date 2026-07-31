name: ty1200_qwen3_embedding_06b
version: "0.1.0"
type: embedding

description: >
  On-device Qwen3-Embedding-0.6B served by vLLM on the TY1200 GPGPU.
  OpenAI-compatible /embeddings endpoint. 1024-dim output.

capabilities:
  - embedding.text
  - memory.embed
  - knowledge.embed

modalities:
  input: [text]
  output: [embedding]

runtime:
  backend: openai_compatible
  protocol: http
  endpoint: http://127.0.0.1:8000/v1
  device: local_gpgpu
  env:
    api_kind: embeddings
    model: qwen3-embedding-0.6b
    # vLLM serves the path-style id when no --served-model-name is set;
    # OpenAICompatRuntime retries with this id on HTTP 404.
    model_fallback: /models/Qwen/Qwen3-Embedding-0.6B
    timeout_sec: "60"
    retries: "1"

safety:
  executable: false
  requires_guard: false

observability:
  log_inputs: false
  log_outputs: false
  trace_level: standard

data_policy:
  allow_text: true
  allow_images: false
  allow_video: false
  allow_raw_robot_logs: false
  allow_credentials: false
