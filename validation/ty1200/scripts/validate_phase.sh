#!/usr/bin/env bash
# Run one validation phase and append results to the run's module matrix.
# Usage: validate_phase.sh <phase> [report_dir]
# Phases: baseline | providers | runtime | trace | practice | seekdb | wiki | memory | sandbox | faults
set -uo pipefail

PHASE="${1:?usage: validate_phase.sh <phase> [report_dir]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ENV="validation/ty1200/reports/current_run.env"
RUN_ID="$(cut -d= -f2 "$RUN_ENV")"
REPORT_DIR="${2:-validation/ty1200/reports/$RUN_ID}"
mkdir -p "$REPORT_DIR"

# Ports: site-local copy wins, template otherwise.
if [[ -f validation/ty1200/configs/ports.env ]]; then
  set -a; source validation/ty1200/configs/ports.env; set +a
else
  set -a; source validation/ty1200/configs/ports.env.example; set +a
fi

PY=.venv/bin/python
LOG="$REPORT_DIR/commands.log"
MATRIX="$REPORT_DIR/module_matrix.json"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

record() { # record <module> <status> <level> <detail>
  "$PY" - "$MATRIX" "$1" "$2" "$3" "$4" <<'PY'
import json, sys, datetime
path, module, status, level, detail = sys.argv[1:6]
try:
    data = json.load(open(path))
except Exception:
    data = {"modules": {}}
data["modules"][module] = {
    "status": status, "level": level, "detail": detail,
    "at": datetime.datetime.now().astimezone().isoformat(),
}
json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)
PY
}

case "$PHASE" in
  baseline)
    log "phase=baseline compileall+ruff(ci scope)+mypy"
    "$PY" -m compileall -q src tests >>"$LOG" 2>&1 \
      && .venv/bin/ruff check src tests >>"$LOG" 2>&1 \
      && "$PY" -m mypy src/rosclaw >>"$LOG" 2>&1 \
      && record M01_install_doctor PASS V1 "compileall/ruff/mypy clean" \
      || record M01_install_doctor FAIL V0 "see commands.log"
    ;;
  providers)
    log "phase=providers benchmark x3"
    ok=0
    for spec in \
      "ty1200_qwen3_embedding_06b|http://127.0.0.1:${TY1200_EMBEDDING_PORT}/v1|/models/Qwen/Qwen3-Embedding-0.6B|embeddings" \
      "ty1200_cosmos_reason2_2b|http://127.0.0.1:${TY1200_COSMOS_PORT}/v1|/models/nv-community/Cosmos-Reason2-2B|chat" \
      "site_deepseekv4|${TY1200_DEEPSEEK_ENDPOINT}|deepseekv4|chat"; do
      IFS='|' read -r name endpoint model kind <<<"$spec"
      if [[ "$endpoint" == *__SITE_LOCAL_HOST__* ]]; then
        log "SKIP $name: site endpoint not configured"; continue
      fi
      "$PY" validation/ty1200/benchmarks/provider_benchmark.py \
        --name "$name" --endpoint "$endpoint" --model "$model" --kind "$kind" \
        --out "$REPORT_DIR/metrics/provider_${name}.json" >>"$LOG" 2>&1 \
        && ok=$((ok+1)) || record M07_provider FAIL V2 "$name benchmark failed"
    done
    [[ $ok -eq 3 ]] && record M07_provider PASS V5 "3 providers benchmarked" || true
    ;;
  runtime)
    log "phase=runtime pytest runtime/eventbus/daemon suites"
    "$PY" -m pytest -q -p no:cacheprovider \
      tests/runtime tests/daemon tests/kernel -x --tb=short >>"$LOG" 2>&1 \
      && record M02_runtime PASS V2 "runtime/daemon/kernel suites green" \
      || record M02_runtime FAIL V2 "see commands.log"
    ;;
  trace)
    log "phase=trace"
    "$PY" -m pytest -q -p no:cacheprovider tests -k 'trace' --tb=short >>"$LOG" 2>&1 \
      && record M13_trace PASS V1 "trace suites green" \
      || record M13_trace FAIL V1 "see commands.log"
    ;;
  practice)
    log "phase=practice fixture loop"
    PRACTICE_ROOT="$(mktemp -d /tmp/ty1200-practice.XXXXXX)"
    export ROSCLAW_HOME="$PRACTICE_ROOT/home"
    .venv/bin/rosclaw practice record --fixture tests/fixtures/practice/rh56_minimal_loop.json \
      --out "$PRACTICE_ROOT/practice" --json >>"$LOG" 2>&1 \
      && .venv/bin/rosclaw practice verify practice_rh56_minimal_loop \
        --data-root "$PRACTICE_ROOT/practice" --strict --json >>"$LOG" 2>&1 \
      && .venv/bin/rosclaw practice distill practice_rh56_minimal_loop \
        --data-root "$PRACTICE_ROOT/practice" --json >>"$LOG" 2>&1 \
      && record M14_practice PASS V2 "record/verify/distill fixture loop" \
      || record M14_practice FAIL V1 "see commands.log"
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

log "phase=$PHASE done"
cat "$MATRIX" 2>/dev/null || true
